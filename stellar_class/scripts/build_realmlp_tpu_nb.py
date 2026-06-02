"""Builds notebooks/stellar-realmlp-tpu.ipynb (run from project root).

Self-contained, shareable Kaggle TPU notebook: trains RealMLP-TD across 8 seeds in
PARALLEL (one per TPU core via jax.pmap), averages them, writes OOF/test probs +
submission. Embeds the validated model block from src/train_realmlp.py verbatim so the
notebook can't drift from the script; the pmap trainer is validated on 8 simulated CPU
devices.
"""
import nbformat as nbf

# --- slice the validated model block (CONFIG..smooth_ce) out of the script ---------
src = open("src/train_realmlp.py").read().splitlines()
start = next(i for i, l in enumerate(src) if l.startswith("PI = math.pi"))
end = next(i for i, l in enumerate(src) if l.strip() == "return (per * w).sum() / w.sum()")
MODEL_BLOCK = "\n".join(src[start:end + 1])

nb = nbf.v4.new_notebook()
cells = []
C = lambda s: cells.append(nbf.v4.new_code_cell(s))
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))

M("""# 🌌 RealMLP-TD × 8 seeds in parallel (Flax NNX, TPU)

A from-scratch **Flax NNX** port of RealMLP-TD — a strong tabular MLP (tuned defaults:
PBLD periodic numerical embeddings, NTK-parametrized linears, an internal ensemble, and
scheduled lr / dropout / label-smoothing).

Seeds are independent, so this trains **8 seeds at once — one per TPU core** via
`jax.pmap`. On a v5e-8 that means an 8-model ensemble for roughly the wall-clock of a
single model. The seeds are averaged into a lower-variance model.

Outputs: `submission.csv`, plus out-of-fold and test class probabilities
(`oof_realmlp_ens.npy`, `test_realmlp_ens.npy`) — handy as a member for stacking.

**Kaggle setup:** accelerator **TPU VM v5e-8**, Internet **On** (to pip-install
flax/optax), and add the `playground-series-s6e6` competition data.""")

M("## 1. Install")
C("""import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flax", "optax"], check=True)
print("installed flax + optax")""")

M("## 2. Imports")
C("""import math
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

print("jax", jax.__version__, "| devices:", jax.devices())
N_SEEDS = jax.device_count()   # one seed per core
print("training", N_SEEDS, "seeds in parallel")""")

M("## 3. RealMLP-TD model")
C(MODEL_BLOCK)

M("## 4. Parallel (pmap-over-seeds) trainer\n\n"
  "The whole training run is a pure function — batches via `lax.scan`, best-epoch "
  "selection carried functionally — so `jax.pmap` maps one seed per device. Each seed "
  "gets its own RNG (init, dropout, data shuffle). Evaluation is batched to bound memory.")
C('''def make_tx(params, cfg, total_steps):
    p = cfg
    def lr(mult):
        return lambda step: p["lr"] * mult * sched_factor(step / total_steps, p["lr_sched"], p["flat_ratio"])
    def aw(mult, wd):
        return optax.adamw(learning_rate=lr(mult), b1=p["mom"], b2=p["sq_mom"], weight_decay=wd)
    transforms = {
        "scale": aw(p["lr_scale_mult"], p["weight_decay"] * p["wd_scale_mult"]),
        "pbld": aw(p["pbld_lr_factor"], p["weight_decay"]),
        "first_w": aw(p["first_layer_lr_factor"], p["weight_decay"] * p["first_layer_wd_factor"]),
        "other_w": aw(1.0, p["weight_decay"]),
        "bias": aw(p["lr_bias_mult"], p["weight_decay"] * p["wd_bias_mult"]),
    }
    labels = jax.tree_util.tree_map_with_path(lambda path, _: label_for(path), params)
    return optax.chain(optax.clip_by_global_norm(p["grad_clip"]),
                       optax.multi_transform(transforms, labels))


def bal_acc(y, pred, C):
    correct = (pred == y).astype(jnp.float32)
    tp = jnp.zeros(C).at[y].add(correct)
    sup = jnp.zeros(C).at[y].add(1.0)
    return (tp / jnp.maximum(sup, 1.0)).mean()


def build_train_fn(graphdef, tx, cfg, n_train, n_batches, n_classes):
    bs, epochs = cfg["train_bs"], cfg["epochs"]
    total_steps = epochs * n_batches
    n_ens, eval_bs = cfg["n_ens"], cfg["eval_bs"]

    def apply(params, xn, xc, drop, key, train):
        return nnx.merge(graphdef, params)(xn, xc, drop, key, train)

    def loss_fn(params, xn, xc, y, ls, drop, key, cw):
        probs = apply(params, xn, xc, drop, key, True)
        return smooth_ce(jnp.repeat(y, n_ens), probs.reshape(-1, n_classes), ls, cw)

    def predict(params, xn, xc):                       # batched (scan) -> bounds memory
        n = xn.shape[0]; nb = -(-n // eval_bs)
        xn = jnp.pad(xn, ((0, nb * eval_bs - n), (0, 0)))
        xc = jnp.pad(xc, ((0, nb * eval_bs - n), (0, 0)))
        def f(_, b):
            s = b * eval_bs
            cn = jax.lax.dynamic_slice_in_dim(xn, s, eval_bs, 0)
            cc = jax.lax.dynamic_slice_in_dim(xc, s, eval_bs, 0)
            return None, apply(params, cn, cc, 0.0, jax.random.key(0), False).mean(1)
        _, outs = jax.lax.scan(f, None, jnp.arange(nb))
        return outs.reshape(nb * eval_bs, -1)[:n]

    def train_fn(params, key, Xtr_n, Xtr_c, ytr, Xva_n, Xva_c, yva, Xte_n, Xte_c, cw):
        opt_state = tx.init(params)
        best_score = jnp.asarray(-1.0); best_params = params
        for epoch in range(epochs):
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, n_train)[:n_batches * bs].reshape(n_batches, bs)
            def step(carry, b):
                params, opt_state, key = carry
                step_i = epoch * n_batches + b
                progress = step_i / total_steps
                ls = cfg["ls_eps"] * sched_factor(progress, cfg["ls_eps_sched"], cfg["flat_ratio"])
                drop = cfg["dropout"] * sched_factor(progress, cfg["p_drop_sched"], cfg["flat_ratio"])
                key, kd = jax.random.split(key)
                idx = perm[b]
                g = jax.grad(loss_fn)(params, Xtr_n[idx], Xtr_c[idx], ytr[idx], ls, drop, kd, cw)
                updates, opt_state = tx.update(g, opt_state, params)
                return (params := optax.apply_updates(params, updates), opt_state, key), None
            (params, opt_state, key), _ = jax.lax.scan(step, (params, opt_state, key), jnp.arange(n_batches))
            score = bal_acc(yva, predict(params, Xva_n, Xva_c).argmax(1), n_classes)
            improved = score > best_score
            best_score = jnp.where(improved, score, best_score)
            best_params = jax.tree.map(lambda bp, p: jnp.where(improved, p, bp), best_params, params)
        return predict(best_params, Xva_n, Xva_c), predict(best_params, Xte_n, Xte_c), best_score

    return train_fn''')

M("## 5. Data, folds, feature engineering")
C('''from pathlib import Path
_k = Path("/kaggle/input")
_hits = sorted(_k.rglob("train.csv")) if _k.exists() else []
DATA = _hits[0].parent if _hits else Path("../data")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print("data dir:", DATA)

train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")
classes = sorted(train_df["class"].unique())
cmap_y = {c: i for i, c in enumerate(classes)}
y = train_df["class"].map(cmap_y).to_numpy()
n_classes = len(classes)

# Fixed stratified 5-fold split (shuffle, seed 42): reproducible OOF, reusable for stacking.
folds = np.zeros(len(y), dtype=int)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    folds[va] = i

raw_cat = ["spectral_type", "galaxy_population"]
raw_num = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
state = {}
Xtr, new_cat, new_num, combos = feature_engineering(train_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=True)
Xte, *_ = feature_engineering(test_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=False)
cat_cols = sorted(raw_cat + new_cat); num_cols = raw_num + new_num
Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1); Xte = Xte.reindex(sorted(Xte.columns), axis=1)
cat_dims = [int(max(Xtr[c].max(), Xte[c].max()) + 1) for c in cat_cols]
print("train", Xtr.shape, "test", Xte.shape)


def tune_class_weights(y_true, proba, n_rounds=3, grid=None):
    if grid is None:
        grid = np.linspace(0.2, 3.0, 29)
    w = np.ones(proba.shape[1]); best = balanced_accuracy_score(y_true, proba.argmax(1))
    for _ in range(n_rounds):
        for k in range(len(w)):
            bw = w[k]
            for g in grid:
                w[k] = g
                s = balanced_accuracy_score(y_true, (proba * w).argmax(1))
                if s > best: best, bw = s, g
            w[k] = bw
    return w / w.mean(), best''')

M("## 6. Train (8 seeds in parallel) over the 5 folds")
C('''cfg = dict(CONFIG)
bs = cfg["train_bs"]
oof = np.zeros((len(Xtr), n_classes))      # seed-averaged out-of-fold
test_probs = np.zeros((len(Xte), n_classes))

for fold in range(5):
    tr = np.where(folds != fold)[0]; va = np.where(folds == fold)[0]
    print(f"=== fold {fold} ===", flush=True)
    enc = make_target_encoder(42)
    tr_te = enc.fit_transform(Xtr.iloc[tr][combos], y[tr]).astype(np.float32)
    va_te = enc.transform(Xtr.iloc[va][combos]).astype(np.float32)
    te_te = enc.transform(Xte[combos]).astype(np.float32)
    pre = NumericalPreprocessor(cfg["tfms"]).fit(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
    Xtr_n = pre.transform(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
    Xva_n = pre.transform(np.hstack([Xtr.iloc[va][num_cols].values.astype(np.float32), va_te]))
    Xte_n = pre.transform(np.hstack([Xte[num_cols].values.astype(np.float32), te_te]))
    clip = np.array(cat_dims) - 1
    Xtr_c = np.clip(Xtr.iloc[tr][cat_cols].values.astype(np.int64), 0, clip)
    Xva_c = np.clip(Xtr.iloc[va][cat_cols].values.astype(np.int64), 0, clip)
    Xte_c = np.clip(Xte[cat_cols].values.astype(np.int64), 0, clip)
    ytr, yva = y[tr], y[va]
    cw = jnp.asarray(compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr), jnp.float32)

    n_train = (len(ytr) // bs) * bs
    n_batches = n_train // bs

    # one differently-seeded initial model per device (leading axis = N_SEEDS)
    @nnx.vmap(in_axes=0)
    def create(seed_key):
        return RealMLP(n_classes, cat_dims, Xtr_n.shape[1], cfg, rngs=nnx.Rngs(params=seed_key))
    graphdef, params8 = nnx.split(create(jax.random.split(jax.random.key(100 + fold), N_SEEDS)))
    params0 = jax.tree.map(lambda x: x[0], params8)
    tx = make_tx(params0, cfg, n_batches * cfg["epochs"])
    train_fn = build_train_fn(graphdef, tx, cfg, n_train, n_batches, n_classes)

    pm = jax.pmap(train_fn, in_axes=(0, 0) + (None,) * 9)
    tkeys = jax.random.split(jax.random.key(200 + fold), N_SEEDS)
    oof8, test8, scores = pm(params8, tkeys,
        jnp.asarray(Xtr_n), jnp.asarray(Xtr_c), jnp.asarray(ytr),
        jnp.asarray(Xva_n), jnp.asarray(Xva_c), jnp.asarray(yva),
        jnp.asarray(Xte_n), jnp.asarray(Xte_c), cw)
    print(f"  per-seed val bal_acc: {np.asarray(scores).round(4)}", flush=True)
    oof[va] = np.asarray(oof8).mean(0)
    test_probs += np.asarray(test8).mean(0) / 5''')

M("## 7. Evaluate, save, submit")
C('''raw = balanced_accuracy_score(y, oof.argmax(1))
weights, tuned = tune_class_weights(y, oof)
print(f"{N_SEEDS}-seed ensemble OOF: raw balanced accuracy={raw:.5f}  tuned={tuned:.5f}")
print(f"class multipliers: {dict(zip(classes, weights.round(3)))}")

np.save(OUT / "oof_realmlp_ens.npy", oof)
np.save(OUT / "test_realmlp_ens.npy", test_probs)
preds = np.asarray(classes)[(test_probs * weights).argmax(1)]
pd.DataFrame({"id": test_df["id"], "class": preds}).to_csv(OUT / "submission.csv", index=False)
print("wrote submission.csv, oof_realmlp_ens.npy, test_realmlp_ens.npy")''')

M("""## Outputs

- **`submission.csv`** — RealMLP 8-seed ensemble predictions.
- **`oof_realmlp_ens.npy`** / **`test_realmlp_ens.npy`** — out-of-fold and test class
  probabilities (columns sorted GALAXY/QSO/STAR, rows in train/test order), a drop-in
  member for a stacking ensemble.

If this notebook helped, an upvote is appreciated 🙂""")

nb.cells = cells
nbf.write(nb, "notebooks/stellar-realmlp-tpu.ipynb")
print(f"wrote notebooks/stellar-realmlp-tpu.ipynb ({len(cells)} cells)")
