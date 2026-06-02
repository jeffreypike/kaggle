"""RealMLP-TD in Flax NNX — the top single model for this competition (~0.97 solo).

Faithful JAX/NNX port of the from-scratch PyTorch RealMLP-TD reference
(yekenot/ps-s6-e6-realmlp-pytorch): NTK-parametrized linears, PBLD periodic numerical
embeddings, per-ensemble categorical one-hot/embeddings, a front ScalingLayer, an
internal n_ens ensemble, 5 parameter groups with distinct lr/wd, and per-step lr
(flat_cos) / dropout (expm4t) / label-smoothing (cos) schedules.

Device-agnostic: same code runs on CPU (smoke test), GPU (3080), or Kaggle TPU.
Produces oof_realmlp.npy on the standardized folds + submission_realmlp.csv.

    python stellar_class/src/train_realmlp.py                 # full run
    python stellar_class/src/train_realmlp.py --smoke         # tiny CPU sanity check
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import (load_data_with_folds, get_custom_cv, evaluate_predictions,
                        save_oof_predictions, save_submission, tune_class_weights, DATA_DIR, PREDICTIONS_DIR)

PI = math.pi

CONFIG = dict(
    n_ens=8, embed_dim=7, onehot_thresh=10, hidden_dims=[512, 512, 512],
    dropout=0.05, p_drop_sched="expm4t", add_front_scale=True,
    pbld_hidden_dim=20, pbld_out_dim=5, pbld_freq_scale=5.0, pbld_lr_factor=0.093,
    lr=0.01, mom=0.9, sq_mom=0.98, lr_sched="flat_cos", flat_ratio=0.3,
    first_layer_lr_factor=1.0, first_layer_wd_factor=0.1, lr_scale_mult=10.0,
    lr_bias_mult=0.1, weight_decay=0.013, wd_scale_mult=0.1, wd_bias_mult=0.5,
    grad_clip=1.0, ls_eps=0.04, ls_eps_sched="cos",
    tfms=["median_center", "robust_scale"],
    epochs=6, train_bs=256, eval_bs=10240, seed=42,
)


# ── Feature engineering (faithful port of the reference cell) ──────────────────
COLOR_PAIRS = [("u", "g"), ("u", "r")]
IMPORTANT_COMBOS = sorted([("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")])


def feature_engineering(df, cat_cols, num_cols, cmap, fit=False):
    df = df.copy()
    df["_g_/_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).astype("float32")
    df["_i_/_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")

    for col in cat_cols:                                   # string categoricals
        if fit:
            codes, uniq = df[col].factorize(); cmap[col] = uniq
        else:
            code_map = {c: i for i, c in enumerate(cmap[col])}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[col] = codes.astype("int32")

    for col in num_cols:                                   # floor-bucketed numerics
        name = f"{col}_cat_"
        if fit:
            codes, uniq = np.floor(df[col]).factorize(); cmap[col] = uniq
        else:
            code_map = {c: i for i, c in enumerate(cmap[col])}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[name] = codes.astype("int32")

    for col, bins_list in {"delta": [100, 500]}.items():   # quantile bins
        for n_bins in bins_list:
            name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                      strategy="quantile", subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                cmap[name] = kb
            else:
                binned = cmap[name].transform(df[[col]]).ravel().astype("int32")
            df[name] = binned

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        name = "_".join(cols) + "_"
        combo_names.append(name)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            codes, uniq = pd.factorize(s, sort=False); cmap[name] = uniq
        else:
            code_map = {c: i for i, c in enumerate(cmap[name])}
            codes = s.map(code_map).fillna(-1).astype("int32")
        df[name] = codes.astype("int32")

    new_cat = [c for c in df.columns if c.endswith("_")]
    new_num = [c for c in df.columns if c.startswith("_")]
    return df, new_cat, new_num, combo_names


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in
                      ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X, y=None):
        if {"median_center", "robust_scale"} & set(self._tfms):
            self._median = np.median(X, axis=0)
            q = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            z = q == 0.0
            q[z] = 0.5 * (X.max(axis=0)[z] - X.min(axis=0)[z])
            self._iqr = 1.0 / (q + 1e-30)
            self._iqr[q == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for t in self._tfms:
            if t == "median_center":
                X -= self._median[None, :]
            elif t == "robust_scale":
                X *= self._iqr[None, :]
            elif t == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif t == "l2_normalize":
                n = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(n == 0, 1.0, n)
        return X


# ── NNX model components ───────────────────────────────────────────────────────
class PReLU(nnx.Module):
    def __init__(self, init=0.25):
        self.a = nnx.Param(jnp.asarray(init, jnp.float32))

    def __call__(self, x):
        return jnp.where(x >= 0, x, self.a[...] * x)


class NTPLinear(nnx.Module):
    def __init__(self, n_ens, in_f, out_f, *, rngs, bias=True):
        self.in_f = in_f
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (n_ens, in_f, out_f)))
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (n_ens, out_f))) if bias else None

    def __call__(self, x):
        x = jnp.einsum("bki,kio->bko", x, self.weight[...]) / jnp.sqrt(self.in_f)
        if self.bias is not None:
            x = x + self.bias[...][None]
        return x


class ScalingLayer(nnx.Module):
    def __init__(self, n_ens, n_features):
        self.scale = nnx.Param(jnp.ones((n_ens, n_features)))

    def __call__(self, x):
        return x * self.scale[...][None]


class PBLDEmbedding(nnx.Module):
    def __init__(self, n_ens, n_features, hidden, out_dim, freq_scale, *, rngs):
        self.out_dim = out_dim
        self.w1 = nnx.Param(jax.random.normal(rngs.params(), (n_ens, n_features, hidden)) * freq_scale)
        self.b1 = nnx.Param(jax.random.uniform(rngs.params(), (n_ens, n_features, hidden), minval=-PI, maxval=PI))
        self.w2 = nnx.Param(jax.random.normal(rngs.params(), (n_ens, n_features, hidden, out_dim - 1)) / math.sqrt(hidden))
        self.b2 = nnx.Param(jnp.zeros((n_ens, n_features, out_dim - 1)))
        self.act = PReLU()

    def __call__(self, x):  # (batch, n_ens, n_features)
        periodic = jnp.cos(2 * PI * (x[..., None] * self.w1[...][None] + self.b1[...][None]))
        transformed = self.act(jnp.einsum("bkfh,kfhd->bkfd", periodic, self.w2[...]) + self.b2[...][None])
        feat = jnp.concatenate([x[..., None], transformed], axis=-1)
        return feat.reshape(x.shape[0], x.shape[1], -1)


class CategoricalFeatureLayer(nnx.Module):
    """Per-ensemble one-hot (dim<=thresh) or embedding (dim>thresh) for categoricals."""
    def __init__(self, n_ens, cat_dims, embed_dim, onehot_thresh, *, rngs):
        self.n_ens = n_ens
        self.oh_idx = [i for i, d in enumerate(cat_dims) if d <= onehot_thresh]
        self.oh_dims = [cat_dims[i] for i in self.oh_idx]
        self.em_idx = [i for i, d in enumerate(cat_dims) if d > onehot_thresh]
        self.embeds = nnx.List([
            nnx.Param(jax.random.normal(rngs.params(), (n_ens, cat_dims[i], embed_dim)))
            for i in self.em_idx
        ])

    def __call__(self, x):  # x: (batch, n_ens, n_cat) int
        feats = []
        for j, i in enumerate(self.oh_idx):
            feats.append(jax.nn.one_hot(x[:, :, i], self.oh_dims[j], dtype=jnp.float32))
        kk = jnp.arange(self.n_ens)[None, :]               # (1, n_ens)
        for emb, i in zip(self.embeds, self.em_idx):
            feats.append(emb[...][kk, x[:, :, i]])         # (batch, n_ens, embed_dim)
        return jnp.concatenate(feats, axis=2) if feats else jnp.zeros((x.shape[0], self.n_ens, 0))


class RealMLP(nnx.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg, *, rngs):
        n_ens = cfg["n_ens"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens, cat_dims, cfg["embed_dim"], cfg["onehot_thresh"], rngs=rngs)
        self.num_embed = PBLDEmbedding(n_ens, n_numerical, cfg["pbld_hidden_dim"],
                                       cfg["pbld_out_dim"], cfg["pbld_freq_scale"], rngs=rngs)
        num_dim = n_numerical * cfg["pbld_out_dim"]
        cat_dim = sum(d if d <= cfg["onehot_thresh"] else cfg["embed_dim"] for d in cat_dims)
        total = num_dim + cat_dim
        self.scale = ScalingLayer(n_ens, total) if cfg["add_front_scale"] else None
        self.linears = nnx.List([])
        in_dim = total
        for k, h in enumerate(cfg["hidden_dims"]):
            self.linears.append(NTPLinear(n_ens, in_dim, h, rngs=rngs))
            in_dim = h
        self.out = NTPLinear(n_ens, in_dim, output_dim, rngs=rngs)

    def __call__(self, x_num, x_cat, drop_rate, key, train):
        x_num = jnp.broadcast_to(x_num[:, None], (x_num.shape[0], self.n_ens, x_num.shape[1]))
        x_cat = jnp.broadcast_to(x_cat[:, None], (x_cat.shape[0], self.n_ens, x_cat.shape[1]))
        x = jnp.concatenate([self.num_embed(x_num), self.cate(x_cat)], axis=2)
        if self.scale is not None:
            x = self.scale(x)
        for i, lin in enumerate(self.linears):
            x = jax.nn.silu(lin(x))
            if train:
                key, sub = jax.random.split(key)
                mask = (jax.random.uniform(sub, x.shape) >= drop_rate).astype(x.dtype)
                x = x * mask / jnp.maximum(1.0 - drop_rate, 1e-6)
        return jax.nn.softmax(self.out(x), axis=2)          # (batch, n_ens, C)


# ── Schedules / optimizer / loss ──────────────────────────────────────────────
def sched_factor(progress, sched, flat_ratio):
    if sched == "constant":
        return jnp.ones_like(progress)
    if sched == "cos":
        return (jnp.cos(PI * progress) + 1) / 2
    if sched == "flat_cos":
        t = jnp.clip((progress - flat_ratio) / (1 - flat_ratio), 0.0, 1.0)
        return jnp.where(progress < flat_ratio, 1.0, (jnp.cos(PI * t) + 1) / 2)
    if sched == "expm4t":
        return jnp.exp(-4 * progress)
    raise ValueError(sched)


def _key_str(k):
    for attr in ("name", "key", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def label_for(path):
    name = "/".join(_key_str(p) for p in path)
    if "num_embed" in name:
        return "pbld"
    if "scale" in name:
        return "scale"
    if "bias" in name:
        return "bias"
    if "linears/0/weight" in name:
        return "first_w"
    return "other_w"


def build_optimizer(model, cfg, total_steps):
    p = cfg
    def lr(mult):
        return lambda step: p["lr"] * mult * sched_factor(step / total_steps, p["lr_sched"], p["flat_ratio"])
    def aw(mult, wd):
        return optax.adamw(learning_rate=lr(mult), b1=p["mom"], b2=p["sq_mom"], weight_decay=wd)
    transforms = {
        "scale":   aw(p["lr_scale_mult"], p["weight_decay"] * p["wd_scale_mult"]),
        "pbld":    aw(p["pbld_lr_factor"], p["weight_decay"]),
        "first_w": aw(p["first_layer_lr_factor"], p["weight_decay"] * p["first_layer_wd_factor"]),
        "other_w": aw(1.0, p["weight_decay"]),
        "bias":    aw(p["lr_bias_mult"], p["weight_decay"] * p["wd_bias_mult"]),
    }
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(lambda path, _: label_for(path), params)
    tx = optax.chain(optax.clip_by_global_norm(p["grad_clip"]),
                     optax.multi_transform(transforms, label_fn))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def smooth_ce(y_true, probs, ls, class_w):  # probs: (N, C)
    C = probs.shape[1]
    y = jnp.full(probs.shape, ls / C).at[jnp.arange(probs.shape[0]), y_true].set(1.0 - ls + ls / C)
    per = -(y * jnp.log(jnp.clip(probs, 1e-15, 1.0))).sum(1)
    w = class_w[y_true]
    return (per * w).sum() / w.sum()


@nnx.jit(static_argnames=("n_ens", "n_classes"))
def train_step(model, optimizer, x_num, x_cat, y, class_w, ls, drop_rate, key, n_ens, n_classes):
    def loss_fn(m):
        probs = m(x_num, x_cat, drop_rate, key, True)           # (b, n_ens, C)
        return smooth_ce(jnp.repeat(y, n_ens), probs.reshape(-1, n_classes), ls, class_w)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def predict_batch(model, x_num, x_cat):
    key = jax.random.key(0)
    return model(x_num, x_cat, 0.0, key, False).mean(axis=1)    # (b, C)


def predict_all(model, Xn, Xc, eval_bs):
    return np.concatenate([
        np.asarray(predict_batch(model, jnp.asarray(Xn[s:s + eval_bs]), jnp.asarray(Xc[s:s + eval_bs])))
        for s in range(0, len(Xn), eval_bs)], axis=0)


def fit_fold(Xtr_n, Xtr_c, ytr, Xva_n, Xva_c, yva, cat_dims, n_classes, cfg):
    rngs = nnx.Rngs(params=cfg["seed"], dropout=cfg["seed"] + 1)
    model = RealMLP(n_classes, cat_dims, Xtr_n.shape[1], cfg, rngs=rngs)
    total_steps = cfg["epochs"] * len(ytr)
    optimizer = build_optimizer(model, cfg, total_steps)
    class_w = jnp.asarray(compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr), jnp.float32)

    Xtr_n, Xtr_c = jnp.asarray(Xtr_n), jnp.asarray(Xtr_c)
    ytr_j = jnp.asarray(ytr)
    key = jax.random.key(cfg["seed"])
    order = np.arange(len(ytr))
    best_score, best_probs, best_state = -np.inf, None, None
    bs = cfg["train_bs"]
    for epoch in range(cfg["epochs"]):
        for start in range(0, len(ytr), bs):
            progress = (epoch * len(ytr) + start) / total_steps
            ls = float(cfg["ls_eps"] * sched_factor(jnp.asarray(progress), cfg["ls_eps_sched"], cfg["flat_ratio"]))
            dr = float(cfg["dropout"] * sched_factor(jnp.asarray(progress), cfg["p_drop_sched"], cfg["flat_ratio"]))
            idx = order[start:start + bs]
            key, sub = jax.random.split(key)
            train_step(model, optimizer, Xtr_n[idx], Xtr_c[idx], ytr_j[idx], class_w,
                       jnp.asarray(ls), jnp.asarray(dr), sub, cfg["n_ens"], n_classes)
        np.random.shuffle(order)
        probs = predict_all(model, Xva_n, Xva_c, cfg["eval_bs"])
        score = balanced_accuracy_score(yva, probs.argmax(1))
        if score > best_score:
            best_score, best_probs = score, probs
            best_state = jax.tree.map(jnp.copy, nnx.state(model))   # snapshot best-epoch weights
        print(f"    epoch {epoch+1}/{cfg['epochs']}  val_bal_acc={score:.5f}  best={best_score:.5f}", flush=True)
    nnx.update(model, best_state)                                   # restore best for test prediction
    return best_probs, best_score, model


def main(smoke=False, rows=6000, epochs=1, seed=None):
    cfg = dict(CONFIG)
    sfx = ""
    if seed is not None:
        cfg["seed"] = seed
        sfx = f"_s{seed}"   # suffix outputs so multiple seeds don't clobber (for seed-ensembling)
    train_df = load_data_with_folds().to_pandas()
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(train_df["class"].unique())
    cmap_y = {c: i for i, c in enumerate(classes)}
    y = train_df["class"].map(cmap_y).to_numpy()
    folds = train_df["fold"].to_numpy()

    raw_cat = ["spectral_type", "galaxy_population"]
    raw_num = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    state = {}
    Xtr, new_cat, new_num, combos = feature_engineering(train_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=True)
    Xte, _, _, _ = feature_engineering(test_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=False)
    cat_cols = sorted(raw_cat + new_cat)
    num_cols = raw_num + new_num
    Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1)
    Xte = Xte.reindex(sorted(Xte.columns), axis=1)

    if smoke:
        cfg.update(n_ens=4, hidden_dims=[256, 256], epochs=epochs, train_bs=1024)
        keep = np.random.RandomState(0).choice(len(Xtr), rows, replace=False)
        Xtr, y, folds = Xtr.iloc[keep].reset_index(drop=True), y[keep], folds[keep]
        Xtr["fold"] = folds  # not used directly; folds array drives splits

    n_classes = len(classes)
    cv = get_custom_cv(load_data_with_folds()) if not smoke else [
        (np.where(folds != f)[0], np.where(folds == f)[0]) for f in range(5)]

    # consistent categorical dims across train+test (max code + 1)
    cat_dims = [int(max(Xtr[c].max(), Xte[c].max()) + 1) for c in cat_cols]

    oof = np.zeros((len(Xtr), n_classes))
    test_probs = np.zeros((len(Xte), n_classes))
    for fold, (tr, va) in enumerate(cv):
        print(f"=== fold {fold} ===", flush=True)
        # Per-fold target encoding of the categorical combos -> extra numeric features
        # (fit on this fold's train only; no leakage). Matches the reference's TE=True.
        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=cfg["seed"])
        tr_te = enc.fit_transform(Xtr.iloc[tr][combos], y[tr]).astype(np.float32)
        va_te = enc.transform(Xtr.iloc[va][combos]).astype(np.float32)
        te_te = enc.transform(Xte[combos]).astype(np.float32)
        num_tr = np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te])
        num_va = np.hstack([Xtr.iloc[va][num_cols].values.astype(np.float32), va_te])
        num_te = np.hstack([Xte[num_cols].values.astype(np.float32), te_te])
        pre = NumericalPreprocessor(cfg["tfms"]).fit(num_tr)
        Xtr_n, Xva_n, Xte_n = pre.transform(num_tr), pre.transform(num_va), pre.transform(num_te)
        clip = np.array(cat_dims) - 1
        Xtr_c = np.clip(Xtr.iloc[tr][cat_cols].values.astype(np.int64), 0, clip)
        Xva_c = np.clip(Xtr.iloc[va][cat_cols].values.astype(np.int64), 0, clip)
        Xte_c = np.clip(Xte[cat_cols].values.astype(np.int64), 0, clip)

        val_probs, score, model = fit_fold(Xtr_n, Xtr_c, y[tr], Xva_n, Xva_c, y[va], cat_dims, n_classes, cfg)
        oof[va] = val_probs
        test_probs += predict_all(model, Xte_n, Xte_c, cfg["eval_bs"]) / len(cv)

    print("\n=== OOF evaluation ===")
    evaluate_predictions(y, oof, classes)
    if smoke:
        print("[smoke] OK"); return
    weights, tuned = tune_class_weights(y, oof)
    print(f"Balanced accuracy after class-weight tuning: {tuned:.5f}  (weights={dict(zip(classes, weights.round(3)))})")
    save_oof_predictions(oof, f"realmlp{sfx}")
    np.save(PREDICTIONS_DIR / f"test_realmlp{sfx}.npy", test_probs)   # for blended submissions
    preds = np.asarray(classes)[(test_probs * weights).argmax(1)]
    save_submission(test_df["id"], preds, f"submission_realmlp{sfx}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reduced CPU sanity run")
    ap.add_argument("--rows", type=int, default=6000, help="subsample size in smoke mode")
    ap.add_argument("--epochs", type=int, default=1, help="epochs in smoke mode")
    ap.add_argument("--seed", type=int, default=None,
                    help="override seed; suffixes outputs (_s<seed>) for seed-ensembling")
    args = ap.parse_args()
    main(smoke=args.smoke, rows=args.rows, epochs=args.epochs, seed=args.seed)
