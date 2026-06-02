"""Builds notebooks/stellar-realmlp-5seed-tpu.ipynb (run from project root).

Self-contained Kaggle TPU notebook: trains RealMLP-TD with 5 seeds, averages them,
and writes the seed-ensemble OOF/test probs + submission. Embeds the validated model
block from src/train_realmlp.py verbatim so the notebook can't drift from the script.
"""
import nbformat as nbf

# --- pull the validated model block (FE -> fit_fold) out of the script -------------
src = open("src/train_realmlp.py").read().splitlines()
start = next(i for i, l in enumerate(src) if l.startswith("PI = math.pi"))
end = next(i for i, l in enumerate(src) if l.strip() == "return best_probs, best_score, model")
MODEL_BLOCK = "\n".join(src[start:end + 1])

nb = nbf.v4.new_notebook()
cells = []
C = lambda s: cells.append(nbf.v4.new_code_cell(s))
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))

M("""# 🌌 RealMLP-TD × 5 seeds (Flax NNX, TPU)

Trains the RealMLP-TD model (the top single model for this competition) with **5 seeds**
on the standardized 5-fold splits, averages them into a lower-variance member, and writes:
`oof_realmlp_ens.npy`, `test_realmlp_ens.npy` (download these to blend with AutoGluon
locally) and `submission.csv` (RealMLP-only).

**Kaggle setup:** accelerator **TPU VM v3-8**, Internet **On** (to pip-install flax/optax),
and add the `playground-series-s6e6` competition data.

*Note:* the training loop is single-device, so it uses one TPU core (small batch + small
model suit TPU poorly). It runs and uses TPU quota; it won't be 8× fast. Each seed's
OOF/test is saved as it finishes, so a session timeout won't lose completed seeds.""")

M("## 1. Install")
C("""import sys, subprocess
# JAX is preinstalled on Kaggle's TPU image; add flax (for nnx) + optax.
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

print("jax", jax.__version__, "| flax", __import__("flax").__version__, "| devices:", jax.devices())""")

M("## 3. RealMLP-TD model (embedded verbatim from `src/train_realmlp.py`)")
C(MODEL_BLOCK)

M("## 4. Data, folds, feature engineering")
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

# Standardized folds — must match the local pipeline (StratifiedKFold 5, shuffle, rs=42)
# so the OOF rows align with the AutoGluon OOF for blending.
folds = np.zeros(len(y), dtype=int)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    folds[va] = i

raw_cat = ["spectral_type", "galaxy_population"]
raw_num = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
state = {}
Xtr, new_cat, new_num, combos = feature_engineering(train_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=True)
Xte, _, _, _ = feature_engineering(test_df[raw_cat + raw_num], raw_cat[:], raw_num[:], state, fit=False)
cat_cols = sorted(raw_cat + new_cat)
num_cols = raw_num + new_num
Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1)
Xte = Xte.reindex(sorted(Xte.columns), axis=1)
cat_dims = [int(max(Xtr[c].max(), Xte[c].max()) + 1) for c in cat_cols]
print("train", Xtr.shape, "test", Xte.shape, "cat_dims", len(cat_dims))


def tune_class_weights(y_true, proba, n_rounds=3, grid=None):
    if grid is None:
        grid = np.linspace(0.2, 3.0, 29)
    w = np.ones(proba.shape[1])
    best = balanced_accuracy_score(y_true, proba.argmax(1))
    for _ in range(n_rounds):
        for k in range(len(w)):
            bw = w[k]
            for g in grid:
                w[k] = g
                s = balanced_accuracy_score(y_true, (proba * w).argmax(1))
                if s > best:
                    best, bw = s, g
            w[k] = bw
    return w / w.mean(), best''')

M("## 5. Train one seed (5 folds, OOF + test)")
C('''def run_seed(seed):
    np.random.seed(seed)                      # makes per-fold data shuffling seed-dependent
    cfg = dict(CONFIG); cfg["seed"] = seed
    oof = np.zeros((len(Xtr), n_classes))
    test_probs = np.zeros((len(Xte), n_classes))
    for fold in range(5):
        tr = np.where(folds != fold)[0]; va = np.where(folds == fold)[0]
        print(f"[seed {seed}] === fold {fold} ===", flush=True)
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
        test_probs += predict_all(model, Xte_n, Xte_c, cfg["eval_bs"]) / 5
    return oof, test_probs''')

M("## 6. Run 5 seeds, average, submit")
C('''SEEDS = [0, 1, 2, 3, 4]
oof_sum = np.zeros((len(Xtr), n_classes))
test_sum = np.zeros((len(Xte), n_classes))
for s in SEEDS:
    oof_s, test_s = run_seed(s)
    np.save(OUT / f"oof_realmlp_s{s}.npy", oof_s)      # incremental: survive a timeout
    np.save(OUT / f"test_realmlp_s{s}.npy", test_s)
    raw = balanced_accuracy_score(y, oof_s.argmax(1)); _, t = tune_class_weights(y, oof_s)
    print(f">>> seed {s}: OOF raw={raw:.5f}  tuned={t:.5f}", flush=True)
    oof_sum += oof_s; test_sum += test_s

oof_ens = oof_sum / len(SEEDS)
test_ens = test_sum / len(SEEDS)
np.save(OUT / "oof_realmlp_ens.npy", oof_ens)
np.save(OUT / "test_realmlp_ens.npy", test_ens)

raw = balanced_accuracy_score(y, oof_ens.argmax(1))
weights, tuned = tune_class_weights(y, oof_ens)
print(f"\\n=== {len(SEEDS)}-seed ensemble OOF: raw={raw:.5f}  tuned={tuned:.5f}  (vs single-seed ~0.96761) ===")
print(f"class multipliers: {dict(zip(classes, weights.round(3)))}")

preds = np.asarray(classes)[(test_ens * weights).argmax(1)]
pd.DataFrame({"id": test_df["id"], "class": preds}).to_csv(OUT / "submission.csv", index=False)
print("wrote submission.csv, oof_realmlp_ens.npy, test_realmlp_ens.npy")''')

M("""## 7. Next

Download `oof_realmlp_ens.npy` and `test_realmlp_ens.npy` into the local repo's
`predictions/` (as `oof_realmlp_ens.npy` / `test_realmlp_ens.npy`), then blend with
AutoGluon:

```
python src/blend.py realmlp_ens autogluon_best_quality --name blend_ens_ag
```""")

nb.cells = cells
nbf.write(nb, "notebooks/stellar-realmlp-5seed-tpu.ipynb")
print(f"wrote notebooks/stellar-realmlp-5seed-tpu.ipynb ({len(cells)} cells)")
