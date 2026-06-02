"""Builds notebooks/stellar-baseline-autogluon.ipynb (run from project root)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
C = lambda s: cells.append(nbf.v4.new_code_cell(s))
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))

M("""# 🌌 Stellar Classification — AutoGluon Baseline

**Goal:** establish a strong leaderboard baseline and verify that our local CV tracks the
public LB (the EDA's adversarial-validation AUC ≈ 0.50 says train/test are
indistinguishable, so we expect CV ≈ LB).

Competition metric: **balanced accuracy**. We let AutoGluon bag + stack the full model
zoo, optimizing balanced accuracy directly, then report out-of-fold (OOF) balanced
accuracy and write a submission.

**Kaggle setup:** Add Data → the `playground-series-s6e6` competition; Settings →
Internet **On** (needed to pip-install AutoGluon).""")

M("## 1. Install AutoGluon")
C("""# setuptools>=81 dropped pkg_resources, which AutoGluon's model import-guards still use
# (silently disabling XGBoost/CatBoost/etc). Pin it below 81, then install AutoGluon.
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "setuptools<81", "autogluon.tabular[all]"], check=True)
import pkg_resources  # noqa: F401  -> raises if the pin didn't take
print("AutoGluon install OK")""")

M("## 2. Imports & data")
C('''from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from autogluon.tabular import TabularPredictor

SEED = 42
TARGET = "class"

# Resolve data dir on Kaggle (/kaggle/input/<comp>) or locally (../data)
_kaggle = Path("/kaggle/input")
_hits = sorted(_kaggle.rglob("train.csv")) if _kaggle.exists() else []
DATA = _hits[0].parent if _hits else Path("../data")
print("Data dir:", DATA)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
print("train", train.shape, "test", test.shape)''')

M("## 3. Feature engineering — color indices")
C('''def add_colors(df):
    df = df.copy()
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    return df

NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR = ["u_g", "g_r", "r_i", "i_z"]
CAT = ["spectral_type", "galaxy_population"]   # AutoGluon handles these natively
FEATS = NUM + COLOR + CAT

train = add_colors(train)
test = add_colors(test)
train_data = train[FEATS + [TARGET]]
test_data = test[FEATS]
CLASSES = sorted(train[TARGET].unique())   # ['GALAXY', 'QSO', 'STAR']
print("Classes:", CLASSES)''')

M("""## 4. Train AutoGluon (best_quality, balanced accuracy)

`dynamic_stacking=False` skips the DyStack sub-fit so the full time budget goes to the
real fit. On Kaggle's larger RAM, fold fitting parallelizes (ray) without the memory
throttling we saw locally. Bump `time_limit` if you want a deeper search.""")
C('''predictor = TabularPredictor(
    label=TARGET,
    eval_metric="balanced_accuracy",
    path="ag_models",
)
predictor.fit(
    train_data,
    presets="best_quality",
    time_limit=14400,  # 4h guardrail (well under the 12h session); stops early if done.
    dynamic_stacking=False,
)
predictor.leaderboard(silent=True)''')

M("## 5. OOF balanced accuracy (our CV estimate to compare against the LB)")
C('''oof = predictor.predict_proba_oof()[CLASSES].to_numpy()
y_true = train[TARGET].to_numpy()

raw_oof_pred = np.array(CLASSES)[oof.argmax(1)]
raw_bal_acc = balanced_accuracy_score(y_true, raw_oof_pred)
print(f"OOF balanced accuracy (raw argmax) : {raw_bal_acc:.5f}   <- clean CV estimate")''')

M("""## 6. Class-weight tuning for balanced accuracy

Plain argmax under-predicts the rare STAR/QSO classes; balanced accuracy punishes that.
We search per-class probability multipliers on the OOF predictions. Note the tuned OOF
score carries mild optimism (weights are fit on the same OOF), so treat the **raw** OOF
as the honest CV number and the LB as the arbiter.""")
C('''def tune_class_weights(y_true, proba, classes, n_rounds=3):
    grid = np.linspace(0.2, 3.0, 29)
    w = np.ones(len(classes))
    score = lambda ww: balanced_accuracy_score(y_true, np.array(classes)[(proba * ww).argmax(1)])
    best = score(w)
    for _ in range(n_rounds):
        for k in range(len(w)):
            best_wk = w[k]
            for g in grid:
                w[k] = g
                s = score(w)
                if s > best:
                    best, best_wk = s, g
            w[k] = best_wk
    return w / w.mean(), best

weights, tuned_bal_acc = tune_class_weights(y_true, oof, CLASSES)
print(f"OOF balanced accuracy (tuned) : {tuned_bal_acc:.5f}  weights={dict(zip(CLASSES, weights.round(3)))}")''')

M("## 7. Predict test & write submission")
C('''test_proba = predictor.predict_proba(test_data)[CLASSES].to_numpy()
test_pred = np.array(CLASSES)[(test_proba * weights).argmax(1)]

submission = pd.DataFrame({"id": test["id"], "class": test_pred})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(submission["class"].value_counts(normalize=True).round(4))''')

M("""## 8. CV ↔ LB check

After submitting `submission.csv`, compare the public LB score to the OOF numbers above:

- **Close to raw OOF** → our local CV is trustworthy; iterate offline with confidence.
- **LB noticeably below tuned OOF** → the class-weight tuning is overfitting the OOF;
  fall back to raw argmax for submissions.
- **Large CV↔LB gap either way** → revisit the adversarial-validation / fold setup.""")

nb.cells = cells
nbf.write(nb, "notebooks/stellar-baseline-autogluon.ipynb")
print(f"wrote notebooks/stellar-baseline-autogluon.ipynb ({len(cells)} cells)")
