"""TabPFN-3 stacker (the current S6E6 meta) — run on the 3080 (CUDA).

Reproduces philippsinger/tabpfn-3-stacker (which builds on cdeotte/gpu-logistic-regression-stacker):
stack diverse base-model OOFs as *logits*, append the raw original features, and use TabPFN-3
as the meta-model. We add our own RealMLP-8seed ensemble as an extra base.

Offline LogReg-logit proxy on our standardized 5-fold (alignment verified) reached:
  public bases only 0.96966 | + our RealMLP 0.96968 | + our RealMLP + AG 0.96970  (our prior ceiling 0.96946)
This script runs the real TabPFN-3 meta, which should beat the LogReg proxy, and an LB submission
is the ground truth (our CV→LB offset is +0.0007–0.0009).

    python stellar_class/src/stack_tabpfn.py                 # fetch bases, LogReg ref + TabPFN, submit
    python stellar_class/src/stack_tabpfn.py --no-tabpfn     # LogReg-logit only (no CUDA needed)
    python stellar_class/src/stack_tabpfn.py --subsample 200000   # cap TabPFN context if 10GB VRAM OOMs

Base OOFs are pulled with the Kaggle API (needs KAGGLE_API_TOKEN). TabPFN-3 weights download
from HuggingFace on first run. On a single 10GB 3080, TabPFN-3 at full 577k context may OOM —
use --subsample to cap the fit context (a stacker tolerates this well).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import load_data_with_folds, tune_class_weights, save_submission, DATA_DIR, PREDICTIONS_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

TMAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
CLASSES = ["GALAXY", "QSO", "STAR"]
CACHE = "/tmp/s6e6_bases"
EPS, CLIP = 1e-15, 30.0

# (name, kaggle kernel, oof relpath, test relpath, loader kind)
#   kinds: 'id'=csv indexed by id (cols below), 'firstn'=last-3-cols first-N rows, 'flat'=single col N*3, 'npy'
BASES = [
    ("xgb0",     "cdeotte/xgb-v0-for-s6e6",            "oof_xgb_cv.csv",                  "test_xgb_preds.csv",            "firstn"),
    ("xgb1",     "cdeotte/xgb-v1-for-s6e6",            "oof_preds.npy",                   "test_preds.npy",               "npy"),
    ("realmlp0", "yekenot/ps-s6-e6-realmlp-pytorch",   "oof_preds.csv",                   "test_preds.csv",               "id:GALAXY,QSO,STAR"),
    ("realmlp1", "cdeotte/realmlp-v1-for-s6e6",        "oof_preds.npy",                   "test_preds.npy",               "npy"),
    ("tabm",     "donmarch14/s6e6-tabm",               "oof_preds.csv",                   "test_preds.csv",               "flat"),
    ("cat",      "cdeotte/cat-v0-for-s6e6",            "catboost_oof_predictions.csv",    "catboost_test_predictions.csv","id:prob_GALAXY,prob_QSO,prob_STAR"),
]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS).astype(np.float64)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)


def fetch(kernel):
    d = os.path.join(CACHE, kernel.replace("/", "_"))
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", kernel, "-p", d], check=True)
    return d


def load_base(kind, path, ids, n):
    if kind == "npy":
        return np.load(path).astype(np.float64)[:n]
    if kind == "firstn":
        return pd.read_csv(path).iloc[:n, -3:].to_numpy(np.float64)
    if kind == "flat":
        return pd.read_csv(path).iloc[:, 0].to_numpy().reshape(-1, 3)[:n]
    if kind.startswith("id:"):
        cols = kind[3:].split(",")
        return pd.read_csv(path).set_index("id").loc[ids, cols].to_numpy(np.float64)
    raise ValueError(kind)


def main(use_tabpfn, subsample, n_estimators, devices="cuda"):
    df = load_data_with_folds().to_pandas()
    test = pd.read_csv(DATA_DIR / "test.csv")
    ids = df["id"].to_numpy(); y = df["class"].map(TMAP).to_numpy(); folds = df["fold"].to_numpy()
    N, M = len(y), len(test); nC = 3
    test_ids = test["id"]

    # ── assemble base OOF/test (logits), with per-base alignment sanity check ──
    oof_parts, test_parts, names = [], [], []
    for name, kernel, oof_rel, test_rel, kind in BASES:
        d = fetch(kernel)
        o = load_base(kind, os.path.join(d, oof_rel), ids, N)
        t = load_base(kind, os.path.join(d, test_rel), test_ids, M)
        ba = balanced_accuracy_score(y, o.argmax(1))
        assert ba > 0.90, f"{name}: solo BA {ba:.3f} too low — misaligned/class-order mismatch"
        print(f"  base {name:9s} solo tuned={tune_class_weights(y, o)[1]:.5f}  (test rows {t.shape[0]})", flush=True)
        oof_parts.append(logit(o)); test_parts.append(logit(t)); names.append(name)

    # add our two *distinct* RealMLP bases. The HPO screen showed these (worse solo, but more
    # decorrelated from the pool's two RealMLPs) contribute more to the stack than our ref ensemble
    # (bs128 +0.00012 / ls0.08 vs ref +0.00006; pair best). Falls back to realmlp_ens if not present.
    OUR = [("our_bs128", "oof_realmlp_bs128.npy", "test_realmlp_bs128.npy"),
           ("our_ls008", "oof_realmlp_ls0.08.npy", "test_realmlp_ls0.08.npy")]
    if not all((PREDICTIONS_DIR / o).exists() for _, o, _ in OUR):
        OUR = [("our_realmlp", "oof_realmlp_ens.npy", "test_realmlp_ens.npy")]
    for nm, oof_f, test_f in OUR:
        o = np.load(PREDICTIONS_DIR / oof_f).astype(np.float64)
        t = np.load(PREDICTIONS_DIR / test_f).astype(np.float64)
        oof_parts.append(logit(o)); test_parts.append(logit(t)); names.append(nm)
        print(f"  base {nm:9s} solo tuned={tune_class_weights(y, o)[1]:.5f}")

    # raw original features appended to the logit stack (TabPFN can use them without overfitting)
    feat_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    Ftr = df[feat_cols].to_numpy(np.float64); Fte = test[feat_cols].to_numpy(np.float64)
    Xtr = np.hstack(oof_parts + [Ftr]).astype(np.float32)
    Xte = np.hstack(test_parts + [Fte]).astype(np.float32)
    print(f"\nstack features: {len(names)} bases x3 logits + {len(feat_cols)} raw = {Xtr.shape[1]} cols; "
          f"members={names}", flush=True)

    # ── LogReg-logit reference (always; no GPU needed) ──
    logreg_oof = np.zeros((N, nC))
    for k in range(5):
        tr, va = folds != k, folds == k
        m = LogisticRegression(max_iter=3000, C=0.1, multi_class="multinomial")
        m.fit(Xtr[tr], y[tr]); logreg_oof[va] = m.predict_proba(Xtr[va])
    w_lr, t_lr = tune_class_weights(y, logreg_oof)
    print(f"\nLogReg-logit stacker  OOF tuned = {t_lr:.5f}", flush=True)

    if not use_tabpfn:
        m = LogisticRegression(max_iter=3000, C=0.1, multi_class="multinomial").fit(Xtr, y)
        test_prob = m.predict_proba(Xte)
        preds = np.asarray(CLASSES)[(test_prob * w_lr).argmax(1)]
        save_submission(test_ids, preds, "submission_stack_logreg.csv")
        np.save(PREDICTIONS_DIR / "oof_stack_logreg.npy", logreg_oof)
        np.save(PREDICTIONS_DIR / "test_stack_logreg.npy", test_prob)
        return

    # ── TabPFN-3 meta (CUDA) — fit on all train, predict test (philippsinger's recipe) ──
    # NOTE: TabPFN is PyTorch/CUDA only — no TPU. On Kaggle GPU T4x2 pass --devices cuda:0,cuda:1
    # (32GB total → full 577k context, no subsample needed); on a single 10GB 3080 use --subsample.
    from tabpfn import TabPFNClassifier
    dev = devices.split(",") if "," in devices else devices
    fit_idx = np.arange(N)
    if subsample and subsample < N:
        rng = np.random.RandomState(42)
        # stratified-ish subsample of the fit context to fit limited VRAM
        fit_idx = rng.choice(N, subsample, replace=False)
        print(f"TabPFN fit context subsampled to {subsample:,} rows", flush=True)
    def make_clf():
        return TabPFNClassifier(device=dev, n_estimators=n_estimators, balance_probabilities=True)
    clf = make_clf()
    clf.fit(Xtr[fit_idx], y[fit_idx])
    test_prob = clf.predict_proba(Xte)

    # OOF estimate for CV trust: TabPFN per-fold (context = other 4 folds, optionally subsampled)
    tabpfn_oof = np.zeros((N, nC))
    for k in range(5):
        tr = np.where(folds != k)[0]; va = np.where(folds == k)[0]
        if subsample and subsample < len(tr):
            tr = np.random.RandomState(k).choice(tr, subsample, replace=False)
        c = make_clf()
        c.fit(Xtr[tr], y[tr]); tabpfn_oof[va] = c.predict_proba(Xtr[va])
        print(f"  tabpfn fold {k} done", flush=True)
    w_tp, t_tp = tune_class_weights(y, tabpfn_oof)
    print(f"\nTabPFN-3 stacker  OOF tuned = {t_tp:.5f}   (LogReg ref {t_lr:.5f})", flush=True)

    preds = np.asarray(CLASSES)[(test_prob * w_tp).argmax(1)]
    save_submission(test_ids, preds, "submission_stack_tabpfn.csv")
    np.save(PREDICTIONS_DIR / "oof_stack_tabpfn.npy", tabpfn_oof)
    np.save(PREDICTIONS_DIR / "test_stack_tabpfn.npy", test_prob)
    print("saved submission_stack_tabpfn.csv + oof/test_stack_tabpfn.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tabpfn", action="store_true", help="LogReg-logit only (no CUDA)")
    ap.add_argument("--subsample", type=int, default=0, help="cap TabPFN fit context rows (VRAM)")
    ap.add_argument("--n-estimators", type=int, default=2)
    ap.add_argument("--devices", default="cuda",
                    help="TabPFN device(s): 'cuda' (single), or 'cuda:0,cuda:1' for Kaggle T4x2")
    args = ap.parse_args()
    main(not args.no_tabpfn, args.subsample, args.n_estimators, args.devices)
