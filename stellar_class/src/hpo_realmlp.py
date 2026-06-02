"""RealMLP hyperparameter screen: single-seed 5-fold OOF balanced accuracy per config.

Run at FULL scale (config ranking doesn't transfer reliably from a subsample), so use a
GPU/TPU box:  python stellar_class/src/hpo_realmlp.py
Local logic check (tiny, CPU):  python stellar_class/src/hpo_realmlp.py --quick

Edit CANDIDATES below. Each is an override on the reference CONFIG; single-axis changes
isolate the effect. Promote the winner to the full 8-seed run / notebook.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_realmlp as T
from validation import load_data_with_folds, tune_class_weights, DATA_DIR
from sklearn.metrics import balanced_accuracy_score

# name -> overrides on the reference CONFIG
CANDIDATES = {
    "ref":        {},
    "wide":       {"hidden_dims": [1024, 1024, 1024]},
    "deep":       {"hidden_dims": [512, 512, 512, 512]},
    "epochs8":    {"epochs": 8},
    "epochs10":   {"epochs": 10},
    "dropout0.1": {"dropout": 0.10},
    "nens16":     {"n_ens": 16},
}


def run_config(overrides, Xtr, Xte, y, folds, combos, cat_cols, num_cols, cat_dims, n_classes, seed=42):
    cfg = dict(T.CONFIG); cfg.update(overrides); cfg["seed"] = seed
    oof = np.zeros((len(Xtr), n_classes))
    for fold in range(5):
        tr = np.where(folds != fold)[0]; va = np.where(folds == fold)[0]
        enc = T.make_target_encoder(seed)
        tr_te = enc.fit_transform(Xtr.iloc[tr][combos], y[tr]).astype(np.float32)
        va_te = enc.transform(Xtr.iloc[va][combos]).astype(np.float32)
        pre = T.NumericalPreprocessor(cfg["tfms"]).fit(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xtr_n = pre.transform(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xva_n = pre.transform(np.hstack([Xtr.iloc[va][num_cols].values.astype(np.float32), va_te]))
        clip = np.array(cat_dims) - 1
        Xtr_c = np.clip(Xtr.iloc[tr][cat_cols].values.astype(np.int64), 0, clip)
        Xva_c = np.clip(Xtr.iloc[va][cat_cols].values.astype(np.int64), 0, clip)
        val_probs, _, _ = T.fit_fold(Xtr_n, Xtr_c, y[tr], Xva_n, Xva_c, y[va], cat_dims, n_classes, cfg)
        oof[va] = val_probs
    raw = balanced_accuracy_score(y, oof.argmax(1))
    _, tuned = tune_class_weights(y, oof)
    return raw, tuned


def main(quick=False):
    train_df = load_data_with_folds().to_pandas()
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(train_df["class"].unique()); cmap = {c: i for i, c in enumerate(classes)}
    y = train_df["class"].map(cmap).to_numpy(); folds = train_df["fold"].to_numpy()
    raw_cat = ["spectral_type", "galaxy_population"]; raw_num = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    st = {}
    Xtr, new_cat, new_num, combos = T.feature_engineering(train_df[raw_cat + raw_num], raw_cat[:], raw_num[:], st, fit=True)
    Xte, *_ = T.feature_engineering(test_df[raw_cat + raw_num], raw_cat[:], raw_num[:], st, fit=False)
    cat_cols = sorted(raw_cat + new_cat); num_cols = raw_num + new_num
    Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1); Xte = Xte.reindex(sorted(Xte.columns), axis=1)
    cat_dims = [int(max(Xtr[c].max(), Xte[c].max()) + 1) for c in cat_cols]
    n_classes = len(classes)

    cands = CANDIDATES
    if quick:                                   # local CPU logic check only
        keep = np.random.RandomState(0).choice(len(Xtr), 8000, replace=False)
        Xtr, y, folds = Xtr.iloc[keep].reset_index(drop=True), y[keep], folds[keep]
        cands = {"ref": {"n_ens": 2, "hidden_dims": [64], "epochs": 1, "train_bs": 2048},
                 "wide": {"n_ens": 2, "hidden_dims": [128], "epochs": 1, "train_bs": 2048}}

    print(f"=== RealMLP HPO screen ({'quick' if quick else 'full'}), single seed ===", flush=True)
    results = {}
    for name, ov in cands.items():
        t0 = time.time()
        raw, tuned = run_config(ov, Xtr, Xte, y, folds, combos, cat_cols, num_cols, cat_dims, n_classes)
        results[name] = tuned
        print(f"{name:11s}: raw={raw:.5f}  tuned={tuned:.5f}  ({time.time()-t0:.0f}s)  {ov}", flush=True)
    best = max(results, key=results.get)
    print(f"\nbest: {best} (tuned={results[best]:.5f})  vs ref ({results.get('ref', float('nan')):.5f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny CPU logic check")
    args = ap.parse_args()
    main(quick=args.quick)
