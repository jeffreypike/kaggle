"""Normalize all stacker base predictions to clean (N,3)/(M,3) float arrays in /tmp/stack_bases,
ready to upload as the private dataset jeffreypikeai/s6e6-stack-bases (the stacker notebook's input).

Why pre-normalize + bundle instead of using Kaggle kernel_sources: public base notebooks share
output filenames (oof_preds.npy, test_preds.csv) and can change their formats (TabM's test file
diverged from its OOF format and broke a kernel_sources run). Here we load each base with a
verified, alignment-checked loader and emit uniform oof_<name>.npy / test_<name>.npy.

Prereqs: the 6 public base outputs cached in /tmp/s6e6_bases (kaggle kernels output …), and our
distinct RealMLP OOFs in predictions/ (oof/test_realmlp_{bs128,ls0.08}.npy).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from validation import load_data_with_folds, PREDICTIONS_DIR, DATA_DIR
from sklearn.metrics import balanced_accuracy_score

B = "/tmp/s6e6_bases/"; OUT = "/tmp/stack_bases/"
TMAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data_with_folds().to_pandas()
    ids = df["id"].to_numpy(); y = df["class"].map(TMAP).to_numpy(); N = len(y)
    test = pd.read_csv(DATA_DIR / "test.csv"); tids = test["id"]; M = len(test)

    def by_id(p, c, idx): return pd.read_csv(B + p).set_index("id").loc[idx, c].to_numpy(float)
    def firstn(p, n):     return pd.read_csv(B + p).iloc[:n, -3:].to_numpy(float)
    def flat(p, n):       return pd.read_csv(B + p).iloc[:, 0].to_numpy().reshape(-1, 3)[:n]
    def npy(p, n):        return np.load(B + p).astype(float)[:n]

    # name -> (oof loader, test loader); loaders verified to align to train/test row order
    S = {
        "xgb0":     (lambda: firstn("cdeotte_xgb-v0-for-s6e6/oof_xgb_cv.csv", N),
                     lambda: firstn("cdeotte_xgb-v0-for-s6e6/test_xgb_preds.csv", M)),
        "xgb1":     (lambda: npy("cdeotte_xgb-v1-for-s6e6/oof_preds.npy", N),
                     lambda: npy("cdeotte_xgb-v1-for-s6e6/test_preds.npy", M)),
        "realmlp0": (lambda: by_id("yekenot_ps-s6-e6-realmlp-pytorch/oof_preds.csv", ["GALAXY", "QSO", "STAR"], ids),
                     lambda: by_id("yekenot_ps-s6-e6-realmlp-pytorch/test_preds.csv", ["GALAXY", "QSO", "STAR"], tids)),
        "realmlp1": (lambda: npy("cdeotte_realmlp-v1-for-s6e6/oof_preds.npy", N),
                     lambda: npy("cdeotte_realmlp-v1-for-s6e6/test_preds.npy", M)),
        "tabm":     (lambda: flat("donmarch14_s6e6-tabm/oof_preds.csv", N),
                     lambda: flat("donmarch14_s6e6-tabm/test_preds.csv", M)),
        "cat":      (lambda: by_id("cdeotte_cat-v0-for-s6e6/catboost_oof_predictions.csv", ["prob_GALAXY", "prob_QSO", "prob_STAR"], ids),
                     lambda: by_id("cdeotte_cat-v0-for-s6e6/catboost_test_predictions.csv", ["prob_GALAXY", "prob_QSO", "prob_STAR"], tids)),
    }
    for nm, (of, tf) in S.items():
        o, t = of(), tf()
        assert o.shape == (N, 3) and t.shape == (M, 3), f"{nm} bad shape {o.shape}/{t.shape}"
        ba = balanced_accuracy_score(y, o.argmax(1)); assert ba > 0.90, f"{nm} BA {ba:.3f}"
        np.save(OUT + f"oof_{nm}.npy", o.astype(np.float32)); np.save(OUT + f"test_{nm}.npy", t.astype(np.float32))
        print(f"  {nm:9s} oof{o.shape} test{t.shape} BA={ba:.4f}")

    for nm, src in [("bs128", "bs128"), ("ls008", "ls0.08")]:
        o = np.load(PREDICTIONS_DIR / f"oof_realmlp_{src}.npy").astype(np.float32)
        t = np.load(PREDICTIONS_DIR / f"test_realmlp_{src}.npy").astype(np.float32)
        np.save(OUT + f"oof_{nm}.npy", o); np.save(OUT + f"test_{nm}.npy", t)
        print(f"  {nm:9s} oof{o.shape} test{t.shape} (ours)")

    with open(OUT + "dataset-metadata.json", "w") as f:
        f.write('{ "title": "s6e6-stack-bases", "id": "jeffreypikeai/s6e6-stack-bases", "licenses": [{ "name": "CC0-1.0" }] }')
    print("staged ->", OUT)


if __name__ == "__main__":
    main()
