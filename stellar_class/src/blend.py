"""Blend member OOFs, search weights on balanced accuracy, write a blended submission.

Members are referenced by name: predictions/oof_<name>.npy (+ predictions/test_<name>.npy
for the submission). Weights are searched to maximize class-weight-tuned OOF balanced
accuracy (the trusted CV); the same weights + class multipliers are applied to the test
probabilities.

    python stellar_class/src/blend.py realmlp autogluon_best_quality
    python stellar_class/src/blend.py realmlp autogluon_best_quality tabpfn --name blend3
"""
import argparse
import itertools
import os
import sys

import numpy as np
import polars as pl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import encode_target
from validation import (load_data_with_folds, tune_class_weights, save_submission,
                        PREDICTIONS_DIR, DATA_DIR)
from sklearn.metrics import balanced_accuracy_score


def _simplex(n, step):
    """All weight vectors on the n-simplex with the given grid step (sum to 1)."""
    k = round(1 / step)
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            w = [c / k for c in combo]
            yield np.array(w + [1 - sum(w)])


def main(names, out_name, step):
    y, _, classes = encode_target(load_data_with_folds())
    oofs = [np.load(PREDICTIONS_DIR / f"oof_{n}.npy").astype(np.float64) for n in names]

    print("Member OOF (tuned balanced accuracy):")
    for n, o in zip(names, oofs):
        _, t = tune_class_weights(y, o)
        print(f"  {n:24s} {t:.5f}")

    best = (-1.0, None)
    for w in _simplex(len(names), step):
        blend = sum(wi * oi for wi, oi in zip(w, oofs))
        _, t = tune_class_weights(y, blend)
        if t > best[0]:
            best = (t, w)
    score, w = best
    blend_oof = sum(wi * oi for wi, oi in zip(w, oofs))
    weights, tuned = tune_class_weights(y, blend_oof)
    print(f"\nBest blend: tuned BA={tuned:.5f}")
    print("  weights : " + ", ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))
    print(f"  class multipliers: {dict(zip(classes, weights.round(3)))}")
    print(f"  (raw blend BA={balanced_accuracy_score(y, blend_oof.argmax(1)):.5f})")

    tests = []
    for n in names:
        p = PREDICTIONS_DIR / f"test_{n}.npy"
        if not p.exists():
            print(f"\n[!] missing {p} — cannot write submission. Re-run {n} to save test probs.")
            return
        tests.append(np.load(p).astype(np.float64))
    blend_test = sum(wi * ti for wi, ti in zip(w, tests))
    test_id = pl.read_csv(DATA_DIR / "test.csv")["id"]
    preds = np.asarray(classes)[(blend_test * weights).argmax(1)]
    save_submission(test_id, preds, f"submission_{out_name}.csv")
    np.save(PREDICTIONS_DIR / f"test_{out_name}.npy", blend_test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+", help="member names (predictions/oof_<name>.npy)")
    ap.add_argument("--name", default="blend", help="output submission/probs name")
    ap.add_argument("--step", type=float, default=0.05, help="simplex grid step for weight search")
    args = ap.parse_args()
    main(args.members, args.name, args.step)
