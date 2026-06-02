"""TabPFN-3 on the standardized folds — a diverse (non-GBDT) member for stacking.

Mirrors Philipp Singer's reference notebook config (raw frame, no feature engineering,
n_estimators=2, eval_metric=balanced_accuracy with internal decision-threshold tuning),
but loops our 5 standardized folds to emit OOF predictions that stack cleanly with the
LightGBM/AutoGluon OOFs. Test is predicted once from a full (capped) context.

Requires a CUDA GPU. Install on that machine:  pip install tabpfn
(weights are gated — accept the license at https://huggingface.co/Prior-Labs/tabpfn_3
 and `huggingface-cli login`, or set TABPFN_MODEL_CACHE_DIR to local weights.)

VRAM lever: --max-context caps the per-fold context (stratified subsample). TabPFN
degrades gracefully on context size, so if full 577k OOMs on a 10GB card, drop this
(e.g. --max-context 200000) — usually within noise of full, and faster.

Examples:
    # quick VRAM probe: one fold, small context, skip test
    python stellar_class/src/train_tabpfn.py --max-context 50000 --max-folds 1 --skip-test
    # full OOF + submission
    python stellar_class/src/train_tabpfn.py --max-context 400000
"""
import argparse
import gc
import os
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import (load_data_with_folds, get_custom_cv, evaluate_predictions,
                        save_oof_predictions, save_submission, tune_class_weights, DATA_DIR)
from features import NUM_COLS, CAT_COLS, encode_target

RAW_FEATS = NUM_COLS + CAT_COLS  # raw frame: categoricals passed as-is, TabPFN encodes them


def _subsample(idx, y, max_context, seed):
    """Stratified subsample of `idx` down to max_context rows (no-op if already small)."""
    if max_context is None or max_context >= len(idx):
        return idx
    keep, _ = train_test_split(idx, train_size=max_context, stratify=y[idx], random_state=seed)
    return keep


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_and_evaluate(max_context=None, device="cuda", n_estimators=2,
                       max_folds=None, skip_test=False, seed=42):
    from tabpfn import TabPFNClassifier

    dev = device.split(",") if "," in device else device  # "cuda:0,cuda:1" -> multi-GPU

    def make_clf():
        return TabPFNClassifier(
            device=dev,
            n_estimators=n_estimators,
            eval_metric="balanced_accuracy",
            tuning_config={"tune_decision_thresholds": True},
        )

    print("=== Loading data with standardized folds ===")
    train_df = load_data_with_folds()
    test_df = pl.read_csv(DATA_DIR / "test.csv")
    X = train_df.select(RAW_FEATS).to_pandas()
    X_test = test_df.select(RAW_FEATS).to_pandas()
    y, le, classes = encode_target(train_df)
    cv = get_custom_cv(train_df)
    n_folds = len(cv) if max_folds is None else min(max_folds, len(cv))

    print(f"=== OOF over {n_folds} fold(s)  (max_context={max_context}, device={dev}) ===")
    oof = np.zeros((len(y), len(classes)))
    done = np.zeros(len(y), dtype=bool)
    for fold, (tr_idx, va_idx) in enumerate(cv[:n_folds]):
        ctx = _subsample(tr_idx, y, max_context, seed)
        print(f"--- fold {fold}: context={len(ctx):,}  predict={len(va_idx):,} ---")
        clf = make_clf()
        clf.fit(X.iloc[ctx], y[ctx])
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])
        done[va_idx] = True
        del clf
        _free_gpu()

    if done.all():
        print("\n=== OOF evaluation ===")
        evaluate_predictions(y, oof, classes)
        weights, tuned = tune_class_weights(y, oof)
        print(f"Balanced accuracy after class-weight tuning: {tuned:.5f}  "
              f"(weights={dict(zip(classes, weights.round(3)))})  "
              f"[note: TabPFN already tunes thresholds internally]")
        save_oof_predictions(oof, "tabpfn")
    else:
        print(f"\n[partial run: {done.sum():,}/{len(y):,} rows scored] — skipping OOF save.")
        weights = np.ones(len(classes))

    if skip_test or not done.all():
        print("Skipping test prediction / submission.")
        return

    print("\n=== Test prediction (single fit on full capped context) ===")
    ctx = _subsample(np.arange(len(y)), y, max_context, seed)
    print(f"final fit context={len(ctx):,}  predict test={len(X_test):,}")
    clf = make_clf()
    clf.fit(X.iloc[ctx], y[ctx])
    test_proba = clf.predict_proba(X_test)
    _free_gpu()
    test_preds = le.inverse_transform((test_proba * weights).argmax(1))
    save_submission(test_df["id"], test_preds, "submission_tabpfn.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN-3 OOF + submission on standardized folds")
    parser.add_argument("--max-context", type=int, default=None,
                        help="Cap per-fold (and final) training context to this many stratified rows (VRAM lever)")
    parser.add_argument("--device", type=str, default="cuda",
                        help='Torch device; comma-separate for multi-GPU, e.g. "cuda:0,cuda:1"')
    parser.add_argument("--n-estimators", type=int, default=2,
                        help="TabPFN ensemble members (higher = slower, small accuracy bump)")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Run only the first N folds (quick VRAM/score probe)")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test prediction / submission (for fast probes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_evaluate(max_context=args.max_context, device=args.device,
                       n_estimators=args.n_estimators, max_folds=args.max_folds,
                       skip_test=args.skip_test, seed=args.seed)
