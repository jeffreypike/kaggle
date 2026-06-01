"""AutoGluon benchmark: the strong ensemble to beat.

Unlike the manual/FLAML models, AutoGluon does its own preprocessing (native categorical
handling — so we feed raw string categoricals, not integer codes), bags + multi-layer
stacks LightGBM/XGBoost/CatBoost/NN, and optimizes balanced accuracy directly. Bagging
gives us out-of-fold predictions via `predict_proba_oof`.

Note: AutoGluon's OOF lives on *its own* internal bagged folds, not the standardized
folds the other models share, so use its score as a benchmark rather than as a clean
stacking input.

Run from anywhere:
    python stellar_class/src/train_autogluon.py --time-limit 600
    python stellar_class/src/train_autogluon.py --dry-run     # fast bagged GBM only
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
from autogluon.tabular import TabularPredictor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import (evaluate_predictions, save_oof_predictions, save_submission,
                        tune_class_weights, DATA_DIR, PROJECT_DIR)
from features import add_colors, FEAT_COLS, TARGET, encode_target

MODEL_DIR = PROJECT_DIR / "models" / "autogluon"


def train_and_evaluate(time_limit=2400, preset=None, dry_run=False):
    print("=== Step 1: Loading data ===")
    train_df = add_colors(pl.read_csv(DATA_DIR / "train.csv"))
    test_df = add_colors(pl.read_csv(DATA_DIR / "test.csv"))

    # Raw features for AutoGluon: keep categoricals as strings, let AG infer types.
    train_pdf = train_df.select(FEAT_COLS + [TARGET]).to_pandas()
    test_pdf = test_df.select(FEAT_COLS).to_pandas()
    y, _, class_names = encode_target(train_df)  # canonical order: [GALAXY, QSO, STAR]

    mode = "dry-run" if dry_run else (preset if preset else "GBDT stack")
    print(f"\n=== Step 2: AutoGluon fit (balanced_accuracy, {mode}) ===")
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric="balanced_accuracy",
        path=str(MODEL_DIR),
        verbosity=2,
    )
    if dry_run:
        # Fast path: a couple bagged GBMs, no stacking — still yields OOF.
        predictor.fit(train_pdf, time_limit=120, hyperparameters={"GBM": {}},
                      num_bag_folds=3, num_stack_levels=0)
    elif preset:
        # Full preset (e.g. best_quality): the whole model zoo + multi-layer stacking, to
        # study what AutoGluon ensembles and how much stacking helps. Slow under sequential
        # (no-ray) fold fitting on 577k rows, so give it a generous time_limit; AG builds
        # the best ensemble from whatever finishes. dynamic_stacking=False keeps the DyStack
        # sub-fit from ~2x-ing the runtime (and from the earlier corruption failure mode).
        predictor.fit(train_pdf, time_limit=time_limit, presets=preset,
                      dynamic_stacking=False)
    else:
        # Default: explicit GBDT stack. The full preset lineup starves under sequential
        # fold fitting, and the three GBDTs are what's competitive on this data anyway.
        predictor.fit(
            train_pdf,
            time_limit=time_limit,
            hyperparameters={"GBM": {}, "XGB": {}, "CAT": {}},
            num_bag_folds=5,
            num_stack_levels=1,
            dynamic_stacking=False,
        )

    print("\n=== Step 3: Leaderboard ===")
    print(predictor.leaderboard(silent=True))

    print("\n=== Step 4: OOF evaluation (AutoGluon internal folds) ===")
    oof_probs = predictor.predict_proba_oof()[class_names].to_numpy()
    evaluate_predictions(y, oof_probs, class_names)

    weights, tuned = tune_class_weights(y, oof_probs)
    print(f"Balanced accuracy after class-weight tuning: {tuned:.5f}  "
          f"(weights={dict(zip(class_names, weights.round(3)))})")

    suffix = f"_{preset}" if preset else ""  # keep preset runs from clobbering the default
    save_oof_predictions(oof_probs, f"autogluon{suffix}")

    print("\n=== Step 5: Test predictions + submission ===")
    test_probs = predictor.predict_proba(test_pdf)[class_names].to_numpy()
    test_preds = np.asarray(class_names)[(test_probs * weights).argmax(1)]
    save_submission(test_df["id"], test_preds, f"submission_autogluon{suffix}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon benchmark for stellar classification")
    parser.add_argument("--time-limit", type=int, default=2400,
                        help="AutoGluon fit time limit in seconds (default 2400)")
    parser.add_argument("--preset", type=str, default=None,
                        help="AutoGluon preset (e.g. best_quality). Omit for the explicit GBDT stack.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fast bagged-GBM-only fit to validate the pipeline")
    args = parser.parse_args()
    train_and_evaluate(time_limit=args.time_limit, preset=args.preset, dry_run=args.dry_run)
