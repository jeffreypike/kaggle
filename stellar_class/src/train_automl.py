"""FLAML AutoML probe: which gradient-boosting library wins on this data?

FLAML searches lgbm/xgboost/catboost on a fast internal holdout (statistically stable
at ~577k rows, and ~5x more trials per minute than 5-fold CV search). The best estimator
is then refit per fold (via sklearn.clone, so its exact tuned hyperparameters carry over
without manual reconstruction) to produce OOF predictions on the standardized folds for a
fair comparison with the other models.

Run from anywhere:
    python stellar_class/src/train_automl.py --time-budget 600
    python stellar_class/src/train_automl.py --dry-run     # ~10s smoke test
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from flaml import AutoML

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import (load_data_with_folds, get_custom_cv, evaluate_predictions,
                        save_oof_predictions, save_submission, tune_class_weights, DATA_DIR)
from features import prepare_features, encode_target

ESTIMATOR_LIST = ["lgbm", "xgboost", "catboost"]


def balanced_accuracy_metric(X_val, y_val, estimator, labels, X_train, y_train,
                             weight_val=None, weight_train=None, *args):
    """Custom FLAML metric: minimize (1 - balanced accuracy), the competition target."""
    score = balanced_accuracy_score(y_val, estimator.predict(X_val))
    return 1 - score, {"balanced_accuracy": score}


def train_and_evaluate(time_budget=300, dry_run=False):
    print("=== Step 1: Loading data with standardized folds ===")
    train_df = load_data_with_folds()
    test_df = pl.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train_df.shape} (with fold col)   Test: {test_df.shape}")

    print("\n=== Step 2: Feature engineering (shared) ===")
    X, X_test = prepare_features(train_df, test_df)
    y, le, classes = encode_target(train_df)
    print(f"Classes: {classes}")
    cv = get_custom_cv(train_df)

    budget = 10 if dry_run else time_budget
    print(f"\n=== Step 3: FLAML search (balanced accuracy, holdout, budget {budget}s) ===")
    automl = AutoML()
    automl.fit(
        X_train=X, y_train=y,
        task="multiclass",
        metric=balanced_accuracy_metric,
        estimator_list=ESTIMATOR_LIST,
        eval_method="holdout",  # fast search; final OOF uses standardized folds below
        split_ratio=0.1,
        time_budget=budget,
        seed=42,
        verbose=1,
    )
    print("\n=== Search complete ===")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config   : {automl.best_config}")
    print(f"Best CV balanced accuracy: {1 - automl.best_loss:.5f}")

    if dry_run:
        print("\n[DRY-RUN] Skipping per-fold refit and submission.")
        return

    print("\n=== Step 4: Refitting best estimator per fold for OOF predictions ===")
    oof_probs = np.zeros((len(X), len(classes)))
    test_probs = np.zeros((len(X_test), len(classes)))
    for fold, (train_idx, val_idx) in enumerate(cv):
        print(f"--- fold {fold} ---")
        model = clone(automl.model.estimator)  # tuned hyperparameters, unfitted
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_probs[val_idx] = model.predict_proba(X.iloc[val_idx])
        test_probs += model.predict_proba(X_test) / len(cv)

    print("\n=== Step 5: Evaluating OOF performance ===")
    evaluate_predictions(y, oof_probs, classes)

    weights, tuned = tune_class_weights(y, oof_probs)
    print(f"Balanced accuracy after class-weight tuning: {tuned:.5f}  "
          f"(weights={dict(zip(classes, weights.round(3)))})")

    save_oof_predictions(oof_probs, f"flaml_{automl.best_estimator}")
    test_preds = le.inverse_transform((test_probs * weights).argmax(1))
    save_submission(test_df["id"], test_preds, "submission_automl.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLAML baseline for stellar classification")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="FLAML search budget in seconds (default 300)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fast ~10s search, no refit/submission")
    args = parser.parse_args()
    train_and_evaluate(time_budget=args.time_budget, dry_run=args.dry_run)
