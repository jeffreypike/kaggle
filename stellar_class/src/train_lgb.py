"""Manual LightGBM baseline on the standardized folds.

Run from anywhere:  python stellar_class/src/train_lgb.py
"""
import os
import sys

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation import (load_data_with_folds, get_custom_cv, evaluate_predictions,
                        save_oof_predictions, save_submission, tune_class_weights, DATA_DIR,
                        PREDICTIONS_DIR)
from features import prepare_features, encode_target, CAT_COLS

LGB_PARAMS = {
    "n_estimators": 2000,   # capped by early stopping below
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def train_and_evaluate():
    print("=== Loading data with standardized folds ===")
    train_df = load_data_with_folds()
    test_df = pl.read_csv(DATA_DIR / "test.csv")

    print("=== Feature engineering (shared) ===")
    X, X_test = prepare_features(train_df, test_df)
    y, le, classes = encode_target(train_df)
    cv = get_custom_cv(train_df)

    print("\n=== Training LightGBM with per-fold early stopping ===")
    oof_probs = np.zeros((len(X), len(classes)))
    test_probs = np.zeros((len(X_test), len(classes)))

    for fold, (train_idx, val_idx) in enumerate(cv):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            categorical_feature=CAT_COLS,
            callbacks=[early_stopping(50, verbose=False)],
        )
        oof_probs[val_idx] = model.predict_proba(X_val)
        test_probs += model.predict_proba(X_test) / len(cv)
        print(f"--- fold {fold}: best_iteration={model.best_iteration_} ---")

    print("\n=== Evaluating OOF performance ===")
    evaluate_predictions(y, oof_probs, classes)

    weights, tuned = tune_class_weights(y, oof_probs)
    print(f"Balanced accuracy after class-weight tuning: {tuned:.5f}  "
          f"(weights={dict(zip(classes, weights.round(3)))})")

    save_oof_predictions(oof_probs, "lgb")
    np.save(PREDICTIONS_DIR / "test_lgb.npy", test_probs)   # for blended submissions
    test_preds = le.inverse_transform((test_probs * weights).argmax(1))
    save_submission(test_df["id"], test_preds, "submission_lgb.csv")


if __name__ == "__main__":
    train_and_evaluate()
