"""Shared validation utilities: standardized folds, metrics, OOF/submission I/O.

All models load folds via `load_data_with_folds` so their out-of-fold predictions live
on identical splits and are directly comparable / stackable. Paths are resolved relative
to this file, so scripts work regardless of the current working directory.
"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, confusion_matrix)

PROJECT_DIR = Path(__file__).resolve().parent.parent  # stellar_class/
DATA_DIR = PROJECT_DIR / "data"
PREDICTIONS_DIR = PROJECT_DIR / "predictions"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"


class PredefinedFoldsSplitter(BaseCrossValidator):
    """Scikit-learn CV splitter driven by a precomputed integer fold column."""

    def __init__(self, folds):
        self.folds = np.asarray(folds)
        self.n_splits = int(self.folds.max() + 1)

    def split(self, X=None, y=None, groups=None):
        for f in range(self.n_splits):
            yield np.where(self.folds != f)[0], np.where(self.folds == f)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def load_data_with_folds(data_dir=DATA_DIR, n_splits=5, seed=42):
    """Load train.csv and append a deterministic stratified `fold` column."""
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find training data at {train_path}")

    df = pl.read_csv(train_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=np.int32)
    y = df["class"].to_numpy()
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        folds[val_idx] = fold_idx
    return df.with_columns(pl.Series("fold", folds))


def get_custom_cv(df):
    """Return [(train_idx, val_idx), ...] from the `fold` column."""
    return list(PredefinedFoldsSplitter(df["fold"].to_numpy()).split())


def evaluate_predictions(y_true, y_pred_proba, classes):
    """Print and return balanced accuracy (competition metric) + supporting metrics."""
    labels = list(range(len(classes)))
    y_pred = y_pred_proba.argmax(1)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    loss = log_loss(y_true, y_pred_proba, labels=labels)

    print("\n" + "=" * 40)
    print("=== MODEL PERFORMANCE EVALUATION ===")
    print("=" * 40)
    print(f"Balanced Accuracy : {bal_acc:.5f}  <- Competition Metric")
    print(f"Accuracy          : {acc:.5f}")
    print(f"Macro-F1          : {macro_f1:.5f}")
    print(f"Log Loss          : {loss:.5f}")
    print("-" * 40)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    print("Row-Normalized Confusion Matrix (Recall):")
    hdr = "True \\ Pred"  # kept out of the f-string expr (backslash-in-{} needs py3.12+)
    print(f"{hdr:<12} | " + " | ".join(f"{c:<8}" for c in classes))
    print("-" * (15 + 11 * len(classes)))
    for i, true_class in enumerate(classes):
        row = " | ".join(f"{cm[i, j]:.4f}" for j in range(len(classes)))
        print(f"{true_class:<12} | {row}")
    print("=" * 40 + "\n")

    return {"balanced_accuracy": bal_acc, "accuracy": acc, "macro_f1": macro_f1,
            "log_loss": loss, "confusion_matrix": cm}


def tune_class_weights(y_true, oof_proba, n_rounds=3, grid=None):
    """Coordinate-ascent search for per-class probability multipliers that maximize
    balanced accuracy on OOF predictions.

    Under class imbalance, plain argmax under-predicts rare classes, which balanced
    accuracy punishes. Re-weighting the posteriors before argmax recovers minority
    recall. Returns (weights, tuned_balanced_accuracy); apply as `(proba * weights).argmax(1)`.
    """
    n_classes = oof_proba.shape[1]
    if grid is None:
        grid = np.linspace(0.2, 3.0, 29)
    w = np.ones(n_classes)
    best = balanced_accuracy_score(y_true, oof_proba.argmax(1))
    for _ in range(n_rounds):
        for k in range(n_classes):
            best_wk = w[k]
            for g in grid:
                w[k] = g
                score = balanced_accuracy_score(y_true, (oof_proba * w).argmax(1))
                if score > best:
                    best, best_wk = score, g
            w[k] = best_wk
    return w / w.mean(), best


def save_oof_predictions(oof_probs, model_name, predictions_dir=PREDICTIONS_DIR):
    """Save OOF probability array as predictions/oof_<model_name>.npy."""
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"oof_{model_name}.npy"
    np.save(out_path, oof_probs)
    print(f"Saved OOF predictions for '{model_name}' to {out_path}")


def load_oof_predictions(model_name, predictions_dir=PREDICTIONS_DIR):
    in_path = Path(predictions_dir) / f"oof_{model_name}.npy"
    if not in_path.exists():
        raise FileNotFoundError(f"No OOF predictions found at {in_path}")
    return np.load(in_path)


def save_submission(test_ids, pred_labels, filename, submissions_dir=SUBMISSIONS_DIR):
    """Write a Kaggle submission CSV with columns id, class."""
    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = submissions_dir / filename
    pl.DataFrame({"id": test_ids, "class": pred_labels}).write_csv(out_path)
    print(f"Saved submission file to: {out_path}")
    return out_path
