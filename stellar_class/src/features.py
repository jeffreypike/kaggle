"""Shared feature engineering and column definitions for the stellar-class models.

Keeping these in one place ensures every model trains on identical features and an
identical, train/test-consistent categorical encoding (see `prepare_features`).
"""
import polars as pl
from sklearn.preprocessing import LabelEncoder

NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_COLS = ["u_g", "g_r", "r_i", "i_z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
FEAT_COLS = NUM_COLS + COLOR_COLS + CAT_COLS
TARGET = "class"


def add_colors(df: pl.DataFrame) -> pl.DataFrame:
    """Add astronomy color indices (differences between adjacent photometric bands)."""
    return df.with_columns([
        (pl.col("u") - pl.col("g")).alias("u_g"),
        (pl.col("g") - pl.col("r")).alias("g_r"),
        (pl.col("r") - pl.col("i")).alias("r_i"),
        (pl.col("i") - pl.col("z")).alias("i_z"),
    ])


def prepare_features(train_df: pl.DataFrame, test_df: pl.DataFrame):
    """Build aligned train/test feature frames.

    Categoricals are integer-encoded with a *single shared mapping* built from the union
    of train and test levels, so the codes can never silently disagree between the two
    sets (which would corrupt tree splits). Returns pandas frames ready for any
    sklearn/LightGBM/XGBoost estimator.
    """
    train_df = add_colors(train_df)
    test_df = add_colors(test_df)

    X = train_df.select(FEAT_COLS).to_pandas()
    X_test = test_df.select(FEAT_COLS).to_pandas()

    for c in CAT_COLS:
        levels = sorted(set(X[c].unique()).union(X_test[c].unique()))
        mapping = {level: code for code, level in enumerate(levels)}
        X[c] = X[c].map(mapping).astype("int32")
        X_test[c] = X_test[c].map(mapping).astype("int32")

    return X, X_test


def encode_target(train_df: pl.DataFrame):
    """Label-encode the target. Returns (y, label_encoder, class_names).

    Classes are alphabetical: ['GALAXY', 'QSO', 'STAR'] -> [0, 1, 2]. All saved OOF /
    test probability arrays use this column order.
    """
    le = LabelEncoder()
    y = le.fit_transform(train_df[TARGET].to_numpy())
    return y, le, list(le.classes_)
