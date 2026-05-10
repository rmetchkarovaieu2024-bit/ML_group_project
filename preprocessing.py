"""
Preprocessing and feature engineering for the Bechdel dataset.

All transformations that depend on dataset statistics are handled inside
a Pipeline to avoid data leakage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOP_GENRES = [
    "Drama", "Comedy", "Action", "Romance", "Thriller", "Horror",
    "Sci-Fi", "Adventure", "Crime", "Fantasy", "Mystery", "War",
    "Animation", "Family", "Biography",
]

NUMERIC_FEATURES = [
    "year", "runtime", "log_budget", "log_revenue",
    "imdb_rating", "log_num_votes", "num_genres",
]
BINARY_FEATURES = [f"genre_{g.lower().replace('-', '_')}" for g in TOP_GENRES]
CATEGORICAL_FEATURES = ["decade"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:     # Stateless feature engineering

    df = df.copy()

    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_num_votes"] = np.log1p(df["num_votes"])

    df["decade"] = (df["year"] // 10 * 10).astype("Int64").astype(str) + "s"

    genre_lists = df["genres"].fillna("").str.split("|")
    for g in TOP_GENRES:
        col = f"genre_{g.lower().replace('-', '_')}"
        df[col] = genre_lists.apply(lambda gs: int(g in gs))

    df["num_genres"] = genre_lists.apply(lambda gs: len([x for x in gs if x]))

    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    from data_loader import load_synthetic_data
    df = load_synthetic_data(n=500, seed=0)
    df = engineer_features(df)
    print("Engineered columns:", [c for c in df.columns if c not in
        ["imdb_id", "title", "year", "runtime", "budget", "revenue",
         "imdb_rating", "num_votes", "genres", "bechdel_score", "bechdel_pass"]][:10], "...")
    pre = build_preprocessor()
    feat_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    X = pre.fit_transform(df[feat_cols])
    print("Transformed shape:", X.shape)