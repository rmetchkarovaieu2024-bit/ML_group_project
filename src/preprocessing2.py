"""
preprocessing.py
================
Feature engineering and sklearn preprocessor for the Bechdel ML pipeline.

Features produced
-----------------
Numeric (always):
  year, log_runtime, imdb_rating, log_num_votes

Numeric (when TMDb enrichment present, ~22% coverage, median-imputed):
  log_budget, log_revenue, tmdb_popularity, log_cast_size

Binary (always):
  has_female_director   (0/1, NaN→0 when missing)
  genre_action … genre_western  (one-hot from genres string)

Categorical:
  decade   (floor decade as string: "1990", "2000", …)

Text (optional, when use_plot_tfidf=True):
  plot → TF-IDF (max 150 features, min_df=3)
  Only meaningful when TMDb enrichment is loaded (~22% non-empty).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Canonical genre list (derived from the real data)
# ---------------------------------------------------------------------------
GENRE_LIST2 = [
    "action", "adventure", "animation", "biography", "comedy", "crime",
    "documentary", "drama", "fantasy", "history", "horror", "music",
    "musical", "mystery", "romance", "sci-fi", "sport", "thriller",
    "war", "western",
]

BINARY_FEATURES2 = [f"genre_{g.replace('-','_').replace(' ','_')}" for g in GENRE_LIST2] + \
                  ["has_female_director"]

NUMERIC_BASE2 = ["year", "log_runtime", "imdb_rating", "log_num_votes"]
NUMERIC_TMDB2 = ["log_budget", "log_revenue", "tmdb_popularity", "log_cast_size"]
NUMERIC_FEATURES2 = NUMERIC_BASE2 + NUMERIC_TMDB2

CATEGORICAL_FEATURES2 = ["decade"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features2(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all derived feature columns added."""
    df = df.copy()

    # --- Log transforms ---
    df["log_runtime"]   = np.log1p(pd.to_numeric(df.get("runtime"),   errors="coerce").clip(0))
    df["log_num_votes"] = np.log1p(pd.to_numeric(df.get("num_votes"), errors="coerce").clip(0))

    for raw2, col2 in [("budget", "log_budget"), ("revenue", "log_revenue"),
                       ("cast_size", "log_cast_size")]:
        if raw2 in df.columns:
            vals2 = pd.to_numeric(df[raw2], errors="coerce").clip(lower=0)
            vals2[vals2 == 0] = np.nan
            df[col2] = np.log1p(vals2)
        else:
            df[col2] = np.nan

    if "tmdb_popularity" not in df.columns:
        df["tmdb_popularity"] = np.nan
    else:
        df["tmdb_popularity"] = np.log1p(
            pd.to_numeric(df["tmdb_popularity"], errors="coerce").clip(lower=0)
        )

    # --- Decade ---
    # Clip to 1920–2020: the ~15 pre-1920 films get bucketed into "1920"
    # so every decade value seen at test time also appears in training folds,
    # eliminating the OHE unknown-categories warning.
    decade_val2 = (pd.to_numeric(df["year"], errors="coerce") // 10 * 10).fillna(1920).astype(int)
    df["decade"] = decade_val2.clip(lower=1920, upper=2020).astype(str)

    # --- Genre one-hots from pipe-separated genres string ---
    genres_str2 = df.get("genres", pd.Series([""] * len(df))).fillna("")
    for genre2 in GENRE_LIST2:
        col2 = f"genre_{genre2.replace('-','_').replace(' ','_')}"
        pattern2 = genre2.replace("-", r"[\-]?").replace("fi", "fi")
        df[col2] = genres_str2.str.lower().str.contains(
            genre2.replace("-", r"[\-]?"), regex=True, na=False
        ).astype(int)

    # --- has_female_director: NaN → 0 ---
    if "has_female_director" in df.columns:
        df["has_female_director"] = pd.to_numeric(
            df["has_female_director"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["has_female_director"] = 0

    # --- plot: ensure string ---
    if "plot" not in df.columns:
        df["plot"] = ""
    df["plot"] = df["plot"].fillna("")

    # Guard: any inf produced by log transforms becomes NaN for the imputer
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


def get_feature_lists2(df: pd.DataFrame):
    """Return (numeric, categorical, binary) lists for columns present in df."""
    numeric2 = [c for c in NUMERIC_FEATURES2 if c in df.columns]
    binary2  = [c for c in BINARY_FEATURES2  if c in df.columns]
    return numeric2, list(CATEGORICAL_FEATURES2), binary2


# ---------------------------------------------------------------------------
# ColumnSelector helper for TF-IDF branch
# ---------------------------------------------------------------------------

class ColumnSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, column): self.column = column
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.column]


# ---------------------------------------------------------------------------
# Dense + TF-IDF union
# ---------------------------------------------------------------------------

class _DensePlotUnion2(BaseEstimator, TransformerMixin):
    """
    Horizontally stack the dense ColumnTransformer with a TF-IDF block.

    The TF-IDF output is passed through its own StandardScaler before stacking
    so that all features are on the same scale.  Without this, lbfgs-based
    logistic regression hits matmul overflow because TF-IDF values are in a
    different range than the already-standardised dense features.
    """
    def __init__(self, dense_ct, tfidf_pipe):
        self.dense_ct2   = dense_ct
        self.tfidf_pipe2 = tfidf_pipe

    def fit(self, X, y=None):
        self.dense_ct2.fit(X, y)
        tfidf_out2 = self.tfidf_pipe2.fit_transform(X, y)
        if hasattr(tfidf_out2, "toarray"):
            tfidf_out2 = tfidf_out2.toarray()
        # Create fresh scaler here (not in __init__) so sklearn.base.clone()
        # doesn't lose it — clone only preserves __init__ parameters.
        self._tfidf_scaler2 = StandardScaler(with_mean=False)
        self._tfidf_scaler2.fit(tfidf_out2)
        return self

    def transform(self, X):
        dense2 = self.dense_ct2.transform(X)
        tfidf2 = self.tfidf_pipe2.transform(X)
        if hasattr(tfidf2, "toarray"):
            tfidf2 = tfidf2.toarray()
        tfidf2 = self._tfidf_scaler2.transform(tfidf2)
        return np.hstack([dense2, tfidf2])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        dense_names2 = list(self.dense_ct2.get_feature_names_out())
        tfidf_names2 = [
            f"plot_{t}"
            for t in self.tfidf_pipe2.named_steps["tfidf"].get_feature_names_out()
        ]
        return np.array(dense_names2 + tfidf_names2)


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def build_preprocessor2(
    numeric_features=None,
    categorical_features=None,
    binary_features=None,
    use_plot_tfidf: bool = False,
    tfidf_max_features: int = 150,
):
    """
    Build the sklearn preprocessing transformer.

    Parameters
    ----------
    numeric_features, categorical_features, binary_features:
        Pass output from get_feature_lists(). Falls back to module constants.
    use_plot_tfidf : bool
        If True, append a TF-IDF block on the `plot` column.
        Only worth enabling when TMDb enrichment is loaded.
    tfidf_max_features : int
        Vocabulary cap for TF-IDF.
    """
    if numeric_features     is None: numeric_features     = NUMERIC_FEATURES2
    if categorical_features is None: categorical_features = CATEGORICAL_FEATURES2
    if binary_features      is None: binary_features      = BINARY_FEATURES2

    dense_ct2 = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False,
                                         drop="first")),
            ]), categorical_features),
            ("bin", SimpleImputer(strategy="constant", fill_value=0),
             binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if not use_plot_tfidf:
        return dense_ct2

    tfidf_pipe2 = Pipeline([
        ("sel",   ColumnSelector2("plot")),
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_max_features,
            sublinear_tf=True,
            min_df=3,
            ngram_range=(1, 2),
            stop_words="english",
        )),
    ])

    return _DensePlotUnion2(dense_ct2, tfidf_pipe2)
