cat > ~/Downloads/bechdel_ml/src/data_loader.py << 'EOF'
"""
Data loader for the Bechdel-prediction project.
Provides two sources: real Kaggle CSV or synthetic fallback.
Both return the same column contract so downstream code works either way.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Columns every downstream function expects
EXPECTED_COLS = [
    "imdb_id", "title", "year", "runtime", "budget", "revenue",
    "imdb_rating", "num_votes", "genres", "bechdel_score", "bechdel_pass",
]


def load_real_data(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle CSV and normalize column names to the contract above."""
    df = pd.read_csv(path)

    # Map Kaggle/FiveThirtyEight column names to our standard names
    rename_map = {
        "tconst":             "imdb_id",
        "imdbID":             "imdb_id",
        "imdbid":             "imdb_id",
        "primaryTitle":       "title",
        "startYear":          "year",
        "runtimeMinutes":     "runtime",
        "averageRating":      "imdb_rating",
        "imdbAverageRating":  "imdb_rating",
        "numVotes":           "num_votes",
        "rating":             "bechdel_score",
        "bechdelRating":      "bechdel_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Merge genre1/genre2/genre3 into one pipe-separated column
    if "genres" not in df.columns and "genre1" in df.columns:
        gcols = [c for c in ["genre1", "genre2", "genre3"] if c in df.columns]
        df["genres"] = df[gcols].apply(
            lambda r: "|".join(v for v in r if pd.notna(v) and v != ""), axis=1)

    # Add budget/revenue as NaN if the CSV doesn't have them
    for col in ["budget", "revenue"]:
        if col not in df.columns:
            df[col] = float("nan")

    # Derive binary target from ordinal score if needed
    if "bechdel_pass" not in df.columns and "bechdel_score" in df.columns:
        df["bechdel_pass"] = (df["bechdel_score"] == 3).astype(int)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Extend rename_map.")

    return df[EXPECTED_COLS].copy()


def load_synthetic_data(n: int = 9_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset for pipeline development only.
    Do NOT report metrics from this data — use the real CSV for results.
    """
    rng = np.random.default_rng(seed)

    # Year: skewed toward recent decades
    year = rng.choice(np.arange(1970, 2024), size=n,
                      p=_triangular_probs(1970, 2024, peak=2015))

    # Runtime: ~105 min average, clipped to realistic range
    runtime = np.clip(rng.normal(105, 18, n), 60, 240).round(0)

    # Budget: log-normal, ~30% missing (mirrors TMDb gaps)
    budget = np.exp(rng.normal(16, 1.8, n))
    budget[rng.random(n) < 0.30] = np.nan

    # Revenue: correlated with budget, ~25% missing
    revenue = budget * rng.lognormal(0.3, 0.9, n)
    revenue[rng.random(n) < 0.25] = np.nan

    # IMDb rating: Beta distribution scaled to 1-10
    imdb_rating = (rng.beta(6, 3, n) * 9 + 1).round(1)

    # Vote count: log-normal heavy tail
    num_votes = np.exp(rng.normal(9, 1.8, n)).round(0)

    # Genres: 1-3 per film, pipe-separated
    genre_pool = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Horror",
                  "Sci-Fi", "Adventure", "Crime", "Fantasy", "Mystery", "War",
                  "Animation", "Family", "Biography"]
    genres = ["|".join(rng.choice(genre_pool, size=rng.integers(1, 4), replace=False))
              for _ in range(n)]

    # Bechdel label: logistic model with year, genre, and budget effects
    logit = np.zeros(n)
    logit += (year - 1995) * 0.015                          # newer = more likely to pass
    for i, g in enumerate(genres):
        if "Romance" in g or "Drama" in g:   logit[i] += 0.5
        if "Action"  in g or "War"   in g:   logit[i] -= 0.6
        if "Horror"  in g:                   logit[i] += 0.15
        if "Animation" in g and "Family" in g: logit[i] -= 0.2
    log_budget = np.log(np.where(np.isnan(budget), np.nanmedian(budget), budget))
    logit += -(log_budget - np.nanmean(log_budget)) * 0.15  # bigger budget = less likely
    logit += -0.25                                           # intercept → ~45% pass rate

    prob_pass    = 1 / (1 + np.exp(-logit))
    bechdel_pass = (rng.random(n) < prob_pass).astype(int)

    # Ordinal score: 3 if pass, else 0/1/2
    bechdel_score = np.where(bechdel_pass == 1, 3,
                             rng.choice([0, 1, 2], size=n, p=[0.25, 0.35, 0.40]))

    df = pd.DataFrame({
        "imdb_id":      [f"tt{i:07d}" for i in range(1, n + 1)],
        "title":        [f"Film_{i}"  for i in range(1, n + 1)],
        "year":         year.astype(int),
        "runtime":      runtime,
        "budget":       budget,
        "revenue":      revenue,
        "imdb_rating":  imdb_rating,
        "num_votes":    num_votes,
        "genres":       genres,
        "bechdel_score": bechdel_score.astype(int),
        "bechdel_pass": bechdel_pass,
    })

    # Add small missingness to runtime/rating/votes to mirror real data
    for col, frac in [("runtime", 0.02), ("imdb_rating", 0.04), ("num_votes", 0.04)]:
        df.loc[rng.random(n) < frac, col] = np.nan

    return df


def _triangular_probs(lo: int, hi: int, peak: int) -> np.ndarray:
    """Triangular probability weights across a year range."""
    years = np.arange(lo, hi)
    weights = np.where(years <= peak,
                       (years - lo + 1) / (peak - lo + 1),
                       (hi - years) / (hi - peak))
    return weights / weights.sum()


if __name__ == "__main__":
    df = load_synthetic_data(n=1000, seed=0)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nPass rate:", df["bechdel_pass"].mean().round(3))
    print("\nMissingness:\n", df.isna().mean().round(3))
EOF
