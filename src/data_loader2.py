"""
data_loader.py
==============
Loads and merges the two datasets used in this project:

  Bechdel_IMDB_Merge0524.csv   — 9 718 films with Bechdel ratings + IMDb metadata
  movies_enriched_5k.csv       — 5 000 films enriched from TMDb (budget, cast, plot…)

The two files share IMDb IDs and are left-joined so all 9 718 Bechdel-rated
films are preserved. ~2 100 (~22 %) will gain TMDb enrichment; the rest get
NaN for budget/revenue/popularity and empty string for plot — both handled by
the sklearn preprocessor downstream.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Female-name heuristic for has_female_director
# ---------------------------------------------------------------------------
_FEMALE_NAMES2 = frozenset(
    "amy,angelina,anna,anne,ava,barbara,bonnie,brenda,carol,caroline,"
    "catherine,chloe,claire,claudia,danielle,diana,elizabeth,ella,emily,"
    "emma,eva,florence,greta,hannah,helen,ida,jacqueline,jane,jennifer,"
    "jessica,joanna,jodie,julia,julie,karen,kate,katherine,kathryn,"
    "kimberly,laura,linda,lisa,lynne,maggie,maria,mary,melissa,michelle,"
    "mira,miriam,nancy,natalie,nicole,olivia,paula,rachel,rebecca,rita,"
    "robin,sally,samantha,sarah,sofia,sophia,stephanie,susan,tamara,"
    "teresa,tina,vanessa,vera,victoria,virginia,wendy,yvonne,zoe"
    .split(",")
)


def _director_is_female2(s):
    if not isinstance(s, str) or not s.strip():
        return 0
    for name2 in re.split(r"[|,;/]", s):
        first2 = name2.strip().split()[0].lower() if name2.strip() else ""
        if first2 in _FEMALE_NAMES2:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data2(
    bechdel_path2: str = ".../data/Bechdel_IMDB_Merge0524.csv",
    tmdb_path2: str | None = ".../data/movies_enriched_5k.csv",
    complete_path2: str | None = None,
) -> pd.DataFrame:
    """
    Load and merge both CSVs into a single modelling-ready DataFrame.

    Parameters
    ----------
    bechdel_path2  : path to Bechdel_IMDB_Merge0524.csv
    tmdb_path2     : path to movies_enriched_5k.csv (used when complete_path2
                    is not available)
    complete_path2 : path to movies_complete.csv produced by
                    data/build_complete_dataset.py — if it exists this is
                    used directly (all 9,718 movies, fully enriched).

    Returns
    -------
    DataFrame with columns:
        imdb_id, title, year, runtime, imdb_rating, num_votes,
        bechdel_score, bechdel_pass, genres,
        budget, revenue, tmdb_popularity, cast_size, has_female_director, plot
    """
    import os
    if complete_path2 is not None and os.path.exists(complete_path2):
        df = _load_complete2(complete_path2)
        print(f"[data_loader] Loaded {len(df):,} films from complete dataset.")
        return df

    df = _load_bechdel2(bechdel_path2)

    if tmdb_path2 is not None:
        tmdb = _load_tmdb2(tmdb_path2 )
        df = _merge2(df, tmdb)

    return df


def load_synthetic_data2(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    """Synthetic fallback with identical schema for pipeline development."""
    rng2 = np.random.default_rng(seed)
    genres_pool2 = ["Action", "Comedy", "Drama", "Horror", "Romance",
                    "Sci-Fi", "Thriller", "Fantasy", "War", "Biography"]

    def rand_genres2():
        k2 = rng2.integers(1, 4)
        return "|".join(rng2.choice(genres_pool2, size=k2, replace=False))

    years2 = rng2.integers(1970, 2024, size=n)
    budgets2 = np.where(rng2.random(n) < 0.78, np.nan, rng2.lognormal(16, 2, n))
    revenues2 = np.where(rng2.random(n) < 0.78, np.nan, rng2.lognormal(17, 2, n))
    popularity2 = np.where(rng2.random(n) < 0.78, np.nan, rng2.lognormal(2, 1.5, n).clip(0.1, 500))
    cast_size2 = np.where(rng2.random(n) < 0.78, np.nan, rng2.integers(1, 20, n).astype(float))
    bechdel_pass2 = rng2.binomial(1, 0.576, n)

    return pd.DataFrame({
        "imdb_id":            [f"tt{i:07d}" for i in range(1, n + 1)],
        "title":              [f"Film_{i}" for i in range(1, n + 1)],
        "year":               years2,
        "runtime":            rng2.normal(105, 20, n).clip(60, 240).round().astype(float),
        "imdb_rating":        rng2.uniform(4.0, 9.5, n).round(1),
        "num_votes":          rng2.lognormal(9, 2, n).astype(float),
        "bechdel_score":      np.where(bechdel_pass2, 3, rng2.integers(0, 3, n)),
        "bechdel_pass":       bechdel_pass2,
        "genres":             [rand_genres2() for _ in range(n)],
        "budget":             budgets2,
        "revenue":            revenues2,
        "tmdb_popularity":    popularity2,
        "cast_size":          cast_size2,
        "has_female_director": rng2.binomial(1, 0.12, n).astype(float),
        "plot":               ["" for _ in range(n)],
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_bechdel2(path: str) -> pd.DataFrame:
    df_raw2 = pd.read_csv(path)

    # Rename to internal schema
    df2 = df_raw2.rename(columns={
        "imdbid":             "imdb_id",
        "bechdelRating":      "bechdel_score",
        "imdbAverageRating":  "imdb_rating",
        "numVotes":           "num_votes",
        "runtimeMinutes":     "runtime",
        "title":              "title",
        "year":               "year",
    })

    # imdb_id: stored as float (e.g. 9.0) → zero-padded "tt0000009"
    df2["imdb_id"] = df2["imdb_id"].apply(
        lambda x: f"tt{int(float(x)):07d}" if pd.notna(x) else ""
    )

    # Combine genre1/genre2/genre3 → pipe-separated genres string
    g_cols2 = [c for c in ["genre1", "genre2", "genre3"] if c in df2.columns]
    df2["genres"] = (
        df2[g_cols2].fillna("")
        .apply(lambda r: "|".join(g.strip() for g in r if g.strip()), axis=1)
    )
    df2 = df2.drop(columns=g_cols2, errors="ignore")

    # Numeric coercion
    df2["runtime"] = pd.to_numeric(df2["runtime"], errors="coerce")
    df2["num_votes"] = pd.to_numeric(df2["num_votes"], errors="coerce")
    df2["imdb_rating"] = pd.to_numeric(df2["imdb_rating"], errors="coerce")
    df2["year"] = pd.to_numeric(df2["year"], errors="coerce")

    # Target
    df2["bechdel_score"] = pd.to_numeric(df2["bechdel_score"], errors="coerce")
    df2["bechdel_pass"] = (df2["bechdel_score"] == 3).astype(int)

    keep2 = ["imdb_id", "title", "year", "runtime", "imdb_rating",
             "num_votes", "bechdel_score", "bechdel_pass", "genres"]
    return df2[[c for c in keep2 if c in df2.columns]]


def _load_tmdb2(path: str) -> pd.DataFrame:
    t2 = pd.read_csv(path)

    # Normalise column names
    t2.columns = t2.columns.str.strip()

    # imdb_id — already in tt-format
    t2 = t2.rename(columns={"IMDb_ID": "imdb_id"})
    t2["imdb_id"] = t2["imdb_id"].astype(str).str.strip()

    # Budget / Revenue: 0 means missing in this dataset
    for col2 in ("Budget", "Revenue"):
        t2[col2] = pd.to_numeric(t2[col2], errors="coerce")
        t2.loc[t2[col2] == 0, col2] = np.nan
    t2 = t2.rename(columns={"Budget": "budget", "Revenue": "revenue",
                             "Popularity": "tmdb_popularity"})

    # cast_size from Actors column (comma-separated)
    t2["cast_size"] = (
        t2["Actors"].fillna("")
        .apply(lambda x: len([a for a in x.split(",") if a.strip()]))
        .replace(0, np.nan)
    )

    # has_female_director
    t2["has_female_director"] = t2["Director"].apply(_director_is_female2).astype(float)

    # plot text
    t2["plot"] = t2["Plot"].fillna("")

    keep2 = ["imdb_id", "budget", "revenue", "tmdb_popularity",
             "cast_size", "has_female_director", "plot"]
    return t2[keep2].drop_duplicates(subset="imdb_id")


def _merge2(base: pd.DataFrame, tmdb: pd.DataFrame) -> pd.DataFrame:
    merged2 = base.merge(tmdb, on="imdb_id", how="left")

    n_enriched2 = merged2["tmdb_popularity"].notna().sum()
    n_total2    = len(merged2)
    print(f"[data_loader] Loaded {n_total2:,} Bechdel-rated films.")
    print(f"[data_loader] TMDb enrichment matched {n_enriched2:,} / {n_total2:,} "
          f"({100*n_enriched2/n_total2:.1f}%). "
          f"Remaining {n_total2-n_enriched2:,} rows will use imputed values.")

    merged2["plot"] = merged2["plot"].fillna("")
    return merged2


def _load_complete2(path: str) -> pd.DataFrame:
    """Load the pre-built complete dataset (output of build_complete_dataset.py)."""
    df2 = pd.read_csv(path)
    for col2 in ("runtime", "num_votes", "imdb_rating", "year",
                 "bechdel_score", "budget", "revenue", "tmdb_popularity",
                 "cast_size", "has_female_director"):
        if col2 in df2.columns:
            df2[col2] = pd.to_numeric(df2[col2], errors="coerce")

    # Drop movies without a Bechdel label — the 1,619 enriched-only movies
    # have no ground-truth rating and must not enter the training set.
    n_before2 = len(df2)
    df2 = df2[df2["bechdel_score"].notna()].copy()
    n_dropped2 = n_before2 - len(df2)
    if n_dropped2:
        print(f"[data_loader] Dropped {n_dropped2:,} unlabelled movies "
              f"(no Bechdel rating). {len(df2):,} labelled films retained.")

    df2["bechdel_pass"] = (df2["bechdel_score"] == 3).astype(int)
    df2["plot"] = df2["plot"].fillna("") if "plot" in df2.columns else ""
    if "tmdb_popularity" not in df2.columns:
        df2["tmdb_popularity"] = np.nan
    return df2
