"""
Bechdel score enrichment — verified sources only.

All URLs below were tested and confirmed live as of May 2026.

Step 1: 3 bulk CSV sources (combined ~8,800 unique films)
Step 2: per-movie API for anything still unmatched

Run:  python add_bechdel_scores.py
Deps: pip install pandas requests
"""

import pandas as pd
import requests
import time
from io import StringIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_id(val):
    """Normalize any IMDb ID to 'tt0123456' format."""
    s = str(val).strip().lower().replace('tt', '')
    try:
        return f"tt{int(s):07d}"
    except ValueError:
        return f"tt{s.zfill(7)}"


CLEAN_TEST_MAP = {
    'ok': 3, 'ok-disagree': 3, 'dubious': 3, 'dubious-disagree': 3,
    'men': 2, 'men-disagree': 2,
    'notalk': 1, 'notalk-disagree': 1,
    'nowomen': 0, 'nowomen-disagree': 0,
}


# ---------------------------------------------------------------------------
# Verified bulk CSV sources
# ---------------------------------------------------------------------------

BULK_SOURCES = [
    {
        # ~8,800 films. imdb_id has NO 'tt' prefix. rating is int 0-3.
        "name": "TidyTuesday raw_bechdel (~8,800 films) ✓",
        "url": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/raw_bechdel.csv",
        "id_col": "imdb_id",
        "score_col": "rating",
        "numeric": True,
    },
    {
        # ~1,800 films. imdb col HAS 'tt' prefix. score is string e.g. 'ok','notalk'.
        "name": "TidyTuesday movies (~1,800 films) ✓",
        "url": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/movies.csv",
        "id_col": "imdb",
        "score_col": "clean_test",
        "numeric": False,
    },
    {
        # ~1,800 films. Same format as TidyTuesday movies.csv.
        "name": "FiveThirtyEight movies (~1,800 films) ✓",
        "url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv",
        "id_col": "imdb",
        "score_col": "clean_test",
        "numeric": False,
    },
]


def load_bulk_sources():
    lookup = {}
    for src in BULK_SOURCES:
        try:
            print(f"  Trying: {src['name']}")
            r = requests.get(src["url"], timeout=20)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), low_memory=False)

            new = 0
            for _, row in df.iterrows():
                try:
                    imdb = normalize_id(row[src["id_col"]])
                    if src["numeric"]:
                        score = int(row[src["score_col"]])
                    else:
                        raw = str(row[src["score_col"]]).strip().lower()
                        score = CLEAN_TEST_MAP.get(raw)
                        if score is None:
                            continue
                    if imdb not in lookup and 0 <= score <= 3:
                        lookup[imdb] = score
                        new += 1
                except Exception:
                    continue

            print(f"  ✓ {new} new entries added  (lookup total: {len(lookup)})")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return lookup


# ---------------------------------------------------------------------------
# Per-movie API fallback
# ---------------------------------------------------------------------------

def enrich_via_api(df, lookup, delay=0.2):
    unmatched = df[~df['IMDb_ID'].apply(normalize_id).isin(lookup)]
    total = len(unmatched)
    if total == 0:
        print("  All films already matched.")
        return lookup

    print(f"\n  Querying bechdeltest.com API for {total} unmatched films")
    print(f"  Estimated time: ~{total * delay / 60:.0f} min\n")

    session = requests.Session()
    session.headers['User-Agent'] = 'bechdel-enricher/1.0'
    found = 0

    for i, (_, row) in enumerate(unmatched.iterrows(), 1):
        imdb = normalize_id(row['IMDb_ID'])
        numeric = imdb.replace('tt', '')
        score = None

        # Try by IMDb ID
        try:
            r = session.get(
                f"https://bechdeltest.com/api/v1/getMovieByImdbId?imdbid={numeric}",
                timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data and 'rating' in data:
                    score = int(data['rating'])
        except Exception:
            pass

        # Fallback: title + year match
        if score is None and pd.notna(row.get('Title')):
            try:
                r = session.get(
                    f"https://bechdeltest.com/api/v1/getMoviesByTitle"
                    f"?title={requests.utils.quote(str(row['Title']))}",
                    timeout=10)
                if r.status_code == 200:
                    results = r.json()
                    yr = int(row['Year']) if pd.notna(row.get('Year')) else None
                    for m in results:
                        if yr is None or abs(int(m.get('year', 0)) - yr) <= 1:
                            score = int(m['rating'])
                            break
            except Exception:
                pass

        if score is not None:
            lookup[imdb] = score
            found += 1

        time.sleep(delay)

        if i % 250 == 0 or i == total:
            print(f"  [{i}/{total}] matched this pass: {found}")

    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_file  = 'data/movies_enriched_5k.csv'
    output_file = 'data/movies_enriched_5k_bechdel.csv'

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} movies.\n")

    # Step 1: bulk CSVs
    print("Step 1 — Bulk CSV sources...")
    lookup = load_bulk_sources()
    bulk_matched = df['IMDb_ID'].apply(normalize_id).isin(lookup).sum()
    print(f"\n  Total after bulk: {bulk_matched} / {len(df)} matched\n")

    # Step 2: API for the rest
    print("Step 2 — Per-movie API for unmatched films...")
    lookup = enrich_via_api(df, lookup, delay=0.2)

    # Apply & report
    df['bechdel_score'] = df['IMDb_ID'].apply(normalize_id).map(lookup)

    found     = df['bechdel_score'].notna().sum()
    not_found = df['bechdel_score'].isna().sum()
    labels    = {0.0: '0 - No 2 named women', 1.0: '1 - No conversation',
                 2.0: '2 - Talk about men',   3.0: '3 - Pass'}

    print(f"\n{'='*45}")
    print(f"Matched:   {found} / {len(df)}  ({found/len(df)*100:.1f}%)")
    print(f"Not found: {not_found}  ({not_found/len(df)*100:.1f}%)")
    print("\nScore distribution:")
    for k, n in df['bechdel_score'].value_counts().sort_index().items():
        print(f"  {labels.get(k, k)}: {n}  ({n/len(df)*100:.1f}%)")

    print("\nSample (first 15):")
    print(df[['Title', 'Year', 'bechdel_score']].head(15).to_string())

    df.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}")


if __name__ == "__main__":
    main()