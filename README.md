# Predicting Bechdel Test Outcomes Using Machine Learning

> *Structure predicts representation. The question now is — what does the industry do about it?*

**IE University · BDBA 2028 · Machine Learning Foundations · Group 8**

Raya Metchkarova · Alexander Glapiak · Alp Kurtbolat · Aysel Zeynalova · Milan Josifovikj · Samuel Sacha Benayoun

---

## What This Project Is

Can a machine predict whether a film gives women a real voice on screen — from metadata alone, without watching a single frame?

We built a supervised ML pipeline to answer that question. Using pre-release movie metadata — genre, release year, runtime, IMDb rating, budget, revenue — we trained five classifiers to predict whether a film passes the **Bechdel Test**: the widely used three-criteria benchmark for female representation in cinema.

Our best model, a tuned XGBoost classifier, reaches **ROC-AUC 0.700** — twenty points above a coin-flip baseline. Genre is the strongest individual signal, but removing it entirely barely hurts the model. Decade, popularity, and financial scale carry more weight than expected. The ceiling isn't the models. It's the data.

---

## The Bechdel Test

A film passes if it meets three criteria:

1. It has **at least two named female characters**
2. Those characters **talk to each other**
3. About **something other than a man**

Only **57% of films** in our dataset pass all three. More than four in ten films — including many critically acclaimed, commercially successful ones — cannot clear that bar. That number is the motivation for this project.

---

## Results at a Glance

| Model | CV ROC-AUC | Test AUC | Test F1 |
|---|---|---|---|
| Dummy (majority) | 0.500 ± 0.000 | 0.500 | 0.732 |
| Decision Tree (d=6) | 0.618 ± 0.014 | 0.633 | 0.721 |
| Logistic Regression | 0.659 ± 0.012 | 0.665 | 0.719 |
| Random Forest (n=300) | 0.668 ± 0.011 | 0.655 | 0.705 |
| **XGBoost (tuned) ★** | **0.689 ± 0.009** | **0.700** | **0.714** |

Key findings from the experiment suite:

- **Genre** produces the largest AUC drop when removed — but most predictive power survives without it. The signal is distributed, not concentrated.
- **Missingness indicators** (`budget_missing`, `revenue_missing`) have a measurable but small effect. The model is not exploiting data gaps as a shortcut.
- **Multiple interacting signals** — decade, `log_num_votes`, financial scale — drive performance rather than one dominant feature.
- The bottleneck is **data sparsity**, not model complexity. 78% of budget/revenue values are median-imputed; TMDb covers only 22% of the corpus.

---

## Project Structure

```
ML_group_project/
│
├── src/
│   ├── data_loader.py       # Real CSV loader + synthetic data generator
│   ├── preprocessing.py     # Feature engineering & sklearn ColumnTransformer
│   ├── models.py            # Pipeline definitions for all 5 classifiers + PARAM_GRIDS
│   └── evaluation.py        # CV, test metrics, confusion matrix, ROC & learning curves
│
├── notebooks/
│   ├── bechdel_1_to_44.ipynb            # Main end-to-end notebook
│   ├── bechdel_experiments.ipynb        # Extended experiment suite
│   └── bechdel_experiments_fixed.ipynb  # Final experiment suite with ablations
│
├── data/
│   ├── movies_complete.csv          # Merged Bechdel + IMDb + TMDb (not committed)
│   ├── Bechdel_IMDB_Merge0524.csv   # Raw Bechdel/IMDb merge
│   └── movies_enriched_5k.csv       # TMDb enrichment subset
│
└── README.md
```

---

## Dataset

Two Kaggle datasets merged on IMDb title ID:

**Bechdel + IMDb merge** — 9,000 films with Bechdel scores (0–3) and IMDb metadata (year, runtime, rating, vote count). Bechdel labels from bechdeltest.com — community-sourced and reviewed.

**TMDb enrichment** — 5,000 films with budget, revenue, popularity, cast size, and plot summaries.

**Merge result** — ~5,000 usable rows. TMDb features present for ~22% of films.

**Target variable** — Bechdel score = 3 → pass (1); score < 3 → fail (0). Class split: 57% pass / 43% fail.

> **Note on missingness:** Budget, revenue, and TMDb popularity are missing for a large fraction of films. This is not random — it correlates with production scale. The project explicitly tests whether data absence is itself a predictive signal (it isn't, meaningfully).

---

## Features

All feature engineering lives in `preprocessing.py` and is row-local — no dataset statistics are needed, so it runs safely before the train/test split.

| Feature | Type | Derivation |
|---|---|---|
| `year` | Numeric | Raw |
| `runtime` | Numeric | Raw |
| `log_budget` | Numeric | `log1p(budget)` |
| `log_revenue` | Numeric | `log1p(revenue)` |
| `imdb_rating` | Numeric | Raw |
| `log_num_votes` | Numeric | `log1p(num_votes)` |
| `num_genres` | Numeric | Count of genres assigned |
| `decade` | Categorical | `(year // 10 * 10)` → one-hot encoded |
| `genre_drama` … `genre_biography` | Binary | 15 flags from `TOP_GENRES` |

Stat-dependent steps — `SimpleImputer`, `StandardScaler`, `OneHotEncoder` — are embedded inside each model's `Pipeline` and fitted only on training data within each CV fold. Leakage is architecturally impossible.

---

## Models

All five models are built by `make_model(name)` in `models.py`, which returns a `Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])`.

| Key | Model | Notes |
|---|---|---|
| `dummy` | DummyClassifier | `strategy="most_frequent"` — sets the floor |
| `logreg` | Logistic Regression | `class_weight="balanced"`, GridSearch over C ∈ [0.01, 0.1, 1.0, 10.0] |
| `tree` | Decision Tree | `max_depth=6`, `class_weight="balanced"` |
| `rf` | Random Forest | 300 trees, `min_samples_leaf=2`, GridSearch over depth and leaf size |
| `xgb` | XGBoost | 400 estimators, `lr=0.05`, `max_depth=5`, RandomizedSearchCV 50 iterations |

---

## Experiments

`bechdel_experiments_fixed.ipynb` runs a structured suite of comparisons, all using a fixed train/test split with model selection based on CV ROC-AUC:

**Baseline** — all five models under identical conditions. Establishes the best-performing model for downstream experiments.

**Imputation strategy** — median vs mean vs most_frequent. Median is robust to outliers in budget/revenue and is used as the default.

**Missingness as signal** — adds binary indicators (`budget_missing`, `revenue_missing`, `popularity_missing`) to test whether the absence of TMDb data is itself predictive. Small effect, not a meaningful one.

**Ablation experiments** — systematically removes feature groups to measure contribution:

| Ablation | Features removed |
|---|---|
| No genre | All 15 genre binary flags |
| No popularity | `log_num_votes`, `tmdb_popularity` |
| No temporal | `year`, `decade` |
| No TMDb enrichment | `log_budget`, `log_revenue`, `tmdb_popularity` |
| No runtime | `log_runtime` |

**Visualisations** — ROC curves, confusion matrices, precision-recall curves, permutation importances, and learning curves via `plot_roc()`, `plot_confusion()`, and `plot_learning_curve()` in `evaluation.py`.

---

## Installation

```bash
git clone https://github.com/rmetchkarovaieu2024-bit/ML_group_project.git
cd ML_group_project
pip install scikit-learn xgboost pandas numpy matplotlib seaborn jupyter
```

Place the Kaggle CSV at `data/movies_complete.csv`. If unavailable, `load_synthetic_data()` in `data_loader.py` generates a realistic fallback dataset automatically — but metrics from synthetic data are watermarked and are not scientific results.

```bash
# Run the full experiment suite
jupyter notebook notebooks/bechdel_experiments_fixed.ipynb
```

Approximate runtime on a modern laptop: ~12 minutes including hyperparameter search.

All random states are fixed at `seed=42` throughout for reproducibility.

---

## Ethical Considerations

This model is **descriptive, not prescriptive**.

It was built to study patterns in historical data — not to evaluate films or filmmakers, and not to inform financing or distribution decisions. The Bechdel Test is a blunt instrument: passing it doesn't make a film feminist, and failing it doesn't mean it lacks female depth. IMDb ratings encode historical inequity in who reviews films. A model trained on that data reproduces those patterns.

Our results are a lens for critical discussion. Not a scoring system. Not a gatekeeping tool.

---

## Reproduction Checklist

1. Download [*9000+ Movies: IMDb and Bechdel*](https://www.kaggle.com/) from Kaggle → save as `data/movies_complete.csv`
2. Run `bechdel_experiments_fixed.ipynb` top to bottom
3. All random states fixed (`seed=42`) — results are fully reproducible
4. Synthetic fallback activates automatically if the CSV is absent — outputs are watermarked
