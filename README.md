# 🎬 Predicting the Bechdel Test from Movie Metadata

**Machine Learning Foundations — Group Project**
IE University · BDBA Class of 2028

**Team:** Raya Metchkarova · Alexander Glapiak · Alp Kurtbolat · Aysel Zeynalova · Milan Josifovikj · Samuel Sacha Benayoun

---

## Table of Contents

1. [Overview](#overview)
2. [Research Questions](#research-questions)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Feature Engineering](#feature-engineering)
6. [Models](#models)
7. [Experiments](#experiments)
8. [Results](#results)
9. [Installation & Usage](#installation--usage)

---

## Overview

This project applies supervised machine learning to predict whether a film **passes the Bechdel Test** — a widely used benchmark for female representation in cinema. A film passes if it (1) has at least two named women, (2) who talk to each other, (3) about something other than a man.

Using pre-release movie metadata (genre, year, runtime, budget, IMDb rating), we train and evaluate a suite of classifiers — from a majority-class dummy to gradient-boosted trees — and run a systematic set of ablation and confound-control experiments to determine *what the model is actually learning*.

---

## Research Questions

1. Is the model learning **meaningful, representation-related patterns** from pre-release movie metadata?
2. Or is it relying on **shortcut / confounding features** — genre stereotypes, popularity signals, temporal trends, or enrichment artifacts?

---

## Dataset

**Primary source:** FiveThirtyEight Bechdel data joined to IMDb/TMDb metadata via the Kaggle dataset [*9000+ Movies: IMDb and Bechdel*](https://www.kaggle.com/).

| Column | Description |
|--------|-------------|
| `imdb_id` | IMDb title identifier |
| `title` | Film title |
| `year` | Release year |
| `runtime` | Runtime in minutes |
| `budget` | Production budget (TMDb; ~50–78% missing) |
| `revenue` | Box office revenue (TMDb; ~50–75% missing) |
| `imdb_rating` | IMDb average user rating |
| `num_votes` | IMDb vote count |
| `genres` | Pipe-separated genre list |
| `bechdel_score` | Ordinal Bechdel score (0–3) |
| `bechdel_pass` | **Binary target** — 1 if score = 3, else 0 |

> **Note on missingness:** Budget, revenue, and TMDb popularity are missing for a large fraction of films. This is by design — TMDb only partially overlaps with the Bechdel/IMDb corpus. Missing values are *not* missing at random; they correlate with production scale. The project explicitly tests whether data absence itself is a predictive signal.

A **synthetic data generator** (`data_loader.py`) is included so the full pipeline can be developed and tested without access to the Kaggle CSV. Metrics from synthetic data are clearly watermarked and are not scientific results.

---

## Project Structure

```
ML_group_project/
│
├── src/
│   ├── data_loader.py       # Real CSV loader + synthetic data generator
│   ├── preprocessing.py     # Feature engineering & sklearn ColumnTransformer
│   ├── models.py            # Pipeline definitions for all 5 classifiers
│   └── evaluation.py        # CV, test metrics, confusion matrix, ROC & learning curves
│
├── notebooks/
│   ├── bechdel_1_to_44.ipynb            # Main end-to-end notebook (EDA → tuning → eval)
│   ├── bechdel_experiments.ipynb        # Extended experiment suite
│   └── bechdel_experiments_fixed.ipynb  # Comprehensive experiment suite (final version)
│
├── data/
│   ├── movies_complete.csv              # Merged Bechdel + IMDb + TMDb dataset (not committed)
│   ├── Bechdel_IMDB_Merge0524.csv       # Raw Bechdel/IMDb merge
│   └── movies_enriched_5k.csv          # TMDb enrichment subset
│
└── README.md
```

---

## Feature Engineering

All feature engineering is **row-local and deterministic** — transformations depend only on a row's own values, so they can safely be applied before the train/test split with zero leakage.

| Feature | Derivation |
|---------|-----------|
| `log_budget` | `log1p(budget)` — compresses heavy-tailed financial data |
| `log_revenue` | `log1p(revenue)` |
| `log_num_votes` | `log1p(num_votes)` |
| `log_runtime` | `log1p(runtime)` |
| `decade` | Year bucketed to decade (e.g. `1990s`) — one-hot encoded |
| `genre_*` | Binary indicator for each of 15 top genres |
| `num_genres` | Number of genres assigned to a film |

Stat-dependent transformations (imputation, scaling, one-hot encoding) are embedded inside each model's `Pipeline` and fitted exclusively on training data within each CV fold.

---

## Models

Five classifiers are evaluated, from trivial to complex:

| Key | Model | Notes |
|-----|-------|-------|
| `dummy` | `DummyClassifier` (most frequent) | Baseline reference |
| `logreg` | Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| `tree` | Decision Tree | `max_depth=6`, `class_weight="balanced"` |
| `rf` | Random Forest | 300 trees, `min_samples_leaf=2`, balanced weights |
| `xgb` | XGBoost | 400 estimators, `lr=0.05`, `max_depth=5` |

Every model is returned as a `sklearn.pipeline.Pipeline` with preprocessing included, ensuring identical, leakage-safe treatment across training and evaluation.

---

## Experiments

The experiment suite in `bechdel_experiments_fixed.ipynb` runs a structured set of comparisons, all using a fixed train/test split with model selection based solely on CV ROC-AUC:

### 1. Baseline — All Models
All five classifiers under identical conditions (median imputation, no TF-IDF, no extra signals). Establishes the best-performing model for downstream experiments.

### 2. Imputation Strategy
Tests three missing-value strategies — `median`, `mean`, `most_frequent` — given that TMDb-derived features are 50–78% missing. Median is robust to outliers in budget/revenue and serves as the default.

### 3. Missingness as Signal
Adds five binary indicator features (`budget_missing`, `revenue_missing`, `popularity_missing`, `cast_size_missing`, `plot_missing`). Tests whether the *absence* of enrichment data predicts Bechdel outcomes — an enrichment artifact check.

### 4. Ablation Experiments
Systematically removes feature groups to measure each group's contribution to ROC-AUC:

| Ablation | Features removed |
|----------|-----------------|
| No genre | All 15 genre binary flags |
| No popularity | `log_num_votes`, `tmdb_popularity` |
| No temporal | `year`, `decade` |
| No TMDb enrichment | `log_budget`, `log_revenue`, `tmdb_popularity`, `log_cast_size` |
| No runtime | `log_runtime` |
| No female director | `female_director` flag |

### 5. Visualisations & Interpretation
ROC curves, confusion matrices, precision-recall curves, permutation importances, partial dependence plots, and learning curves — each paired with a written interpretation of what it shows and which research question it addresses.

---

## Results

Performance is reported as **ROC-AUC** (primary) and F1 (secondary). Model selection is based exclusively on cross-validated ROC-AUC on the training split — the test set is evaluated exactly once per experiment.

> Full numeric results are produced when you run the notebooks against the real dataset. See [Reproducing the Results](#reproducing-the-results) below.

Key findings:
- **Genre** is the strongest individual feature group; removing it produces the largest AUC drop.
- **Temporal features** (year, decade) contribute meaningfully, reflecting changing industry norms over time.
- **Missingness indicators** provide a small but measurable signal, confirming that data absence is not missing at random.
- The best model substantially outperforms the dummy baseline, indicating that the metadata contains genuine signal — but the ablation results suggest genre stereotypes account for a significant fraction of it.

---

## Installation & Usage

### Requirements

- Python ≥ 3.9
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`

```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn jupyter
```

### Running the pipeline

```bash
# Clone the repository
git clone https://github.com/rmetchkarovaieu2024-bit/ML_group_project.git
cd ML_group_project
```

Place the Kaggle CSV at `data/movies_complete.csv` (or use the synthetic fallback — the loader detects automatically).

```bash
# Run the main notebook
jupyter notebook notebooks/bechdel_1_to_44.ipynb

# Run the full experiment suite
jupyter notebook notebooks/bechdel_experiments_fixed.ipynb
```

---

## Key Design Decisions

**Leakage prevention** — All stat-dependent transforms (imputation, scaling, OHE) live inside each model's `Pipeline` and are fit only on training folds during CV.

**Class imbalance** — All classifiers use `class_weight="balanced"` where supported; XGBoost uses `scale_pos_weight`. Evaluation prioritises F1 and ROC-AUC over accuracy.

**Shortcut auditing** — The ablation and missingness experiments are explicitly designed to test whether the model exploits confounders rather than representation-related signal.
