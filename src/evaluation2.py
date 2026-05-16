"""
evaluation.py
=============
Cross-validation, test evaluation, and plotting utilities.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

# Suppress non-critical numerical warnings from sklearn optimizers
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._linear_loss")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


# ---------------------------------------------------------------------------
# CV scoring
# ---------------------------------------------------------------------------

def cv_score_model2(pipeline, X, y, cv_splits=5, scoring="roc_auc",
                   random_state=42) -> dict:
    """Stratified K-fold CV. Returns mean, std, and raw fold scores."""
    cv2 = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=random_state)
    scores2 = cross_val_score(pipeline, X, y, cv=cv2, scoring=scoring, n_jobs=-1)
    return {"mean": float(scores2.mean()), "std": float(scores2.std()),
            "scores": scores2.tolist()}


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test2(pipeline, X_test, y_test) -> dict:
    """Return a dict of metrics for a fitted pipeline on the test set."""
    y_pred2 = pipeline.predict(X_test)
    y_prob2 = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred2)),
        "f1":        float(f1_score(y_test, y_pred2, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob2)),
        "y_pred":    y_pred2,
        "y_prob":    y_prob2,
    }


def results_table2(test_results: dict) -> pd.DataFrame:
    rows2 = []
    for name2, m2 in test_results.items():
        rows2.append({
            "model":    name2,
            "accuracy": round(m2["accuracy"], 4),
            "f1":       round(m2["f1"], 4),
            "roc_auc":  round(m2["roc_auc"], 4),
        })
    return pd.DataFrame(rows2).set_index("model").sort_values("roc_auc",
                                                               ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc2(pipelines: dict, X_test, y_test, title="ROC curves") -> plt.Figure:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for name2, pipe2 in pipelines.items():
        y_prob2 = pipe2.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(
            y_test, y_prob2, name=name2, ax=ax2, alpha=0.8,
        )
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax2.set_title(title)
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig2


def plot_confusion2(y_true, y_pred, title="Confusion matrix") -> plt.Figure:
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Fail", "Pass"],
        colorbar=False, ax=ax2 ,
    )
    ax2.set_title(title)
    plt.tight_layout()
    return fig2


def plot_learning_curve2(pipeline, X_train, y_train,
                        title="Learning curve") -> plt.Figure:
    train_sizes2, train_scores2, val_scores2 = learning_curve(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.fill_between(train_sizes2,
                     train_scores2.mean(1) - train_scores2.std(1),
                     train_scores2.mean(1) + train_scores2.std(1),
                     alpha=0.15, color="#2ca02c")
    ax2.fill_between(train_sizes2,
                    val_scores2.mean(1) - val_scores2.std(1),
                    val_scores2.mean(1) + val_scores2.std(1),
                    alpha=0.15, color="#1f77b4")
    ax2.plot(train_sizes2, train_scores2.mean(1), "o-", color="#2ca02c", label="Train")
    ax2.plot(train_sizes2, val_scores2.mean(1),   "o-", color="#1f77b4", label="CV val")
    ax2.set_xlabel("Training set size")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title(title)
    ax2.legend()
    plt.tight_layout()
    return fig2
