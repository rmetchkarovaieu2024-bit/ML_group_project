"""
Utilities for model evaluation: cross-validation, test metrics, and plots.

Plotting functions return matplotlib Figure objects so the caller
can decide whether to display or save them.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve


def cv_score_model(pipeline, X, y, cv_splits: int = 5,
                   scoring: str = "roc_auc", random_state: int = 42) -> dict:
    """Run stratified K-fold cross-validation and return mean, std, and scores."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
    }


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """Compute standard evaluation metrics on the test set."""
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Convert results into a sorted DataFrame."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
        })
    return (
        pd.DataFrame(rows)
        .set_index("Model")
        .round(4)
        .sort_values("ROC-AUC", ascending=False)
    )


def plot_confusion(y_true, y_pred, title: str = "Confusion Matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(pipelines: dict, X_test, y_test,
             title: str = "ROC Curves") -> plt.Figure:
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in pipelines.items():
        if hasattr(pipe, "predict_proba"):
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_learning_curve(pipeline, X, y, title: str = "Learning Curve",
                        cv_splits: int = 5) -> plt.Figure:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42) # Plot training and validation performance vs dataset size
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
        random_state=42,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Train AUC")
    ax.fill_between(
        sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15
    )
    ax.plot(sizes, val_scores.mean(axis=1), "o-", label="CV AUC")
    ax.fill_between(
        sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig