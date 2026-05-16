"""
models.py
=========
Model definitions and hyperparameter search grids.

All PARAM_GRIDS keys use the 'clf__' prefix so they work directly with
a Pipeline(steps=[('preprocess', ...), ('clf', ...)]) without modification.
"""

from __future__ import annotations
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB2 = True
except ImportError:
    HAS_XGB2 = False

# ---------------------------------------------------------------------------
# Classifier catalogue
# ---------------------------------------------------------------------------

def get_classifiers2(random_state: int = 42) -> dict:
    clfs2 = {
        "dummy":  DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=2000, random_state=random_state, solver="liblinear",
            class_weight="balanced", C=1.0,
        ),
        "tree":   DecisionTreeClassifier(
            max_depth=6, random_state=random_state
        ),
        "rf":     RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
    }
    if HAS_XGB2:
        clfs2["xgb"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
    return clfs2


MODEL_NAMES2 = list(get_classifiers2().keys())

# ---------------------------------------------------------------------------
# Hyperparameter search grids (clf__ prefix for Pipeline compatibility)
# ---------------------------------------------------------------------------

PARAM_GRIDS2 = {
    "logreg": {
        "clf__C":       [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver":  ["liblinear"],
    },
    "xgb": {
        "clf__n_estimators":  [100, 200, 300, 500],
        "clf__max_depth":     [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":     [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    },
    "rf": {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
    },
}
