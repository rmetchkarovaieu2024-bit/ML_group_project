"""
Model definitions for the Bechdel classification task.

Each model is returned as a Pipeline with preprocessing included,
ensuring consistent training and evaluation.
"""

from __future__ import annotations
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import build_preprocessor


def make_model(name: str, random_state: int = 42) -> Pipeline:    # Return a Pipeline for the specified model

    if name == "dummy":
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)

    elif name == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )

    elif name == "tree":
        clf = DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )

    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )

    elif name == "xgb":
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")

    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", clf),
    ])


PARAM_GRIDS = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    },
    "rf": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "xgb": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 0.9, 1.0],
    },
}


MODEL_NAMES = ["dummy", "logreg", "tree", "rf", "xgb"]