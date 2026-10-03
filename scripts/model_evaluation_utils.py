"""Shared model-evaluation utilities for modelling.

All group members should import this module so models use the same data split,
preprocessing, random state, cross-validation strategy and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DATA_PATH = ROOT / "data" / "processed" / "hotel_booking_features.csv"
OUTPUT_DIR = ROOT / "06_Model_Evaluation"
TARGET = "is_canceled"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
HIGH_CARDINALITY_COLUMNS = ["country", "agent", "company"]

COMMON_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_feature_dataset(path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    """Load the feature-engineered dataset with ID-like fields as text."""
    dtype = {"agent": "string", "company": "string", "country": "string"}
    data = pd.read_csv(path, dtype=dtype)
    if TARGET not in data.columns:
        raise ValueError(f"Target column {TARGET!r} is missing from {path}.")
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Feature dataset contains missing values in: {missing}")
    return data


def split_feature_dataset(data: pd.DataFrame) -> PreparedData:
    """Create the shared stratified train/test split."""
    X = data.drop(columns=TARGET)
    y = data[TARGET].astype("int8")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def get_cv() -> StratifiedKFold:
    """Return the shared cross-validation strategy."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def class_distribution(y: pd.Series) -> pd.DataFrame:
    """Summarise binary target counts and percentages."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({
        "class": counts.index.astype(int),
        "count": counts.to_numpy(),
        "percent": (counts.to_numpy() / len(y) * 100).round(4),
    })


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the shared preprocessing transformer.

    Numeric fields are robust-scaled, nominal fields are one-hot encoded, and
    high-cardinality ID-like fields use infrequent-category grouping.
    """
    categorical = [
        column
        for column in X.columns
        if (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_string_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        )
    ]
    high_cardinality = [c for c in categorical if c in HIGH_CARDINALITY_COLUMNS]
    low_cardinality = [c for c in categorical if c not in high_cardinality]
    binary = [
        c
        for c in X.columns
        if c not in categorical and X[c].nunique(dropna=False) == 2
    ]
    numeric = [
        c
        for c in X.columns
        if c not in set(categorical + binary)
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", RobustScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_cardinality),
            (
                "high_cardinality",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    max_categories=30,
                    sparse_output=True,
                ),
                high_cardinality,
            ),
            ("binary", "passthrough", binary),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    """Create a model pipeline with the shared preprocessor."""
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", model),
    ])


def evaluate_predictions(
    model_name: str,
    version: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    fit_seconds: float | None = None,
) -> dict:
    """Calculate the common final-evaluation metrics."""
    row = {
        "model": model_name,
        "version": version,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "fit_seconds": fit_seconds,
    }
    if y_score is not None:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["pr_auc"] = average_precision_score(y_true, y_score)
    return row


def positive_class_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class probabilities or decision scores when available."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    return None


def cross_validation_summary(model_name: str, version: str, pipeline: Pipeline, X, y) -> pd.DataFrame:
    """Run shared CV scoring and return mean/std metrics."""
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        cv=get_cv(),
        scoring=COMMON_SCORING,
        n_jobs=-1,
        return_train_score=True,
    )
    rows = []
    for metric in COMMON_SCORING:
        train_key = f"train_{metric}"
        test_key = f"test_{metric}"
        rows.append({
            "model": model_name,
            "version": version,
            "metric": metric,
            "train_mean": cv_result[train_key].mean(),
            "train_std": cv_result[train_key].std(),
            "cv_mean": cv_result[test_key].mean(),
            "cv_std": cv_result[test_key].std(),
        })
    return pd.DataFrame(rows)


def fit_and_evaluate(model_name: str, version: str, pipeline: Pipeline, data: PreparedData) -> tuple[Pipeline, dict]:
    """Fit on training data and evaluate once on the final test set."""
    start = perf_counter()
    pipeline.fit(data.X_train, data.y_train)
    fit_seconds = perf_counter() - start
    y_pred = pipeline.predict(data.X_test)
    y_score = positive_class_scores(pipeline, data.X_test)
    return pipeline, evaluate_predictions(
        model_name=model_name,
        version=version,
        y_true=data.y_test,
        y_pred=y_pred,
        y_score=y_score,
        fit_seconds=fit_seconds,
    )


def confusion_matrix_frame(y_true: pd.Series, y_pred: Iterable[int]) -> pd.DataFrame:
    """Return a labelled confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )


