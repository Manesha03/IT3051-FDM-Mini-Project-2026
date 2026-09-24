"""Split, encode and scale cleaned bookings without creating derived predictors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from scripts.clean_hotel_bookings import CATEGORY_COLUMNS, PII_COLUMNS, RAW_PATH, REPORT_DIR, ROOT


MODEL_DATA_DIR = ROOT / "data/processed/model_inputs"
TARGET = "is_canceled"
SEED = 42
HIGH_CARDINALITY = ["country", "agent", "company"]
EXCLUSIONS = {
    "reservation_status": "Final booking outcome directly reveals the target.",
    "reservation_status_date": "Date of final status is only known after the outcome.",
    **{c: "Personal/identifier field; not an appropriate predictor." for c in PII_COLUMNS},
    "assigned_room_type": "Actual room assignment may occur after the initial booking.",
    "booking_changes": "Accumulates changes after the initial booking.",
    "days_in_waiting_list": "Completed waiting duration is not known at initial booking.",
    "required_car_parking_spaces": "Request timing is unverified; conservatively exclude mutable requests.",
    "total_of_special_requests": "Request timing is unverified; conservatively exclude mutable requests.",
    "adr": "Recorded average rate is not verified as the original quoted booking rate.",
}


def prepare_predictors(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Use original columns only and document each column's preprocessing treatment."""
    X = cleaned.drop(columns=[TARGET, *EXCLUSIONS]).copy()
    y = cleaned[TARGET].astype("int8").copy()
    if X.empty or X.isna().any().any() or not y.isin([0, 1]).all():
        raise ValueError("Provide non-empty cleaned data with no missing predictors and valid labels.")
    # Remove pandas category metadata learned from the full cleaned file before fitting.
    for c in set(CATEGORY_COLUMNS).intersection(X.columns):
        X[c] = X[c].astype(str)
    records = []
    for c in cleaned.columns:
        if c == TARGET:
            action, reason = "target", "Separate outcome from predictors; preserve each retained label."
        elif c in EXCLUSIONS:
            action, reason = "exclude", EXCLUSIONS[c]
        elif c in HIGH_CARDINALITY:
            action, reason = "one_hot_with_rare_grouping", "Fit rare-category grouping and vocabulary on training rows only."
        elif c in CATEGORY_COLUMNS:
            action, reason = "one_hot", "Fit nominal category vocabulary on training rows only."
        elif c == "is_repeated_guest":
            action, reason = "binary_passthrough", "Keep the existing binary flag unchanged."
        else:
            action, reason = "robust_scale", "Use training median and interquartile range; retain valid extremes."
        records.append({"original_column": c, "action": action, "reason": reason})
    return X, y, pd.DataFrame(records)


def split_positions(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    # Group only by available predictor values, excluding labels and outcome fields.
    # Hashes are split metadata; they never become predictors.
    groups = pd.util.hash_pandas_object(X, index=False)
    if groups.nunique() < 5 or y.nunique() != 2:
        raise ValueError("The grouped split requires at least five profiles and both target classes.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(X, y, groups))
    return train, test, groups


def fit_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame):
    categories = [c for c in X_train if c in CATEGORY_COLUMNS]
    high = [c for c in categories if c in HIGH_CARDINALITY]
    low = [c for c in categories if c not in HIGH_CARDINALITY]
    binary = [c for c in X_train if c == "is_repeated_guest"]
    numeric = [c for c in X_train if c not in categories + binary]
    transformer = ColumnTransformer([
        ("numeric", RobustScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low),
        ("high_cardinality", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=50,
            max_categories=30, sparse_output=True,
        ), high),
        ("binary", "passthrough", binary),
    ], remainder="drop", sparse_threshold=1.0)
    train_matrix = sparse.csr_matrix(transformer.fit_transform(X_train))
    test_matrix = sparse.csr_matrix(transformer.transform(X_test))
    return transformer, train_matrix, test_matrix


def preprocess(cleaned: pd.DataFrame) -> dict:
    X, y, columns = prepare_predictors(cleaned)
    train, test, groups = split_positions(X, y)
    X_train, X_test = X.iloc[train], X.iloc[test]
    transformer, train_matrix, test_matrix = fit_preprocessor(X_train, X_test)
    names = transformer.get_feature_names_out()
    numeric_columns = transformer.transformers_[0][2]
    median = X_train[numeric_columns].median().to_numpy()
    q1 = X_train[numeric_columns].quantile(0.25).to_numpy()
    q3 = X_train[numeric_columns].quantile(0.75).to_numpy()
    expected_scale = q3 - q1
    expected_scale[expected_scale == 0] = 1
    vocabulary_train_only = all(
        set(observed) == set(X_train[c].unique())
        for key in ["categorical", "high_cardinality"]
        for c, observed in zip(
            next(cols for name, _, cols in transformer.transformers_ if name == key),
            transformer.named_transformers_[key].categories_,
        )
    )
    checks = {
        "No excluded columns or target among predictors": not set(X.columns).intersection([TARGET, *EXCLUSIONS]),
        "All inputs are original columns; no derived predictors": set(X.columns).issubset(cleaned.columns),
        "No remaining missing input values": not X.isna().any().any(),
        "Every cleaned row belongs to exactly one partition": (
            len(train) + len(test) == len(cleaned)
            and len(set(train).intersection(test)) == 0
            and set(train).union(test) == set(range(len(cleaned)))
        ),
        "Identical predictor profiles never cross partitions": set(groups.iloc[train]).isdisjoint(groups.iloc[test]),
        "Both target classes occur in each partition": y.iloc[train].nunique() == y.iloc[test].nunique() == 2,
        "Training and test target rows align with source": (
            np.array_equal(y.iloc[train], cleaned.iloc[train][TARGET])
            and np.array_equal(y.iloc[test], cleaned.iloc[test][TARGET])
        ),
        "Matrices have the same encoded columns": train_matrix.shape[1] == test_matrix.shape[1] == len(names),
        "Encoded column names are unique": len(set(names)) == len(names),
        "Matrix rows match target lengths": train_matrix.shape[0] == len(train) and test_matrix.shape[0] == len(test),
        "All transformed values are finite": np.isfinite(train_matrix.data).all() and np.isfinite(test_matrix.data).all(),
        "Scaler center equals training median": np.allclose(transformer.named_transformers_["numeric"].center_, median),
        "Scaler scale equals training IQR with zero-IQR fallback": np.allclose(transformer.named_transformers_["numeric"].scale_, expected_scale),
        "Encoder vocabularies contain training categories only": vocabulary_train_only,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("Preprocessing validation failed: " + "; ".join(failures))
    assignment = pd.concat([
        pd.DataFrame({
            "split": label, "matrix_row": np.arange(len(positions)),
            "cleaned_row_position": positions, "source_row_index": cleaned.index[positions],
            "profile_group": groups.iloc[positions].to_numpy(),
        }) for label, positions in [("train", train), ("test", test)]
    ], ignore_index=True)
    split_summary = pd.DataFrame([
        {"split": label, "rows": len(positions), "percent_of_cleaned_rows": len(positions) / len(cleaned) * 100,
         "target_0": int(y.iloc[positions].eq(0).sum()), "target_1": int(y.iloc[positions].eq(1).sum()),
         "cancellation_rate_pct": y.iloc[positions].mean() * 100,
         "profile_groups": groups.iloc[positions].nunique(), "encoded_columns": len(names)}
        for label, positions in [("train", train), ("test", test)]
    ])
    return {
        "transformer": transformer, "X_train": train_matrix, "X_test": test_matrix,
        "y_train": y.iloc[train].reset_index(drop=True), "y_test": y.iloc[test].reset_index(drop=True),
        "feature_names": names, "input_columns": list(X.columns),
        "column_decisions": columns, "split_assignment": assignment, "split_summary": split_summary,
        "validation": pd.DataFrame({"check": checks.keys(), "passed": checks.values()}),
        "verification_input": X_test.head(16),
    }


def export_preprocessed(result: dict, output_dir: Path = MODEL_DATA_DIR, report_dir: Path = REPORT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for label in ["train", "test"]:
        matrix_path = output_dir / f"X_{label}.npz"
        target_path = output_dir / f"y_{label}.csv"
        sparse.save_npz(matrix_path, result[f"X_{label}"])
        result[f"y_{label}"].to_frame(TARGET).to_csv(target_path, index=False)
        restored = sparse.load_npz(matrix_path)
        if restored.shape != result[f"X_{label}"].shape or (restored != result[f"X_{label}"]).nnz:
            raise ValueError(f"{label} matrix export changed values.")
        if not np.array_equal(pd.read_csv(target_path)[TARGET], result[f"y_{label}"]):
            raise ValueError(f"{label} target export changed labels.")
    joblib.dump(result["transformer"], output_dir / "preprocessor.joblib")
    restored_transformer = joblib.load(output_dir / "preprocessor.joblib")
    restored_sample = sparse.csr_matrix(restored_transformer.transform(result["verification_input"]))
    expected_sample = result["X_test"][:len(result["verification_input"])]
    if restored_sample.shape != expected_sample.shape or (restored_sample != expected_sample).nnz:
        raise ValueError("Saved preprocessor does not reproduce the test transformations.")
    pd.DataFrame({"matrix_column": np.arange(len(result["feature_names"])),
                  "encoded_column": result["feature_names"]}).to_csv(output_dir / "encoded_columns.csv", index=False)
    result["split_assignment"].to_csv(output_dir / "split_assignment.csv", index=False)
    result["column_decisions"].to_csv(report_dir / "preprocessing_columns.csv", index=False)
    result["split_summary"].to_csv(report_dir / "train_test_summary.csv", index=False)
    validation = pd.concat([result["validation"], pd.DataFrame([
        {"check": "Saved training and test matrices preserve values", "passed": True},
        {"check": "Saved training and test targets preserve labels", "passed": True},
        {"check": "Reloaded preprocessor reproduces test sample transformations", "passed": True},
    ])], ignore_index=True)
    validation.to_csv(report_dir / "preprocessing_validation.csv", index=False)
    metadata = {
        "target": TARGET, "input_columns": result["input_columns"], "derived_predictors": [],
        "split_method": "First fold of StratifiedGroupKFold(n_splits=5, shuffle=True)", "random_state": SEED,
        "group_definition": "Hash of original retained predictor values, without target or outcome fields",
        "prediction_point": "Initial booking; retained fields assumed available then; source snapshots do not verify this",
        "numeric_transform": "RobustScaler fitted on training rows only",
        "categorical_transform": "One-hot; country/agent/company training rare levels grouped at min_frequency=50, max_categories=30",
        "raw_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "scipy": scipy.__version__, "joblib": joblib.__version__},
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
