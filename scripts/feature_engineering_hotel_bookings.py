"""Feature engineering, feature selection and encoding-strategy documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.clean_hotel_bookings import (
    CATEGORY_COLUMNS,
    PII_COLUMNS,
    ROOT,
    load_cleaned,
)


OUTPUT_DIR = ROOT / "04_Feature_Engineering"
FEATURED_PATH = ROOT / "data/processed/hotel_booking_features.csv"
TARGET = "is_canceled"
ENGINEERED_COLUMNS = [
    "total_stay",
    "total_guests",
    "has_children",
    "has_babies",
    "previous_total_bookings",
    "previous_cancellation_rate",
]
LEAKAGE_COLUMNS = {
    "reservation_status": "Directly encodes the final booking outcome.",
    "reservation_status_date": "Recorded after the final booking outcome/status is known.",
}
IDENTIFIER_COLUMNS = {c: "Personal identifier; not useful or appropriate for modelling." for c in PII_COLUMNS}
PREDICTION_TIME_EXCLUSIONS = {
    "assigned_room_type": "Actual assigned room may be known after reservation creation.",
    "booking_changes": "Counts changes that can happen after the initial reservation.",
    "days_in_waiting_list": "Completed waiting duration is not known at initial booking.",
}
HIGH_CARDINALITY_COLUMNS = ["country", "agent", "company"]
INITIAL_RESERVATION_ASSUMPTIONS = {
    "required_car_parking_spaces": (
        "Retained under the project assumption that parking requirements are recorded "
        "when the initial reservation is created."
    ),
    "total_of_special_requests": (
        "Retained under the project assumption that special requests are recorded when "
        "the initial reservation is created."
    ),
}
COMPONENT_COLUMNS_RETAINED = {
    "stays_in_weekend_nights": "Retained with total_stay because weekend and weekday stay patterns can differ.",
    "stays_in_week_nights": "Retained with total_stay because weekend and weekday stay patterns can differ.",
    "adults": "Retained with total_guests because party composition can matter beyond party size.",
    "children": "Retained with total_guests and has_children because exact composition can matter.",
    "babies": "Retained with total_guests and has_babies because exact composition can matter.",
    "previous_cancellations": "Retained with engineered history summaries to preserve raw history counts.",
    "previous_bookings_not_canceled": "Retained with engineered history summaries to preserve raw history counts.",
}


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create project-approved derived features from cleaned source columns."""
    data = cleaned.copy(deep=True)
    data["total_stay"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]
    data["has_children"] = data["children"].gt(0).astype("int8")
    data["has_babies"] = data["babies"].gt(0).astype("int8")
    data["previous_total_bookings"] = (
        data["previous_cancellations"] + data["previous_bookings_not_canceled"]
    )
    data["previous_cancellation_rate"] = np.where(
        data["previous_total_bookings"].gt(0),
        data["previous_cancellations"] / data["previous_total_bookings"],
        0.0,
    )
    return data


def feature_type_report(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        dtype = data[column].dtype
        if column == TARGET:
            role = "target"
        elif column in PII_COLUMNS:
            role = "identifier"
        elif isinstance(dtype, pd.CategoricalDtype):
            role = "categorical"
        elif pd.api.types.is_numeric_dtype(dtype):
            role = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            role = "date"
        else:
            role = "text"
        rows.append({
            "column": column,
            "role": role,
            "dtype": str(dtype),
            "missing": int(data[column].isna().sum()),
            "unique_values": int(data[column].nunique(dropna=False)),
        })
    return pd.DataFrame(rows)


def evaluate_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ENGINEERED_COLUMNS:
        series = data[column]
        cancelled = data.loc[data[TARGET].eq(1), column]
        not_cancelled = data.loc[data[TARGET].eq(0), column]
        rows.append({
            "feature": column,
            "dtype": str(series.dtype),
            "unique_values": int(series.nunique(dropna=False)),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean_not_canceled": float(not_cancelled.mean()),
            "mean_canceled": float(cancelled.mean()),
            "mean_difference_canceled_minus_not": float(cancelled.mean() - not_cancelled.mean()),
            "absolute_correlation_with_target": float(abs(series.corr(data[TARGET]))),
            "decision": "retain_candidate",
            "decision_note": (
                "Retained as a candidate predictor because it has business meaning; "
                "weak individual linear correlation does not rule out non-linear or "
                "interaction value. Final usefulness should be tested during modelling."
            ),
        })
    return pd.DataFrame(rows)


def build_feature_decisions(data: pd.DataFrame) -> pd.DataFrame:
    exclusions = {
        **LEAKAGE_COLUMNS,
        **IDENTIFIER_COLUMNS,
        **PREDICTION_TIME_EXCLUSIONS,
    }
    rows = []
    for column in data.columns:
        if column == TARGET:
            action = "target"
            reason = "Prediction outcome; kept separately for later ML preparation."
        elif column in ENGINEERED_COLUMNS:
            action = "retain_engineered"
            reason = "Derived from cleaned booking fields and evaluated for target relationship."
        elif column in exclusions:
            action = "exclude"
            reason = exclusions[column]
        elif column in COMPONENT_COLUMNS_RETAINED:
            action = "retain_original_component"
            reason = COMPONENT_COLUMNS_RETAINED[column]
        elif column in INITIAL_RESERVATION_ASSUMPTIONS:
            action = "retain_original"
            reason = INITIAL_RESERVATION_ASSUMPTIONS[column]
        else:
            action = "retain_original"
            reason = "Cleaned source feature retained for later modelling."
        rows.append({"column": column, "action": action, "reason": reason})
    return pd.DataFrame(rows)


def selected_feature_columns(decisions: pd.DataFrame) -> list[str]:
    keep_actions = {"retain_original", "retain_original_component", "retain_engineered"}
    return decisions.loc[decisions["action"].isin(keep_actions), "column"].tolist()


def build_encoding_strategy(data: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    rows = []
    for column in selected:
        if column in HIGH_CARDINALITY_COLUMNS:
            method = "one_hot_with_rare_grouping"
            reason = "High-cardinality nominal field; group rare levels using training data only in ML preparation."
        elif column in CATEGORY_COLUMNS or isinstance(data[column].dtype, pd.CategoricalDtype):
            method = "one_hot"
            reason = "Nominal categorical field; one-hot encoding avoids artificial ordering."
        elif column == "is_repeated_guest" or pd.api.types.is_integer_dtype(data[column]) and data[column].nunique() == 2:
            method = "binary_passthrough"
            reason = "Already represented as a 0/1 indicator."
        else:
            method = "numeric_passthrough_or_scale_later"
            reason = "Keep numerical value; scaling choice belongs to ML data preparation after train/test split."
        rows.append({
            "column": column,
            "encoding_strategy": method,
            "unique_values_full_cleaned_data": int(data[column].nunique(dropna=False)),
            "reason": reason,
        })
    return pd.DataFrame(rows)


def validate_feature_dataset(featured: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    selected = selected_feature_columns(decisions)
    expected_columns = [TARGET, *selected]
    prepared = featured[expected_columns]
    checks = {
        "Target column is present": TARGET in prepared.columns,
        "All engineered columns selected": set(ENGINEERED_COLUMNS).issubset(prepared.columns),
        "Confirmed leakage columns absent": not set(LEAKAGE_COLUMNS).intersection(prepared.columns),
        "PII columns absent": not set(PII_COLUMNS).intersection(prepared.columns),
        "Prediction-time uncertain columns absent": not set(PREDICTION_TIME_EXCLUSIONS).intersection(prepared.columns),
        "No missing values in selected dataset": not prepared.isna().any().any(),
        "Target remains binary": set(prepared[TARGET].unique()) == {0, 1},
        "Row count preserved from cleaned dataset": len(prepared) == len(featured),
        "Selected column names are unique": len(prepared.columns) == len(set(prepared.columns)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("Feature engineering validation failed: " + "; ".join(failures))
    return pd.DataFrame({"check": checks.keys(), "passed": checks.values()})


def prepare_feature_engineering_outputs(cleaned: pd.DataFrame | None = None) -> dict:
    cleaned = load_cleaned() if cleaned is None else cleaned.copy(deep=True)
    featured = engineer_features(cleaned)
    decisions = build_feature_decisions(featured)
    selected = selected_feature_columns(decisions)
    prepared = featured[[TARGET, *selected]].copy()
    return {
        "cleaned": cleaned,
        "featured": featured,
        "prepared": prepared,
        "feature_types": feature_type_report(featured),
        "engineered_evaluation": evaluate_engineered_features(featured),
        "feature_decisions": decisions,
        "selected_features": pd.DataFrame({"selected_feature": selected}),
        "encoding_strategy": build_encoding_strategy(featured, selected),
        "validation": validate_feature_dataset(featured, decisions),
    }


def export_feature_engineering_outputs(result: dict, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    FEATURED_PATH.parent.mkdir(parents=True, exist_ok=True)
    result["prepared"].to_csv(FEATURED_PATH, index=False)
    result["feature_types"].to_csv(output_dir / "feature_type_review.csv", index=False)
    result["engineered_evaluation"].to_csv(output_dir / "engineered_feature_evaluation.csv", index=False)
    result["feature_decisions"].to_csv(output_dir / "feature_selection_decisions.csv", index=False)
    result["selected_features"].to_csv(output_dir / "final_selected_features.csv", index=False)
    result["encoding_strategy"].to_csv(output_dir / "encoding_strategy.csv", index=False)
    result["validation"].to_csv(output_dir / "feature_engineering_validation.csv", index=False)
    reloaded = pd.read_csv(FEATURED_PATH)
    if reloaded.shape != result["prepared"].shape:
        raise ValueError("Saved feature dataset shape changed on reload.")
