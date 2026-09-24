"""Member 1 data cleaning, type correction and data-quality checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/hotel_booking.csv"
REPORT_DIR = ROOT / "03_Data_Preprocessing_and_Feature_Engineering"
PROCESSED_PATH = ROOT / "data/processed/hotel_booking_cleaned.csv"
PII_COLUMNS = ["name", "email", "phone-number", "credit_card"]
COUNT_COLUMNS = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "required_car_parking_spaces",
    "total_of_special_requests",
]
DATE_COLUMNS = ["arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"]
BINARY_COLUMNS = ["is_canceled", "is_repeated_guest"]
ID_COLUMNS = ["agent", "company"]
CATEGORY_COLUMNS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type", "reservation_status", *ID_COLUMNS,
]
MONTHS = {name: i for i, name in enumerate([
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
], start=1)}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    # Keep PII textual, particularly telephone numbers and masked card strings.
    return pd.read_csv(path, dtype={c: "string" for c in PII_COLUMNS})


def parse_status_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    # Raw lecturer file uses month/day/year; exported CSV uses ISO dates.
    parsed = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")
    return parsed.fillna(pd.to_datetime(values, format="%Y-%m-%d", errors="coerce"))


def invalid_masks(data: pd.DataFrame) -> pd.DataFrame:
    """Row-level checks against the documented cleaning rules."""
    counts = data[COUNT_COLUMNS]
    dates = data[DATE_COLUMNS]
    arrival = pd.to_datetime(pd.DataFrame({
        "year": data["arrival_date_year"],
        "month": data["arrival_date_month"].map(MONTHS),
        "day": data["arrival_date_day_of_month"],
    }), errors="coerce")
    return pd.DataFrame({
        "missing_children": data["children"].isna(),
        "invalid_counts": (
            counts.isna() | ~np.isfinite(counts) | counts.lt(0) | counts.mod(1).ne(0)
        ).any(axis=1),
        "invalid_binary": ~data[BINARY_COLUMNS].isin([0, 1]).all(axis=1),
        "zero_guests": data[["adults", "children", "babies"]].sum(axis=1, min_count=3).eq(0),
        "invalid_adr": data["adr"].isna() | ~np.isfinite(data["adr"]) | data["adr"].lt(0),
        "invalid_arrival": arrival.isna() | (
            dates.isna() | ~np.isfinite(dates) | dates.mod(1).ne(0)
        ).any(axis=1) | ~data["arrival_date_week_number"].between(1, 53),
        "invalid_status_date": parse_status_dates(data["reservation_status_date"]).isna(),
    }, index=data.index)


def remove_invalid_and_duplicates(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw.index.is_unique:
        raise ValueError("A unique source-row index is required for the audit trail.")
    reasons = invalid_masks(raw)
    reasons["exact_duplicate"] = raw.duplicated(keep="first")
    excluded = reasons.any(axis=1)
    audit = pd.DataFrame({
        "source_row_index": raw.index[excluded],
        "reason": [
            "; ".join(reasons.columns[row])
            for row in reasons.loc[excluded].to_numpy(dtype=bool)
        ],
    })
    return raw.loc[~excluded].copy(), audit


def correct_types_and_missing(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy(deep=True)
    cleaned["country"] = cleaned["country"].fillna("Unknown")
    for column in ID_COLUMNS:
        ids = pd.to_numeric(cleaned[column], errors="raise")
        observed = ids.dropna()
        if ((~np.isfinite(observed)) | observed.lt(0) | observed.mod(1).ne(0)).any():
            raise ValueError(f"Invalid {column} IDs; review the source before conversion.")
        # Not_recorded avoids claiming that a missing ID proves no agency/company.
        cleaned[column] = ids.astype("Int64").astype("string").fillna("Not_recorded")
    for column in COUNT_COLUMNS + DATE_COLUMNS:
        cleaned[column] = cleaned[column].astype("int64")
    for column in BINARY_COLUMNS:
        cleaned[column] = cleaned[column].astype("int8")
    cleaned["reservation_status_date"] = parse_status_dates(cleaned["reservation_status_date"])
    for column in CATEGORY_COLUMNS:
        cleaned[column] = cleaned[column].astype("category")
    return cleaned


def quality_metrics(data: pd.DataFrame) -> pd.Series:
    masks = invalid_masks(data)
    metrics = {
        "rows": len(data), "columns": len(data.columns),
        "missing_cells": int(data.isna().sum().sum()),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "matching_non_pii_rows_review_only": int(data.drop(columns=PII_COLUMNS).duplicated().sum()),
        "invalid_rows_any_rule": int(masks.any(axis=1).sum()),
        "zero_night_rows_retained": int(data[["stays_in_weekend_nights", "stays_in_week_nights"]].sum(axis=1).eq(0).sum()),
        "zero_adr_rows_retained": int(data["adr"].eq(0).sum()),
        "waiting_exceeds_lead_time_review_only": int(data["days_in_waiting_list"].gt(data["lead_time"]).sum()),
        "not_repeated_with_history_review_only": int((data["is_repeated_guest"].eq(0) & data[["previous_cancellations", "previous_bookings_not_canceled"]].sum(axis=1).gt(0)).sum()),
        "repeated_without_history_review_only": int((data["is_repeated_guest"].eq(1) & data[["previous_cancellations", "previous_bookings_not_canceled"]].sum(axis=1).eq(0)).sum()),
        "target_0_count": int(data["is_canceled"].eq(0).sum()),
        "target_1_count": int(data["is_canceled"].eq(1).sum()),
        "cancellation_rate_pct": data["is_canceled"].mean() * 100,
    }
    metrics.update({f"rows_{name}": int(mask.sum()) for name, mask in masks.items()})
    metrics.update({f"missing_{c}": int(data[c].isna().sum()) for c in data.columns})
    return pd.Series(metrics, name="value")


def outlier_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in COUNT_COLUMNS + ["adr"]:
        q1, q3 = raw[column].quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append({
            "column": column, "q1_raw": q1, "q3_raw": q3,
            "lower_fence_diagnostic_only": lower, "upper_fence_diagnostic_only": upper,
            "flagged_before": int((raw[column].lt(lower) | raw[column].gt(upper)).sum()),
            "flagged_after_same_fences": int((cleaned[column].lt(lower) | cleaned[column].gt(upper)).sum()),
            "min_before": raw[column].min(), "max_before": raw[column].max(),
            "min_after": cleaned[column].min(), "max_after": cleaned[column].max(),
            "action": "Retain valid extremes; IQR is diagnostic only, never used for filtering/capping.",
        })
    return pd.DataFrame(rows)


def validate_cleaned(raw: pd.DataFrame, cleaned: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    checks = {
        "Non-empty cleaned dataset": len(cleaned) > 0,
        "All original columns preserved during cleaning": cleaned.columns.equals(raw.columns),
        "No remaining missing values": not cleaned.isna().any().any(),
        "No invalid values under documented rules": not invalid_masks(cleaned).any().any(),
        "No exact duplicate records": not cleaned.duplicated().any(),
        "Row counts reconcile with removal audit": len(raw) == len(cleaned) + len(audit),
        "Retained rows and excluded rows partition source": (
            set(cleaned.index).isdisjoint(audit["source_row_index"])
            and set(cleaned.index) | set(audit["source_row_index"]) == set(raw.index)
        ),
        "Target labels unchanged for every retained row": np.array_equal(
            raw.loc[cleaned.index, "is_canceled"].to_numpy(), cleaned["is_canceled"].to_numpy()
        ),
        "Counts have integer types": all(pd.api.types.is_integer_dtype(cleaned[c]) for c in COUNT_COLUMNS + DATE_COLUMNS),
        "Binary fields have integer types": all(pd.api.types.is_integer_dtype(cleaned[c]) for c in BINARY_COLUMNS),
        "Categorical fields have category types": all(isinstance(cleaned[c].dtype, pd.CategoricalDtype) for c in CATEGORY_COLUMNS),
        "Status date has datetime type": pd.api.types.is_datetime64_any_dtype(cleaned["reservation_status_date"]),
        "Numerical values unchanged on retained rows": all(
            np.array_equal(raw.loc[cleaned.index, c].to_numpy(), cleaned[c].to_numpy())
            for c in COUNT_COLUMNS + ["adr"]
        ),
    }
    for name, passed in checks.items():
        require(bool(passed), name)
    return pd.DataFrame({"check": checks.keys(), "passed": checks.values()})


def load_cleaned(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """CSV has no dtype metadata; restore the cleaning-stage schema explicitly."""
    data = pd.read_csv(path, dtype={c: "string" for c in PII_COLUMNS + ID_COLUMNS})
    for column in COUNT_COLUMNS + DATE_COLUMNS:
        data[column] = data[column].astype("int64")
    for column in BINARY_COLUMNS:
        data[column] = data[column].astype("int8")
    for column in CATEGORY_COLUMNS:
        data[column] = data[column].astype("category")
    data["reservation_status_date"] = pd.to_datetime(data["reservation_status_date"], format="%Y-%m-%d")
    return data
