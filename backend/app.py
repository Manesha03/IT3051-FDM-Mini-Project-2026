"""Backend service for the hotel booking cancellation predictor.

Run from the project root with:
    python backend/app.py
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "06_Model_Evaluation" / "Models" / "tuned_random_forest_pipeline.joblib"
METADATA_PATH = ROOT / "06_Model_Evaluation" / "Models" / "tuned_random_forest_metadata.json"
FEATURE_DATA_PATH = ROOT / "data" / "processed" / "hotel_booking_features.csv"


TEXT_FIELDS = {
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
}
INT_RANGES = {
    "lead_time": (0, 1000),
    "arrival_date_year": (2015, 2035),
    "arrival_date_week_number": (1, 53),
    "arrival_date_day_of_month": (1, 31),
    "stays_in_weekend_nights": (0, 30),
    "stays_in_week_nights": (0, 60),
    "adults": (0, 20),
    "children": (0, 20),
    "babies": (0, 10),
    "previous_cancellations": (0, 100),
    "previous_bookings_not_canceled": (0, 100),
    "required_car_parking_spaces": (0, 10),
    "total_of_special_requests": (0, 10),
}
FLOAT_RANGES = {"adr": (0.0, 10000.0)}
FIELD_LABELS = {
    "adr": "average daily rate",
    "agent": "agent ID",
    "company": "company ID",
    "is_repeated_guest": "repeated guest",
}
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


MODEL = joblib.load(MODEL_PATH)
METADATA = load_json(METADATA_PATH)
PREDICTOR_COLUMNS = METADATA["predictor_columns"]


def feature_options() -> dict[str, list[str]]:
    data = pd.read_csv(
        FEATURE_DATA_PATH,
        usecols=[column for column in TEXT_FIELDS if column in PREDICTOR_COLUMNS],
        dtype="string",
    )
    options = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in data.columns
    }
    if "arrival_date_month" in options:
        options["arrival_date_month"] = [month for month in MONTH_ORDER if month in options["arrival_date_month"]]
    return options


OPTIONS = feature_options()


def chart_rows(series: pd.Series, limit: int | None = None) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    total = int(counts.sum())
    return [
        {"label": str(label), "count": int(count), "percent": round(int(count) / total * 100, 2)}
        for label, count in counts.items()
    ]


def cancellation_rate_by(data: pd.DataFrame, column: str, limit: int = 12) -> list[dict[str, Any]]:
    grouped = (
        data.groupby(column, observed=False)["is_canceled"]
        .agg(bookings="count", cancellation_rate="mean")
        .sort_values("bookings", ascending=False)
        .head(limit)
        .reset_index()
    )
    return [
        {
            "label": str(row[column]),
            "bookings": int(row["bookings"]),
            "cancellation_rate": round(float(row["cancellation_rate"]) * 100, 2),
        }
        for _, row in grouped.iterrows()
    ]


def grouped_feature_importance(limit: int = 12) -> list[dict[str, Any]]:
    try:
        preprocessor = MODEL.named_steps["preprocessor"]
        forest = MODEL.named_steps["model"]
        transformed_names = preprocessor.get_feature_names_out()
        rows = []
        for name, importance in zip(transformed_names, forest.feature_importances_):
            clean_name = name.split("__", 1)[-1]
            matches = [
                column for column in PREDICTOR_COLUMNS
                if clean_name == column or clean_name.startswith(f"{column}_")
            ]
            feature = max(matches, key=len) if matches else clean_name
            rows.append({"feature": feature, "importance": float(importance)})
        importance = (
            pd.DataFrame(rows)
            .groupby("feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .head(limit)
        )
        return [
            {"label": row["feature"], "importance": round(float(row["importance"]), 5)}
            for _, row in importance.iterrows()
        ]
    except Exception:
        return []


def dashboard_payload() -> dict[str, Any]:
    data = pd.read_csv(FEATURE_DATA_PATH, dtype={"agent": "string", "company": "string", "country": "string"})
    rows = len(data)
    cancelled = int(data["is_canceled"].sum())
    not_cancelled = rows - cancelled
    avg_lead_time = round(float(data["lead_time"].mean()), 1)
    avg_adr = round(float(data["adr"].mean()), 2)
    avg_stay = round(float(data["total_stay"].mean()), 1)
    return {
        "ok": True,
        "overview": {
            "total_bookings": rows,
            "cancelled_bookings": cancelled,
            "not_cancelled_bookings": not_cancelled,
            "cancellation_rate": round(cancelled / rows * 100, 2),
            "avg_lead_time": avg_lead_time,
            "avg_adr": avg_adr,
            "avg_stay": avg_stay,
            "predictor_count": len(PREDICTOR_COLUMNS),
            "model": METADATA["model"],
            "version": METADATA["version"],
        },
        "charts": {
            "target_distribution": [
                {"label": "Not cancelled", "count": not_cancelled},
                {"label": "Cancelled", "count": cancelled},
            ],
            "hotel_type": chart_rows(data["hotel"]),
            "arrival_month": cancellation_rate_by(data, "arrival_date_month", limit=12),
            "market_segment": cancellation_rate_by(data, "market_segment", limit=10),
            "deposit_type": cancellation_rate_by(data, "deposit_type", limit=6),
            "customer_type": cancellation_rate_by(data, "customer_type", limit=8),
            "lead_time_bands": lead_time_bands(data),
            "top_countries": cancellation_rate_by(data, "country", limit=10),
            "feature_importance": grouped_feature_importance(),
        },
        "recommendations": [
            {
                "title": "Prioritize high-risk bookings",
                "priority": "High",
                "detail": "Review bookings with non-refundable deposits, long lead times, or previous cancellations before the arrival date.",
            },
            {
                "title": "Confirm early bookings",
                "priority": "High",
                "detail": "Bookings made more than 90 days in advance show higher cancellation risk, so send confirmation reminders closer to arrival.",
            },
            {
                "title": "Monitor market segments",
                "priority": "Medium",
                "detail": "Track segments with higher cancellation percentages and plan follow-up calls or email confirmations for those reservations.",
            },
            {
                "title": "Use special requests as a service signal",
                "priority": "Low",
                "detail": "Guests with clear preferences or special requests can be handled with personalized communication to improve confidence.",
            },
        ],
    }


def lead_time_bands(data: pd.DataFrame) -> list[dict[str, Any]]:
    bins = [-1, 7, 30, 90, 180, 365, 1000]
    labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    frame = data.copy()
    frame["lead_time_band"] = pd.cut(frame["lead_time"], bins=bins, labels=labels)
    return cancellation_rate_by(frame, "lead_time_band", limit=len(labels))


def display_name(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))


def require_text(payload: dict[str, Any], field: str, errors: dict[str, str]) -> str | None:
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        errors[field] = f"{display_name(field).title()} is required."
        return None
    return str(value).strip()


def require_int(payload: dict[str, Any], field: str, errors: dict[str, str]) -> int | None:
    value = payload.get(field)
    low, high = INT_RANGES[field]
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors[field] = f"{display_name(field).title()} must be a whole number."
        return None
    if number < low or number > high:
        errors[field] = f"{display_name(field).title()} must be between {low} and {high}."
        return None
    return number


def require_float(payload: dict[str, Any], field: str, errors: dict[str, str]) -> float | None:
    value = payload.get(field)
    low, high = FLOAT_RANGES[field]
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors[field] = f"{display_name(field).title()} must be a number."
        return None
    if number < low or number > high:
        errors[field] = f"{display_name(field).title()} must be between {low:g} and {high:g}."
        return None
    return number


def build_features(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str]]:
    errors: dict[str, str] = {}
    features: dict[str, Any] = {}

    for field in TEXT_FIELDS:
        if field in PREDICTOR_COLUMNS:
            value = require_text(payload, field, errors)
            if value is not None:
                features[field] = value

    for field in INT_RANGES:
        if field in PREDICTOR_COLUMNS:
            value = require_int(payload, field, errors)
            if value is not None:
                features[field] = value

    for field in FLOAT_RANGES:
        if field in PREDICTOR_COLUMNS:
            value = require_float(payload, field, errors)
            if value is not None:
                features[field] = value

    repeated_guest = payload.get("is_repeated_guest")
    if repeated_guest in (True, "true", "1", 1):
        features["is_repeated_guest"] = 1
    elif repeated_guest in (False, "false", "0", 0):
        features["is_repeated_guest"] = 0
    else:
        errors["is_repeated_guest"] = "Repeated guest must be yes or no."

    if not errors:
        features["total_stay"] = features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
        features["total_guests"] = features["adults"] + features["children"] + features["babies"]
        features["has_children"] = int(features["children"] > 0)
        features["has_babies"] = int(features["babies"] > 0)
        features["previous_total_bookings"] = (
            features["previous_cancellations"] + features["previous_bookings_not_canceled"]
        )
        if features["previous_total_bookings"] > 0:
            features["previous_cancellation_rate"] = (
                features["previous_cancellations"] / features["previous_total_bookings"]
            )
        else:
            features["previous_cancellation_rate"] = 0.0
        if features["total_guests"] <= 0:
            errors["total_guests"] = "At least one adult, child, or baby is required."

    if errors:
        return None, errors
    return {column: features[column] for column in PREDICTOR_COLUMNS}, {}


def predict(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    features, errors = build_features(payload)
    if errors:
        return {"ok": False, "errors": errors}, HTTPStatus.BAD_REQUEST

    frame = pd.DataFrame([features], columns=PREDICTOR_COLUMNS)
    prediction = int(MODEL.predict(frame)[0])
    probability = None
    if hasattr(MODEL, "predict_proba"):
        probability = float(MODEL.predict_proba(frame)[0][1])

    return {
        "ok": True,
        "prediction": prediction,
        "prediction_label": "Likely to cancel" if prediction == 1 else "Likely not to cancel",
        "cancellation_probability": probability,
        "model": METADATA["model"],
        "version": METADATA["version"],
        "features": features,
    }, HTTPStatus.OK


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path.startswith("/api/v1/"):
            path = "/api/" + path.removeprefix("/api/v1/")
        if path == "/api/health":
            self.send_json({"ok": True, "model": METADATA["model"], "version": METADATA["version"]})
            return
        if path == "/api/schema":
            self.send_json({
                "ok": True,
                "predictor_columns": PREDICTOR_COLUMNS,
                "options": OPTIONS,
                "int_ranges": INT_RANGES,
                "float_ranges": FLOAT_RANGES,
            })
            return
        if path == "/api/dashboard":
            self.send_json(dashboard_payload())
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path.startswith("/api/v1/"):
            path = "/api/" + path.removeprefix("/api/v1/")
        if path != "/api/predict":
            self.send_json({"ok": False, "error": "Endpoint not found."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        result, status = predict(payload)
        self.send_json(result, status)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)
    print(f"Hotel cancellation prediction service running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
