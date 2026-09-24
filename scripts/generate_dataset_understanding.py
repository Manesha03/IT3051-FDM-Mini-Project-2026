from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "raw" / "hotel_booking.csv"
OUT_DIR = ROOT / "01_Dataset_Understanding"
RAW_DIR = ROOT / "data" / "raw"

EXTRA_CSVS_TO_REMOVE = [
    "categorical_value_distributions.csv",
    "initial_class_distribution_analysis.csv",
    "initial_data_quality_findings.csv",
    "missing_value_summary.csv",
    "numerical_descriptive_statistics.csv",
    "potential_leakage_variable_list.csv",
    "target_variable_analysis.csv",
]

ATTRIBUTE_DESCRIPTIONS = {
    "hotel": ("Type of hotel for the booking.", "Categorical"),
    "is_canceled": ("Target variable: 1 if the booking was canceled and 0 if it was not canceled.", "Binary"),
    "lead_time": ("Number of days between booking creation and arrival date.", "Numerical"),
    "arrival_date_year": ("Year of arrival.", "Date/Time"),
    "arrival_date_month": ("Month of arrival.", "Date/Time"),
    "arrival_date_week_number": ("Week number of the arrival date.", "Date/Time"),
    "arrival_date_day_of_month": ("Day of month of arrival.", "Date/Time"),
    "stays_in_weekend_nights": ("Number of weekend nights included in the stay.", "Numerical"),
    "stays_in_week_nights": ("Number of week nights included in the stay.", "Numerical"),
    "adults": ("Number of adults in the booking.", "Numerical"),
    "children": ("Number of children in the booking.", "Numerical"),
    "babies": ("Number of babies in the booking.", "Numerical"),
    "meal": ("Meal package booked by the guest.", "Categorical"),
    "country": ("Guest country of origin.", "Categorical"),
    "market_segment": ("Market segment through which the booking was made.", "Categorical"),
    "distribution_channel": ("Booking distribution channel.", "Categorical"),
    "is_repeated_guest": ("Indicates whether the guest is a repeated guest.", "Binary"),
    "previous_cancellations": ("Number of previous bookings canceled by the customer.", "Numerical"),
    "previous_bookings_not_canceled": ("Number of previous bookings not canceled by the customer.", "Numerical"),
    "reserved_room_type": ("Room type originally reserved.", "Categorical"),
    "assigned_room_type": ("Room type assigned to the booking.", "Categorical"),
    "booking_changes": ("Number of changes made to the booking.", "Numerical"),
    "deposit_type": ("Type of deposit made for the booking.", "Categorical"),
    "agent": ("Travel agency identifier.", "Categorical"),
    "company": ("Company/entity identifier associated with the booking.", "Categorical"),
    "days_in_waiting_list": ("Number of days the booking stayed on the waiting list.", "Numerical"),
    "customer_type": ("Type of customer or booking contract.", "Categorical"),
    "adr": ("Average Daily Rate calculated for the booking.", "Numerical"),
    "required_car_parking_spaces": ("Number of car parking spaces required by the guest.", "Numerical"),
    "total_of_special_requests": ("Number of special requests made by the guest.", "Numerical"),
    "reservation_status": ("Final reservation status.", "Categorical"),
    "reservation_status_date": ("Date when the final reservation status was last updated.", "Date/Time"),
    "name": ("Guest name included in the validated dataset.", "Identifier/PII"),
    "email": ("Guest email address included in the validated dataset.", "Identifier/PII"),
    "phone-number": ("Guest phone number included in the validated dataset.", "Identifier/PII"),
    "credit_card": ("Masked credit card value included in the validated dataset.", "Identifier/PII"),
}

POTENTIAL_LEAKAGE = {
    "reservation_status": "Confirmed leakage risk because it directly represents the final booking outcome.",
    "reservation_status_date": "Confirmed leakage risk because it may be known only after cancellation/check-out status is confirmed.",
    "assigned_room_type": "Not confirmed leakage, but a possible post-booking operational attribute that requires investigation before modeling.",
}

ATTRIBUTE_NOTES = {
    "hotel": "Useful categorical predictor; should be encoded during preprocessing.",
    "is_canceled": "Prediction target; must not be used as an input feature.",
    "lead_time": "Important booking-timing predictor; check for high/extreme values during EDA.",
    "arrival_date_year": "Date component; useful for period coverage and possible time-based patterns.",
    "arrival_date_month": "Date component; useful for seasonality analysis and should be encoded carefully.",
    "arrival_date_week_number": "Date component; useful for weekly arrival patterns.",
    "arrival_date_day_of_month": "Date component; useful for arrival-day pattern checks.",
    "stays_in_weekend_nights": "Stay-length feature; check zero-night and extreme-stay records.",
    "stays_in_week_nights": "Stay-length feature; check zero-night and extreme-stay records.",
    "adults": "Guest-count feature; check records with zero total guests.",
    "children": "Guest-count feature; contains a small number of missing values.",
    "babies": "Guest-count feature; check rare or unusual high values during EDA.",
    "meal": "Categorical predictor; Undefined/SC categories should be reviewed before encoding.",
    "country": "Categorical predictor with many unique values and some missing values.",
    "market_segment": "Categorical predictor; Undefined values require investigation.",
    "distribution_channel": "Categorical predictor; Undefined values require investigation.",
    "is_repeated_guest": "Binary predictor that may indicate guest history.",
    "previous_cancellations": "Customer-history feature; likely relevant to cancellation behavior.",
    "previous_bookings_not_canceled": "Customer-history feature; likely relevant to repeat guest behavior.",
    "reserved_room_type": "Original room type selected at booking time; should be encoded during preprocessing.",
    "booking_changes": "May reflect customer/booking changes before final outcome; timing should be understood before modeling.",
    "deposit_type": "Categorical predictor that may be strongly related to cancellation behavior.",
    "agent": "Identifier-like categorical field with missing values; missing may mean no agent.",
    "company": "Identifier-like categorical field with many missing values; missing may mean no company association.",
    "days_in_waiting_list": "Operational feature; check skewness and high values during EDA.",
    "customer_type": "Categorical predictor describing booking/customer contract type.",
    "adr": "Numerical price-related feature; check negative and extreme values.",
    "required_car_parking_spaces": "Guest request feature; may indicate stronger booking intention.",
    "total_of_special_requests": "Guest request feature; may indicate stronger booking intention.",
    "name": "PII/identifier field; should not be used as a predictive feature.",
    "email": "PII/identifier field; should not be used as a predictive feature.",
    "phone-number": "PII/identifier field; should not be used as a predictive feature.",
    "credit_card": "Masked PII/identifier field; should not be used as a predictive feature.",
}


def make_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    rows = []
    for column in df.columns:
        description, attribute_type = ATTRIBUTE_DESCRIPTIONS[column]
        examples = df[column].dropna().astype(str).unique()[:5]
        rows.append(
            {
                "attribute": column,
                "description": description,
                "attribute_type": attribute_type,
                "pandas_dtype": str(df[column].dtype),
                "missing_count": int(missing[column]),
                "missing_percentage": round(float(missing[column] / len(df) * 100), 3),
                "unique_values": int(df[column].nunique(dropna=True)),
                "example_values": "; ".join(examples),
                "notes": POTENTIAL_LEAKAGE.get(column, ATTRIBUTE_NOTES.get(column, "")),
            }
        )
    return pd.DataFrame(rows)


def make_characteristics(df: pd.DataFrame, data_dictionary: pd.DataFrame) -> pd.DataFrame:
    type_counts = (
        data_dictionary["attribute_type"]
        .value_counts()
        .rename_axis("feature_type")
        .reset_index(name="count")
        .sort_values("feature_type")
    )
    summary = pd.DataFrame(
        [
            {"metric": "Dataset file", "value": "hotel_booking.csv"},
            {"metric": "Dataset status", "value": "Validated lecturer-approved raw dataset"},
            {"metric": "Original source", "value": "Hotel Booking Demand Datasets by Antonio, de Almeida, and Nunes (2019)"},
            {"metric": "Period covered", "value": "Scheduled arrivals from 1 July 2015 to 31 August 2017"},
            {"metric": "Problem type", "value": "Binary classification"},
            {"metric": "Prediction objective", "value": "Predict whether a hotel booking will be canceled"},
            {"metric": "Target variable", "value": "is_canceled"},
            {"metric": "Number of records", "value": len(df)},
            {"metric": "Number of columns including target", "value": df.shape[1]},
            {"metric": "Fully duplicated records", "value": int(df.duplicated().sum())},
            {"metric": "Columns with missing values", "value": int((df.isna().sum() > 0).sum())},
        ]
    )
    feature_rows = type_counts.assign(
        metric=lambda x: "Feature type - " + x["feature_type"],
        value=lambda x: x["count"],
    )[["metric", "value"]]
    return pd.concat([summary, feature_rows], ignore_index=True)


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            "# Hotel Booking Cancellation Prediction\n\n"
            "## 1 - Dataset Selection & Understanding\n\n"
            "### Objective\n\n"
            "This notebook documents the selection and initial understanding of the validated Hotel Booking dataset used for the IT3051 FDM Mini Project.\n\n"
            "The main objectives are to:\n\n"
            "1. Understand the real-world hotel booking cancellation problem.\n"
            "2. Document the approved dataset source and dataset description.\n"
            "3. Identify the target variable, features, and feature types.\n"
            "4. Create a complete data dictionary for the relevant attributes.\n"
            "5. Examine basic numerical statistics and categorical distributions.\n"
            "6. Investigate missing values, duplicates, invalid/unusual values, and potential leakage variables.\n"
            "7. Confirm the suitability of the dataset for hotel booking cancellation prediction.\n\n"
            "### Selected Dataset\n\n"
            "The selected dataset is the lecturer-validated `hotel_booking.csv` dataset.\n\n"
            "Each row represents a hotel booking record. The columns contain information about the hotel type, booking timing, arrival date, length of stay, guest counts, meal plan, country, market segment, distribution channel, previous booking behavior, room type, deposit type, customer type, average daily rate, reservation status, and selected guest identifier fields.\n\n"
            "The target variable is `is_canceled`, which indicates whether a booking was canceled."
        ),
        markdown_cell("## 1. Problem Definition\n\nHotel booking cancellations create operational and financial challenges for hotels. When a confirmed booking is canceled, the hotel may lose expected revenue and may also face difficulties with room allocation, staffing, inventory planning, and demand forecasting. Understanding cancellation patterns helps hotels make better decisions about pricing, overbooking, and customer management."),
        markdown_cell("## 2. Prediction Objective\n\nThe prediction objective is to use historical booking information to predict whether a hotel booking will be canceled. This is a supervised binary classification problem because the output has two classes: canceled and not canceled."),
        markdown_cell(
            "## 3. Stakeholders and Decision Support\n\n"
            "The primary users of the proposed Hotel Booking Cancellation Prediction system are hotel managers, reservation staff, revenue managers, front-office staff, and operations teams.\n\n"
            "The prediction system can help hotels identify bookings with a higher risk of cancellation before the final booking outcome is known. This information may support decisions related to:\n\n"
            "- Room and occupancy planning\n"
            "- Revenue management\n"
            "- Staffing and resource planning\n"
            "- Booking follow-up strategies\n"
            "- Overbooking and inventory decisions\n\n"
            "### Expected System Input\n\n"
            "The final system is expected to use booking-related information available at prediction time, such as hotel type, lead time, arrival period, stay details, guest information, meal type, market segment, distribution channel, deposit type, customer history, parking requirements, and special requests.\n\n"
            "### Expected System Output\n\n"
            "The final system is expected to predict whether a booking is likely to be canceled or not canceled. Where supported by the selected model, it may also provide a cancellation probability or risk level."
        ),
        markdown_cell("## 4. Import Libraries"),
        code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "from pathlib import Path\n\n"
            "pd.set_option('display.max_columns', None)\n"
            "pd.set_option('display.max_rows', 120)"
        ),
        markdown_cell(
            "## 5. Load the Approved Raw Dataset\n\n"
            "The approved raw dataset is stored in the project folder under `data/raw/hotel_booking.csv`. The raw dataset is loaded without preprocessing or transformation.\n\n"
            "**Original source:** Hotel Booking Demand Datasets by Antonio, de Almeida, and Nunes (2019).\n\n"
            "**Citation:** Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126\n\n"
            "**Original URL:** https://www.sciencedirect.com/science/article/pii/S2352340918315191\n\n"
            "**Access mirror:** https://www.kaggle.com/datasets/mojtaba142/hotel-booking\n\n"
            "**Period covered:** Scheduled arrivals from 1 July 2015 to 31 August 2017."
        ),
        code_cell(
            "DATA_PATH = Path('../data/raw/hotel_booking.csv')\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "print('Dataset loaded successfully')"
        ),
        markdown_cell("## 6. Preview the Dataset\n\nThe first and last records are displayed to understand the general appearance of the dataset."),
        code_cell("df.head()"),
        code_cell("df.tail()"),
        markdown_cell("## 7. Dataset Shape and General Structure\n\nThis section identifies the number of records, number of columns, and the general structure of the dataset."),
        code_cell(
            "dataset_shape = pd.DataFrame({\n"
            "    'Property': ['Number of Records', 'Number of Columns'],\n"
            "    'Value': [df.shape[0], df.shape[1]]\n"
            "})\n"
            "dataset_shape"
        ),
        markdown_cell("## 8. Column Information\n\nThe following table shows all attributes and their pandas data types. This helps identify numerical, categorical, binary, date/time, and identifier fields."),
        code_cell("pd.DataFrame({'column': df.columns, 'pandas_dtype': df.dtypes.astype(str).values})"),
        markdown_cell("## 9. Target Variable Identification\n\nThe target variable is `is_canceled`.\n\n- `0` means the booking was not canceled.\n- `1` means the booking was canceled.\n\nThis target makes the project a binary classification problem."),
        code_cell(
            "target_distribution = (\n"
            "    df['is_canceled']\n"
            "    .value_counts()\n"
            "    .sort_index()\n"
            "    .rename_axis('is_canceled')\n"
            "    .reset_index(name='count')\n"
            ")\n"
            "target_distribution['meaning'] = target_distribution['is_canceled'].map({0: 'Not canceled', 1: 'Canceled'})\n"
            "target_distribution['percentage'] = (target_distribution['count'] / len(df) * 100).round(3)\n"
            "target_distribution"
        ),
        markdown_cell("## 10. Feature and Attribute Type Summary\n\nThe dataset contains numerical, categorical, binary, date/time, and identifier/PII attributes. The target variable is included in the binary category."),
        code_cell(
            "data_dictionary = pd.read_csv('data_dictionary.csv')\n"
            "feature_type_summary = (\n"
            "    data_dictionary['attribute_type']\n"
            "    .value_counts()\n"
            "    .rename_axis('Attribute Type')\n"
            "    .reset_index(name='Count')\n"
            "    .sort_values('Attribute Type')\n"
            ")\n"
            "feature_type_summary"
        ),
        markdown_cell("## 11. Data Dictionary\n\nThe separate `data_dictionary.csv` file is kept as a reusable artifact because it provides a clean reference for all 36 attributes. It includes the attribute name, description, attribute type, pandas data type, missing-value information, number of unique values, example values, and important notes."),
        code_cell("data_dictionary = pd.read_csv('data_dictionary.csv')\ndata_dictionary"),
        markdown_cell("## 12. Numerical Descriptive Statistics\n\nThe following table provides initial descriptive statistics for numerical attributes. These statistics help understand central tendency, spread, minimum values, maximum values, and possible unusual values."),
        code_cell(
            "numeric_columns = [col for col in df.select_dtypes(include='number').columns if col != 'is_canceled']\n"
            "df[numeric_columns].describe().T.round(3)"
        ),
        markdown_cell("## 13. Categorical Value Investigation\n\nThis section examines unique values and basic category distributions for selected categorical attributes. The purpose is to understand dominant categories and identify unusual labels such as `Undefined`."),
        code_cell(
            "categorical_columns = [\n"
            "    'hotel', 'meal', 'country', 'market_segment', 'distribution_channel',\n"
            "    'reserved_room_type', 'assigned_room_type', 'deposit_type',\n"
            "    'customer_type', 'reservation_status'\n"
            "]\n\n"
            "categorical_summary = []\n"
            "for column in categorical_columns:\n"
            "    categorical_summary.append({\n"
            "        'attribute': column,\n"
            "        'unique_values': df[column].nunique(dropna=True),\n"
            "        'top_value': df[column].mode(dropna=True).iloc[0],\n"
            "        'top_value_count': int(df[column].value_counts(dropna=True).iloc[0]),\n"
            "        'top_value_percentage': round(df[column].value_counts(dropna=True).iloc[0] / len(df) * 100, 3)\n"
            "    })\n\n"
            "pd.DataFrame(categorical_summary)"
        ),
        code_cell(
            "for column in categorical_columns:\n"
            "    print(f'\\n{column}')\n"
            "    display(df[column].value_counts(dropna=False).head(10).to_frame('count'))"
        ),
        markdown_cell("## 14. Missing Value Investigation\n\nThis section identifies attributes with missing values. At the dataset understanding stage, the objective is to observe and document missingness, not to perform imputation or cleaning."),
        code_cell(
            "missing_summary = df.isna().sum().reset_index()\n"
            "missing_summary.columns = ['attribute', 'missing_count']\n"
            "missing_summary['missing_percentage'] = (missing_summary['missing_count'] / len(df) * 100).round(3)\n"
            "missing_summary = missing_summary[missing_summary['missing_count'] > 0]\n"
            "missing_summary.sort_values('missing_count', ascending=False)"
        ),
        markdown_cell("## 15. Duplicate Investigation\n\nThis section checks whether the dataset contains fully duplicated records."),
        code_cell("pd.DataFrame({'metric': ['Fully duplicated records'], 'value': [int(df.duplicated().sum())]})"),
        markdown_cell("## 16. Invalid / Unusual Values\n\nThis section performs an initial check for values that may require deeper investigation later. No cleaning is performed here because this stage is limited to dataset understanding."),
        code_cell(
            "unusual_values = pd.DataFrame([\n"
            "    {'check': 'Negative ADR values', 'count': int((df['adr'] < 0).sum())},\n"
            "    {'check': 'Zero total guests', 'count': int(((df['adults'] + df['children'].fillna(0) + df['babies']) == 0).sum())},\n"
            "    {'check': 'Zero total nights', 'count': int(((df['stays_in_weekend_nights'] + df['stays_in_week_nights']) == 0).sum())},\n"
            "    {'check': 'Undefined meal values', 'count': int((df['meal'] == 'Undefined').sum())},\n"
            "    {'check': 'Undefined market segment values', 'count': int((df['market_segment'] == 'Undefined').sum())},\n"
            "    {'check': 'Undefined distribution channel values', 'count': int((df['distribution_channel'] == 'Undefined').sum())},\n"
            "])\n"
            "unusual_values"
        ),
        markdown_cell("## 17. Initial Class Distribution\n\nThe target-variable distribution shows whether the cancellation classes are balanced or imbalanced. This is useful for later modeling decisions, but no modeling is performed in this stage."),
        code_cell("target_distribution"),
        markdown_cell(
            "## 18. Initial Data-Quality Observations\n\n"
            "The initial investigation identified several data-quality issues requiring further analysis during EDA and preprocessing:\n\n"
            "- Missing values are present in four attributes: `company`, `agent`, `country`, and `children`.\n"
            "- `company` contains 112,593 missing values, which is 94.307% of the dataset.\n"
            "- `agent` contains 16,340 missing values, which is 13.686% of the dataset.\n"
            "- `country` contains 488 missing values and `children` contains 4 missing values.\n"
            "- No fully duplicated rows were identified.\n"
            "- 180 records contain zero total guests.\n"
            "- 715 records contain zero total stay nights.\n"
            "- 1 record contains a negative ADR value.\n"
            "- Some categorical attributes contain `Undefined` values: `meal` has 1,169, `market_segment` has 2, and `distribution_channel` has 5.\n"
            "- PII/identifier attributes such as `name`, `email`, `phone-number`, and `credit_card` should not be used as predictive features.\n\n"
            "These findings should influence later EDA, preprocessing, feature engineering, and feature-selection decisions."
        ),
        markdown_cell("## 19. Potential Data Leakage\n\nPotential leakage variables are attributes that may reveal the booking outcome directly or may only be known after the cancellation decision has already happened. In this dataset, `reservation_status` and `reservation_status_date` are treated as clear leakage risks because they describe the final reservation outcome or the timing of that outcome. `assigned_room_type` is not treated as confirmed leakage at this stage, but it is flagged for further investigation because it may be assigned after booking operations and may not always be available at prediction time."),
        code_cell(
            "potential_leakage = pd.DataFrame([\n"
            "    {'attribute': 'reservation_status', 'leakage_status': 'Clear leakage risk', 'reason': 'Directly represents the final booking outcome.'},\n"
            "    {'attribute': 'reservation_status_date', 'leakage_status': 'Clear leakage risk', 'reason': 'May be known only after cancellation/check-out status is confirmed.'},\n"
            "    {'attribute': 'assigned_room_type', 'leakage_status': 'Requires investigation', 'reason': 'Not confirmed leakage, but may be a post-booking operational attribute depending on when room assignment occurs.'},\n"
            "])\n"
            "potential_leakage"
        ),
        markdown_cell(
            "### Identifier and PII Attributes\n\n"
            "The dataset also contains `name`, `email`, `phone-number`, and `credit_card`. These attributes represent personal or identifier-like information rather than meaningful booking characteristics.\n\n"
            "They are not the same as data-leakage variables. `reservation_status` and `reservation_status_date` create leakage risk because they can reveal the final booking outcome. In contrast, the PII/identifier attributes should be excluded for privacy, ethics, and poor predictive-feature relevance. Their removal will be formally handled and justified during preprocessing."
        ),
        code_cell(
            "identifier_pii = pd.DataFrame([\n"
            "    {'attribute': 'name', 'reason': 'PII/identifier field; not a meaningful booking predictor.'},\n"
            "    {'attribute': 'email', 'reason': 'PII/identifier field; should not be used directly for modelling.'},\n"
            "    {'attribute': 'phone-number', 'reason': 'PII/identifier field; should not be used directly for modelling.'},\n"
            "    {'attribute': 'credit_card', 'reason': 'Masked PII/identifier field; should not be used directly for modelling.'},\n"
            "])\n"
            "identifier_pii"
        ),
        markdown_cell("## 20. Dataset Characteristics Summary\n\nThe following table summarizes the main dataset characteristics required for the Dataset Selection & Understanding stage."),
        code_cell(
            "dataset_summary = {\n"
            "    'Dataset Name': 'Hotel Booking Dataset',\n"
            "    'Dataset Status': 'Lecturer-validated raw dataset',\n"
            "    'Number of Records': df.shape[0],\n"
            "    'Number of Columns': df.shape[1],\n"
            "    'Target Variable': 'is_canceled',\n"
            "    'Prediction Type': 'Binary classification',\n"
            "    'Canceled Bookings': int((df['is_canceled'] == 1).sum()),\n"
            "    'Non-Canceled Bookings': int((df['is_canceled'] == 0).sum()),\n"
            "    'Columns With Missing Values': int((df.isna().sum() > 0).sum()),\n"
            "    'Duplicate Rows': int(df.duplicated().sum())\n"
            "}\n\n"
            "summary_df = pd.DataFrame(dataset_summary.items(), columns=['Property', 'Value'])\n"
            "summary_df"
        ),
        markdown_cell("## 21. Final Overview"),
        code_cell(
            "print('=' * 70)\n"
            "print('HOTEL BOOKING DATASET - FINAL OVERVIEW')\n"
            "print('=' * 70)\n\n"
            "print('Dataset: Hotel Booking Dataset')\n"
            "print(f'Records: {df.shape[0]}')\n"
            "print(f'Columns: {df.shape[1]}')\n"
            "print('Target: is_canceled')\n"
            "print('Problem Type: Binary Classification')\n\n"
            "print('Target Distribution:')\n"
            "print(target_distribution[['meaning', 'count', 'percentage']])\n\n"
            "print('\\nMissing Values by Column:')\n"
            "print(missing_summary.sort_values('missing_count', ascending=False))\n\n"
            "print('\\nDuplicate Rows:')\n"
            "print(df.duplicated().sum())\n\n"
            "print('\\nPotential Leakage Variables:')\n"
            "print(potential_leakage)\n"
            "print('=' * 70)"
        ),
        markdown_cell(
            "## 22. Initial Dataset Limitations and Considerations\n\n"
            "Although the dataset is suitable for the prediction problem, several limitations must be considered:\n\n"
            "- Missing values exist in some attributes.\n"
            "- Some attributes contain unusual or potentially invalid values.\n"
            "- Some categorical variables have high cardinality.\n"
            "- Some variables may not be available at the intended prediction time.\n"
            "- Final reservation-status variables create data-leakage risks.\n"
            "- PII/identifier attributes should not be used for modelling.\n\n"
            "These issues will be investigated further during EDA and addressed appropriately during preprocessing."
        ),
        markdown_cell("## 23. Dataset Suitability\n\nThe dataset is suitable for the Hotel Booking Cancellation Prediction problem because it contains a clear binary target variable, a large number of booking records, and multiple relevant predictor attributes related to booking timing, stay details, guest composition, customer history, market channel, deposit type, and reservation behavior. It also contains realistic data-quality conditions such as missing values and unusual records, which makes it appropriate for Evaluation 1 tasks such as data understanding, initial investigation, and preparation for later EDA and preprocessing."),
        markdown_cell("## 24. Conclusion\n\nThe lecturer-validated Hotel Booking dataset was selected as the approved raw dataset for the Hotel Booking Cancellation Prediction project.\n\nThe dataset contains 119,390 booking records and 36 columns. Each record represents a hotel booking, and the target variable is `is_canceled`, where `0` represents a non-canceled booking and `1` represents a canceled booking. Therefore, the project is suitable for a supervised binary classification problem.\n\nThe initial dataset understanding stage identified the main dataset structure, attribute meanings, feature types, target distribution, stakeholder decision-support context, expected system inputs and outputs, missing-value conditions, duplicate status, unusual values, privacy/identifier concerns, limitations, and potential leakage variables. The raw dataset has been preserved without preprocessing or transformation. The data dictionary and dataset characteristics summary have been prepared for reference in the next stages of the project."),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_DATA)
    data_dictionary = make_data_dictionary(df)
    data_dictionary.to_csv(OUT_DIR / "data_dictionary.csv", index=False)
    make_characteristics(df, data_dictionary).to_csv(OUT_DIR / "dataset_characteristics_summary.csv", index=False)

    for filename in EXTRA_CSVS_TO_REMOVE:
        path = OUT_DIR / filename
        if path.exists():
            path.unlink()

    notebook = build_notebook()
    (OUT_DIR / "Dataset_Selection_and_Understanding.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
