from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "02_EDA"
FIGURES_DIR = EDA_DIR / "Figures" / "stay_reservation"
DATA_PATH = ROOT / "data" / "raw" / "hotel_booking.csv"


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
            "# 2 - Exploratory Data Analysis (EDA)\n\n"
            "## Hotel Booking Cancellation Prediction\n\n"
            "This notebook is prepared for the EDA stage of the IT3051 FDM Mini Project. "
            "The current completed section is **Member 2 - Stay & Reservation Analysis**. "
            "Placeholder sections are included for Member 1 and Member 3 so their analyses can be added later."
        ),
        markdown_cell(
            "## EDA Objective\n\n"
            "The purpose of EDA is to explore the Hotel Booking dataset to identify cancellation-related patterns, "
            "feature distributions, unusual observations, and potential feature-engineering opportunities. "
            "Findings from this stage will guide preprocessing, feature engineering, feature selection, and future modelling decisions."
        ),
        markdown_cell("## Import Libraries and Load Dataset"),
        code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "pd.set_option('display.max_columns', None)\n"
            "pd.set_option('display.max_rows', 120)\n"
            "sns.set_theme(style='whitegrid', palette='Set2')\n\n"
            "DATA_PATH = Path('../data/raw/hotel_booking.csv')\n"
            "FIGURE_DIR = Path('Figures/stay_reservation')\n"
            "FIGURE_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "df.head()"
        ),
        markdown_cell("## Helper Functions"),
        code_cell(
            "def cancellation_rate_table(data, column):\n"
            "    summary = (\n"
            "        data.groupby(column, dropna=False)['is_canceled']\n"
            "        .agg(total_bookings='count', canceled_bookings='sum', cancellation_rate='mean')\n"
            "        .reset_index()\n"
            "    )\n"
            "    summary['cancellation_rate'] = (summary['cancellation_rate'] * 100).round(2)\n"
            "    return summary.sort_values('cancellation_rate', ascending=False)\n\n"
            "def save_current_figure(filename):\n"
            "    path = FIGURE_DIR / filename\n"
            "    plt.tight_layout()\n"
            "    plt.savefig(path, dpi=150, bbox_inches='tight')\n"
            "    plt.show()\n"
            "    print(f'Saved figure: {path}')"
        ),
        markdown_cell("## Overall Target Distribution\n\nThis gives context for Member 2 target comparisons. A cancellation rate of the full dataset is useful as a baseline."),
        code_cell(
            "target_summary = df['is_canceled'].value_counts().sort_index().reset_index()\n"
            "target_summary.columns = ['is_canceled', 'count']\n"
            "target_summary['meaning'] = target_summary['is_canceled'].map({0: 'Not canceled', 1: 'Canceled'})\n"
            "target_summary['percentage'] = (target_summary['count'] / len(df) * 100).round(2)\n"
            "target_summary"
        ),
        markdown_cell("## Member 1 - Booking & Customer Behaviour Analysis (Placeholder)\n\nThis section is reserved for Member 1. Suggested variables include lead time, previous cancellations, previous bookings not canceled, repeated guest, booking changes, days in waiting list, special requests, customer type, and related target comparisons."),
        markdown_cell("## Member 2 - Stay & Reservation Analysis\n\nThis section analyses stay and reservation attributes assigned to Member 2:\n\n- Hotel type\n- Arrival period\n- Weekend-night stays\n- Week-night stays\n- Adults, children, babies\n- Meal\n- Reserved room type\n- Parking requirements\n- Related target comparisons\n- Outliers or unusual observations\n- Feature-engineering opportunities such as `total_stay` and `total_guests`"),
        markdown_cell("### 2.1 Hotel Type Distribution and Cancellation Rate"),
        code_cell(
            "hotel_counts = df['hotel'].value_counts().reset_index()\n"
            "hotel_counts.columns = ['hotel', 'count']\n"
            "hotel_counts['percentage'] = (hotel_counts['count'] / len(df) * 100).round(2)\n"
            "hotel_counts"
        ),
        code_cell(
            "plt.figure(figsize=(7, 4))\n"
            "sns.countplot(data=df, x='hotel', hue='is_canceled')\n"
            "plt.title('Hotel Type by Cancellation Status')\n"
            "plt.xlabel('Hotel Type')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('hotel_type_by_cancellation.png')"
        ),
        code_cell(
            "hotel_cancel_rate = cancellation_rate_table(df, 'hotel')\n"
            "hotel_cancel_rate"
        ),
        code_cell(
            "highest_hotel = hotel_cancel_rate.iloc[0]\n"
            "lowest_hotel = hotel_cancel_rate.iloc[-1]\n"
            "rate_gap = highest_hotel['cancellation_rate'] - lowest_hotel['cancellation_rate']\n"
            "print(f\"Observed result: {highest_hotel['hotel']} has the higher cancellation rate \"\n"
            "      f\"({highest_hotel['cancellation_rate']:.2f}%) compared with {lowest_hotel['hotel']} \"\n"
            "      f\"({lowest_hotel['cancellation_rate']:.2f}%).\")\n"
            "print(f\"Difference: approximately {rate_gap:.2f} percentage points.\")"
        ),
        markdown_cell("**Finding:** The hotel-type comparison should be interpreted using the observed cancellation-rate table above. The hotel category with the higher cancellation rate may capture different booking behaviour, customer segments, or reservation conditions. This variable should be retained as a categorical predictor and encoded during preprocessing."),
        markdown_cell("### 2.2 Arrival Period Analysis\n\nThis section focuses on arrival month and arrival year because they are the most meaningful period-level variables for identifying broad seasonal and yearly cancellation patterns. Week number and day of month are not visualized here to avoid adding less interpretable plots without a clear analytical purpose."),
        code_cell(
            "month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']\n"
            "arrival_month_summary = cancellation_rate_table(df, 'arrival_date_month')\n"
            "arrival_month_summary['arrival_date_month'] = pd.Categorical(arrival_month_summary['arrival_date_month'], categories=month_order, ordered=True)\n"
            "arrival_month_summary.sort_values('arrival_date_month')"
        ),
        code_cell(
            "plt.figure(figsize=(12, 5))\n"
            "sns.countplot(data=df, x='arrival_date_month', hue='is_canceled', order=month_order)\n"
            "plt.title('Arrival Month by Cancellation Status')\n"
            "plt.xlabel('Arrival Month')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.xticks(rotation=45)\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('arrival_month_by_cancellation.png')"
        ),
        code_cell(
            "arrival_year_summary = cancellation_rate_table(df, 'arrival_date_year')\n"
            "arrival_year_summary"
        ),
        code_cell(
            "arrival_month_rates = arrival_month_summary.sort_values('cancellation_rate', ascending=False)\n"
            "arrival_year_rates = arrival_year_summary.sort_values('cancellation_rate', ascending=False)\n"
            "print(f\"Highest cancellation month: {arrival_month_rates.iloc[0]['arrival_date_month']} \"\n"
            "      f\"({arrival_month_rates.iloc[0]['cancellation_rate']:.2f}%).\")\n"
            "print(f\"Lowest cancellation month: {arrival_month_rates.iloc[-1]['arrival_date_month']} \"\n"
            "      f\"({arrival_month_rates.iloc[-1]['cancellation_rate']:.2f}%).\")\n"
            "print(f\"Highest cancellation year: {arrival_year_rates.iloc[0]['arrival_date_year']} \"\n"
            "      f\"({arrival_year_rates.iloc[0]['cancellation_rate']:.2f}%).\")"
        ),
        markdown_cell("**Finding:** Arrival month and year show period-level differences in cancellation rates. These observed differences suggest that arrival timing may contain useful seasonal or yearly information. During preprocessing/feature engineering, month and year should be handled as time-related features rather than ignored."),
        markdown_cell("### 2.3 Stay Duration Analysis\n\nThis section examines weekend-night stays and week-night stays. It also proposes `total_stay` as a future engineered feature."),
        code_cell(
            "stay_columns = ['stays_in_weekend_nights', 'stays_in_week_nights']\n"
            "df[stay_columns].describe().T.round(2)"
        ),
        code_cell(
            "plt.figure(figsize=(12, 4))\n"
            "for i, column in enumerate(stay_columns, start=1):\n"
            "    plt.subplot(1, 2, i)\n"
            "    sns.boxplot(data=df, x='is_canceled', y=column)\n"
            "    plt.title(f'{column} by Cancellation Status')\n"
            "    plt.xlabel('is_canceled')\n"
            "    plt.ylabel(column)\n"
            "save_current_figure('stay_nights_boxplots.png')"
        ),
        code_cell(
            "df_member2 = df.copy()\n"
            "df_member2['total_stay'] = df_member2['stays_in_weekend_nights'] + df_member2['stays_in_week_nights']\n"
            "total_stay_summary = df_member2['total_stay'].describe().to_frame('total_stay').T.round(2)\n"
            "zero_night_count = int((df_member2['total_stay'] == 0).sum())\n"
            "total_stay_summary['zero_night_bookings'] = zero_night_count\n"
            "total_stay_summary"
        ),
        code_cell(
            "plt.figure(figsize=(9, 4))\n"
            "sns.histplot(data=df_member2, x='total_stay', hue='is_canceled', bins=40, kde=False, multiple='stack')\n"
            "plt.title('Total Stay Distribution by Cancellation Status')\n"
            "plt.xlabel('Total Stay Nights')\n"
            "plt.ylabel('Number of Bookings')\n"
            "save_current_figure('total_stay_distribution.png')"
        ),
        code_cell(
            "df_member2['stay_length_group'] = pd.cut(\n"
            "    df_member2['total_stay'],\n"
            "    bins=[-1, 0, 2, 7, np.inf],\n"
            "    labels=['Zero-night stay', 'Short stay (1-2 nights)', 'Medium stay (3-7 nights)', 'Long stay (8+ nights)']\n"
            ")\n"
            "stay_length_cancel_rate = cancellation_rate_table(df_member2, 'stay_length_group')\n"
            "stay_length_cancel_rate.sort_values('stay_length_group')"
        ),
        code_cell(
            "stay_rates = stay_length_cancel_rate.sort_values('cancellation_rate', ascending=False)\n"
            "print(f\"Highest cancellation stay group: {stay_rates.iloc[0]['stay_length_group']} \"\n"
            "      f\"({stay_rates.iloc[0]['cancellation_rate']:.2f}%).\")\n"
            "print(f\"Lowest cancellation stay group: {stay_rates.iloc[-1]['stay_length_group']} \"\n"
            "      f\"({stay_rates.iloc[-1]['cancellation_rate']:.2f}%).\")"
        ),
        markdown_cell("**Finding:** The binned stay-length comparison provides a clearer target comparison for `total_stay`. If cancellation rates differ across stay-length groups, `total_stay` may be useful as an engineered feature. Zero-night and very long-stay records should still be investigated during preprocessing rather than removed during EDA."),
        markdown_cell("### 2.4 Guest Composition Analysis\n\nThis section analyses adults, children, and babies. It also proposes `total_guests` as a possible engineered feature."),
        code_cell(
            "guest_columns = ['adults', 'children', 'babies']\n"
            "df[guest_columns].describe().T.round(2)"
        ),
        code_cell(
            "# EDA-only temporary treatment: children has 4 missing values.\n"
            "# For exploratory calculation of total_guests, missing children values are temporarily treated as 0.\n"
            "# Final missing-value treatment must be justified and performed during preprocessing.\n"
            "df_member2['children'] = df_member2['children'].fillna(0)\n"
            "df_member2['total_guests'] = df_member2['adults'] + df_member2['children'] + df_member2['babies']\n"
            "guest_summary = df_member2[['adults', 'children', 'babies', 'total_guests']].describe().T.round(2)\n"
            "guest_summary['zero_guest_bookings'] = [\n"
            "    int((df_member2['adults'] == 0).sum()),\n"
            "    int((df_member2['children'] == 0).sum()),\n"
            "    int((df_member2['babies'] == 0).sum()),\n"
            "    int((df_member2['total_guests'] == 0).sum())\n"
            "]\n"
            "guest_summary"
        ),
        code_cell(
            "plt.figure(figsize=(9, 4))\n"
            "sns.countplot(data=df_member2[df_member2['total_guests'] <= 6], x='total_guests', hue='is_canceled')\n"
            "plt.title('Total Guests by Cancellation Status')\n"
            "plt.xlabel('Total Guests')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('total_guests_by_cancellation.png')"
        ),
        code_cell(
            "df_member2['has_children'] = np.where(df_member2['children'] > 0, 'Has children', 'No children')\n"
            "df_member2['has_babies'] = np.where(df_member2['babies'] > 0, 'Has babies', 'No babies')\n"
            "children_babies_cancel_rate = pd.concat(\n"
            "    [\n"
            "        cancellation_rate_table(df_member2, 'has_children').assign(feature='children_presence'),\n"
            "        cancellation_rate_table(df_member2, 'has_babies').assign(feature='babies_presence'),\n"
            "    ],\n"
            "    ignore_index=True\n"
            ")\n"
            "children_babies_cancel_rate"
        ),
        code_cell(
            "children_rates = cancellation_rate_table(df_member2, 'has_children')\n"
            "babies_rates = cancellation_rate_table(df_member2, 'has_babies')\n"
            "print('Children presence cancellation rates:')\n"
            "print(children_rates[['has_children', 'cancellation_rate']].to_string(index=False))\n"
            "print('\\nBabies presence cancellation rates:')\n"
            "print(babies_rates[['has_babies', 'cancellation_rate']].to_string(index=False))"
        ),
        markdown_cell("**Finding:** `total_guests` is a useful feature-engineering opportunity, but individual guest-composition indicators are also informative. Comparing bookings with and without children/babies gives a clearer view of how family-related guest composition relates to cancellation. Records with zero total guests should be treated as unusual and investigated during preprocessing."),
        markdown_cell("### 2.5 Meal Type Analysis"),
        code_cell("df['meal'].value_counts(dropna=False).to_frame('count')"),
        code_cell(
            "plt.figure(figsize=(8, 4))\n"
            "sns.countplot(data=df, x='meal', hue='is_canceled')\n"
            "plt.title('Meal Type by Cancellation Status')\n"
            "plt.xlabel('Meal Type')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('meal_type_by_cancellation.png')"
        ),
        code_cell("cancellation_rate_table(df, 'meal')"),
        markdown_cell("**Finding:** Meal is a categorical feature and should be encoded during preprocessing. Undefined or rare categories should be reviewed before encoding."),
        markdown_cell("### 2.6 Reserved Room Type Analysis"),
        code_cell("df['reserved_room_type'].value_counts(dropna=False).to_frame('count')"),
        code_cell(
            "plt.figure(figsize=(10, 4))\n"
            "sns.countplot(data=df, x='reserved_room_type', hue='is_canceled', order=sorted(df['reserved_room_type'].unique()))\n"
            "plt.title('Reserved Room Type by Cancellation Status')\n"
            "plt.xlabel('Reserved Room Type')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('reserved_room_type_by_cancellation.png')"
        ),
        code_cell("cancellation_rate_table(df, 'reserved_room_type')"),
        markdown_cell("**Finding:** Reserved room type may reflect customer preference and booking conditions. It should be retained as a categorical feature, while rare room categories may require careful encoding."),
        markdown_cell("### 2.7 Parking Requirement Analysis"),
        code_cell("df['required_car_parking_spaces'].value_counts(dropna=False).sort_index().to_frame('count')"),
        code_cell(
            "plt.figure(figsize=(8, 4))\n"
            "sns.countplot(data=df, x='required_car_parking_spaces', hue='is_canceled')\n"
            "plt.title('Required Parking Spaces by Cancellation Status')\n"
            "plt.xlabel('Required Car Parking Spaces')\n"
            "plt.ylabel('Number of Bookings')\n"
            "plt.legend(title='is_canceled', labels=['Not canceled', 'Canceled'])\n"
            "save_current_figure('parking_spaces_by_cancellation.png')"
        ),
        code_cell(
            "parking_cancel_rate = cancellation_rate_table(df, 'required_car_parking_spaces')\n"
            "parking_cancel_rate"
        ),
        code_cell(
            "no_parking_rate = parking_cancel_rate.loc[parking_cancel_rate['required_car_parking_spaces'].eq(0), 'cancellation_rate'].iloc[0]\n"
            "parking_requested = df.assign(parking_requested=np.where(df['required_car_parking_spaces'] > 0, 'Parking requested', 'No parking requested'))\n"
            "parking_request_rate = cancellation_rate_table(parking_requested, 'parking_requested')\n"
            "print(parking_request_rate.to_string(index=False))\n"
            "print(f\"Cancellation rate for zero parking spaces: {no_parking_rate:.2f}%\")"
        ),
        markdown_cell("**Finding:** Parking should be interpreted from the observed cancellation-rate table above. If bookings with requested parking show a lower cancellation rate than bookings with no parking request, this supports the idea that parking requests may indicate stronger booking intention. The variable is discrete numerical and may be useful without complex transformation, while rare high values should be reviewed."),
        markdown_cell("### 2.8 Member 2 Outlier and Unusual Observation Summary"),
        code_cell(
            "member2_unusual = pd.DataFrame([\n"
            "    {'check': 'Zero total stay nights', 'count': int((df_member2['total_stay'] == 0).sum())},\n"
            "    {'check': 'Very long stays above 30 nights', 'count': int((df_member2['total_stay'] > 30).sum())},\n"
            "    {'check': 'Missing children values', 'count': int(df['children'].isna().sum())},\n"
            "    {'check': 'Zero total guests', 'count': int((df_member2['total_guests'] == 0).sum())},\n"
            "    {'check': 'Bookings with more than 6 guests', 'count': int((df_member2['total_guests'] > 6).sum())},\n"
            "    {'check': 'Undefined meal values', 'count': int((df['meal'] == 'Undefined').sum())},\n"
            "    {'check': 'Parking spaces greater than 2', 'count': int((df['required_car_parking_spaces'] > 2).sum())},\n"
            "])\n"
            "member2_unusual"
        ),
        markdown_cell("### 2.9 Member 2 Preprocessing and Feature-Engineering Implications\n\n- Encode categorical variables: `hotel`, `arrival_date_month`, `meal`, and `reserved_room_type`.\n- Consider engineering `total_stay = stays_in_weekend_nights + stays_in_week_nights`.\n- Consider engineering `total_guests = adults + children + babies`.\n- Investigate zero-night bookings and zero-guest bookings before cleaning.\n- Handle the small number of missing `children` values during preprocessing.\n- Review rare categories in `meal` and `reserved_room_type` before encoding.\n- Keep `required_car_parking_spaces` as a useful discrete numerical feature, while reviewing rare high values."),
        markdown_cell("## Member 3 - Commercial & Distribution Analysis (Placeholder)\n\nThis section is reserved for Member 3. Suggested variables include market segment, distribution channel, deposit type, agent/company, ADR, country, assigned room type, correlations, high-cardinality variables, missing-value patterns, and data-leakage variables such as `reservation_status` and `reservation_status_date`."),
        markdown_cell("## Overall EDA Conclusion and Recommendations (To Be Completed After All Members Finish)\n\nThis section should combine Member 1, Member 2, and Member 3 findings into one final EDA conclusion and preprocessing/feature-engineering recommendation section."),
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


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_member2_figures() -> None:
    df = pd.read_csv(DATA_PATH)
    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="hotel", hue="is_canceled")
    plt.title("Hotel Type by Cancellation Status")
    plt.xlabel("Hotel Type")
    plt.ylabel("Number of Bookings")
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "hotel_type_by_cancellation.png")

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="arrival_date_month", hue="is_canceled", order=month_order)
    plt.title("Arrival Month by Cancellation Status")
    plt.xlabel("Arrival Month")
    plt.ylabel("Number of Bookings")
    plt.xticks(rotation=45)
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "arrival_month_by_cancellation.png")

    stay_columns = ["stays_in_weekend_nights", "stays_in_week_nights"]
    plt.figure(figsize=(12, 4))
    for index, column in enumerate(stay_columns, start=1):
        plt.subplot(1, 2, index)
        sns.boxplot(data=df, x="is_canceled", y=column)
        plt.title(f"{column} by Cancellation Status")
        plt.xlabel("is_canceled")
        plt.ylabel(column)
    save_plot(FIGURES_DIR / "stay_nights_boxplots.png")

    df_member2 = df.copy()
    df_member2["total_stay"] = df_member2["stays_in_weekend_nights"] + df_member2["stays_in_week_nights"]
    df_member2["children"] = df_member2["children"].fillna(0)
    df_member2["total_guests"] = df_member2["adults"] + df_member2["children"] + df_member2["babies"]

    plt.figure(figsize=(9, 4))
    sns.histplot(data=df_member2, x="total_stay", hue="is_canceled", bins=40, kde=False, multiple="stack")
    plt.title("Total Stay Distribution by Cancellation Status")
    plt.xlabel("Total Stay Nights")
    plt.ylabel("Number of Bookings")
    save_plot(FIGURES_DIR / "total_stay_distribution.png")

    plt.figure(figsize=(9, 4))
    sns.countplot(data=df_member2[df_member2["total_guests"] <= 6], x="total_guests", hue="is_canceled")
    plt.title("Total Guests by Cancellation Status")
    plt.xlabel("Total Guests")
    plt.ylabel("Number of Bookings")
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "total_guests_by_cancellation.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="meal", hue="is_canceled")
    plt.title("Meal Type by Cancellation Status")
    plt.xlabel("Meal Type")
    plt.ylabel("Number of Bookings")
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "meal_type_by_cancellation.png")

    plt.figure(figsize=(10, 4))
    sns.countplot(
        data=df,
        x="reserved_room_type",
        hue="is_canceled",
        order=sorted(df["reserved_room_type"].unique()),
    )
    plt.title("Reserved Room Type by Cancellation Status")
    plt.xlabel("Reserved Room Type")
    plt.ylabel("Number of Bookings")
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "reserved_room_type_by_cancellation.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="required_car_parking_spaces", hue="is_canceled")
    plt.title("Required Parking Spaces by Cancellation Status")
    plt.xlabel("Required Car Parking Spaces")
    plt.ylabel("Number of Bookings")
    plt.legend(title="is_canceled", labels=["Not canceled", "Canceled"])
    save_plot(FIGURES_DIR / "parking_spaces_by_cancellation.png")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    notebook_path = EDA_DIR / "EDA.ipynb"
    if notebook_path.exists():
        existing = json.loads(notebook_path.read_text(encoding="utf-8"))

        def section_bounds(cells):
            start = next(i for i, cell in enumerate(cells)
                         if "".join(cell["source"]).startswith("## Member 2 -"))
            end = next(i for i in range(start + 1, len(cells))
                       if "".join(cells[i]["source"]).startswith("## Member 3 -"))
            return start, end

        start, end = section_bounds(existing["cells"])
        new_start, new_end = section_bounds(notebook["cells"])
        existing["cells"][start:end] = notebook["cells"][new_start:new_end]
        notebook = existing
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    generate_member2_figures()


if __name__ == "__main__":
    main()
