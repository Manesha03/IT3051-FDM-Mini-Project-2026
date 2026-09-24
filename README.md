# Hotel Booking Cancellation Prediction

IT3051 Fundamentals of Data Mining - Mini Project

## Project Overview

This project uses the lecturer-validated Hotel Booking dataset to study hotel booking cancellations. The final goal of the full project is to support cancellation prediction using historical booking information.

Current completed work:

- Dataset Selection & Understanding
- EDA - Member 1: Booking & Customer Behaviour Analysis
- EDA - Member 2: Stay & Reservation Analysis
- EDA - Member 3: Commercial & Distribution Analysis
- Preprocessing - Member 1: Data Cleaning & Quality Preparation

## Dataset

Dataset file:

```text
data/raw/hotel_booking.csv
```

Target variable:

```text
is_canceled
```

Target meaning:

```text
0 = Booking was not canceled
1 = Booking was canceled
```

## Repository Structure

```text
IT3051-FDM-Mini-Project-2026/
├── 01_Dataset_Understanding/
│   ├── Dataset_Selection_and_Understanding.ipynb
│   ├── data_dictionary.csv
│   └── dataset_characteristics_summary.csv
├── 02_EDA/
│   ├── EDA.ipynb
│   └── Figures/
│       ├── booking_customer/
│       ├── commercial_distribution/
│       └── stay_reservation/
├── 03_Data_Preprocessing_and_Feature_Engineering/
│   ├── Data_Preprocessing_and_Feature_Engineering.ipynb
│   ├── data_quality_report.csv
│   ├── preprocessing_summary.csv
│   ├── cleaning_validation.csv
│   ├── schema_report.csv
│   ├── outlier_review.csv
│   ├── removed_rows.csv
│   └── target_distribution_comparison.csv
├── data/
│   ├── raw/
│   │   └── hotel_booking.csv
│   └── processed/
│       └── hotel_booking_cleaned.csv
├── scripts/
│   ├── generate_dataset_understanding.py
│   ├── generate_eda.py
│   ├── clean_hotel_bookings.py
│   └── generate_preprocessing.py
├── tests/
│   └── test_data_cleaning.py
├── requirements.txt
└── README.md
```

## Main Deliverables

### Dataset Understanding

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

Includes:

- Problem definition
- Dataset source and description
- Dataset shape and structure
- Target variable explanation
- Data dictionary
- Numerical descriptive statistics
- Categorical value investigation
- Missing-value investigation
- Duplicate investigation
- Invalid/unusual value checks
- Initial class distribution
- Potential data-leakage discussion
- Dataset suitability and conclusion

Supporting files:

```text
01_Dataset_Understanding/data_dictionary.csv
01_Dataset_Understanding/dataset_characteristics_summary.csv
```

### EDA

```text
02_EDA/EDA.ipynb
```

Completed sections:

```text
Member 1 - Booking & Customer Behaviour Analysis
Member 2 - Stay & Reservation Analysis
Member 3 - Commercial & Distribution Analysis
```

Member 1's analysis and findings are in the notebook. Its figures are saved in
`02_EDA/Figures/booking_customer/`.

Member 2 analysis covers:

- Hotel type
- Arrival period
- Weekend-night stays
- Week-night stays
- Adults, children, and babies
- Meal type
- Reserved room type
- Parking requirements
- Related cancellation comparisons
- Outliers and unusual observations
- Feature-engineering opportunities such as `total_stay` and `total_guests`
- Preprocessing implications

Generated Member 2 figures:

```text
02_EDA/Figures/stay_reservation/
```

Member 3 analysis covers commercial, distribution, country, room-assignment, missing-value, correlation, and leakage-related variables.

Generated Member 3 figures:

```text
02_EDA/Figures/commercial_distribution/
```

### Preprocessing — Member 1 (Razna)

Open `03_Data_Preprocessing_and_Feature_Engineering/Data_Preprocessing_and_Feature_Engineering.ipynb`
for the executed cleaning workflow, EDA-based justifications, and before/after validation.

- Excludes 4 records with unknown child counts, 180 zero-guest records, and 1 negative-ADR record under the documented cleaning policies.
- Fills missing country with `Unknown` and missing agent/company IDs with `Not_recorded`.
- Corrects count, categorical ID, binary and date types; audits exact duplicates.
- Retains valid extreme values, zero-night stays and ambiguous categories instead of applying arbitrary clipping or recoding.
- Verifies retained target labels, numerical values, row accounting, raw-file integrity and exported CSV fidelity.

The cleaned handoff has **119,205 rows and 36 columns**, with no remaining nulls or
exact duplicate records. Before/after metrics are in `data_quality_report.csv`; decisions
are in `preprocessing_summary.csv`. `removed_rows.csv` gives zero-based source-row
indices and exclusion reasons (source CSV line = index + 2).

This is a cleaning-stage handoff. Member 2 still owns feature engineering, selection,
encoding and removal of leakage/PII columns; Member 3 owns train/test preparation,
train-only fitted transformations and final modelling validation. The cleaned CSV
retains all source columns and must not be used directly as the model's predictors.
Matching booking profiles remain for group/split review; they are not proven duplicate bookings.

CSV loses dtype metadata. To restore categories, dates and textual identifiers:

```python
from scripts.clean_hotel_bookings import load_cleaned
cleaned = load_cleaned()
```

## Setup

Run commands from the repository root folder.

Install required Python packages:

```powershell
python -m pip install -r requirements.txt
```

The requirements include the notebook kernel and execution tools.

## How to Run the Notebooks

Open a notebook in VS Code or Jupyter:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
02_EDA/EDA.ipynb
03_Data_Preprocessing_and_Feature_Engineering/Data_Preprocessing_and_Feature_Engineering.ipynb
```

Select a Python kernel and run all cells from top to bottom.

## How to Regenerate Outputs

Regenerate Dataset Understanding files:

```powershell
python scripts/generate_dataset_understanding.py
```

Run Member 1's cells in the notebook to regenerate its analysis and figures.

Regenerate the EDA notebook and figures:

```powershell
python scripts/generate_eda.py
```

Regenerate and execute the Member 1 preprocessing notebook, cleaned CSV and reports:

```powershell
python scripts/generate_preprocessing.py
```

Run the cleaning regression checks:

```powershell
python -m unittest discover -s tests -v
```

## Team Workflow Notes

- Do not use machine-specific absolute paths in notebooks or documentation.
- Keep dataset paths relative to the repository, such as `../data/raw/hotel_booking.csv`.
- The current EDA stage investigates possible feature-engineering ideas but does not perform final preprocessing or model training.

## Current Scope

This repository covers Dataset Understanding, all three EDA sections, and Member 1's
data cleaning and quality preparation. Feature engineering, feature selection, encoding,
train/test preparation, final ML validation, modelling, model evaluation and deployment
remain outside the completed scope.
