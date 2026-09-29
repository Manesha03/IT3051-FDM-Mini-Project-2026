# Hotel Booking Cancellation Prediction

IT3051 Fundamentals of Data Mining - Mini Project

## Project Overview

This project uses the lecturer-validated Hotel Booking dataset to study and prepare data for hotel booking cancellation prediction. The target is to predict whether a booking will be cancelled using historical booking details available before the final reservation outcome.

Completed work so far:

- Dataset selection and understanding
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and feature selection
- ML data preparation and final validation

Remaining project work:

- Model training
- Model evaluation and comparison
- Final model selection
- Final report/presentation preparation

## Dataset

Raw dataset:

```text
data/raw/hotel_booking.csv
```

Target variable:

```text
is_canceled
```

Target meaning:

```text
0 = Booking was not cancelled
1 = Booking was cancelled
```

Original dataset size:

```text
119,390 rows x 36 columns
```

Cleaned dataset size:

```text
119,205 rows x 36 columns
```

Feature-engineered dataset:

```text
data/processed/hotel_booking_features.csv
```

## Repository Structure

```text
IT3051-FDM-Mini-Project-2026/
|-- 01_Dataset_Understanding/
|   |-- Dataset_Selection_and_Understanding.ipynb
|   |-- data_dictionary.csv
|   `-- dataset_characteristics_summary.csv
|-- 02_EDA/
|   |-- EDA.ipynb
|   `-- Figures/
|       |-- booking_customer/
|       |-- commercial_distribution/
|       `-- stay_reservation/
|-- 03_Data_Preprocessing/
|   |-- Data_Preprocessing.ipynb
|   |-- cleaning_validation.csv
|   |-- data_quality_report.csv
|   |-- outlier_review.csv
|   |-- preprocessing_summary.csv
|   |-- removed_rows.csv
|   |-- schema_report.csv
|   `-- target_distribution_comparison.csv
|-- 04_Feature_Engineering/
|   |-- Feature_Engineering_and_Selection.ipynb
|   |-- encoding_strategy.csv
|   |-- engineered_feature_evaluation.csv
|   |-- feature_engineering_validation.csv
|   |-- feature_selection_decisions.csv
|   |-- feature_type_review.csv
|   `-- final_selected_features.csv
|-- 05_ML_Data_Preparation/
|   |-- ML_Data_Preparation_and_Final_Validation.ipynb
|   `-- data_quality_report.csv
|-- data/
|   |-- raw/
|   |   `-- hotel_booking.csv
|   `-- processed/
|       |-- hotel_booking_cleaned.csv
|       `-- hotel_booking_features.csv
|-- scripts/
|   |-- clean_hotel_bookings.py
|   |-- feature_engineering_hotel_bookings.py
|   |-- generate_dataset_understanding.py
|   |-- generate_eda.py
|   |-- generate_feature_engineering.py
|   |-- generate_preprocessing.py
|   `-- preprocess_hotel_bookings.py
|-- tests/
|   |-- test_data_cleaning.py
|   `-- test_preprocessing.py
|-- requirements.txt
`-- README.md
```

## Main Deliverables

### 1. Dataset Understanding

Notebook:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

This section covers:

- Problem definition
- Dataset source and description
- Dataset shape and structure
- Target variable explanation
- Data dictionary
- Numerical and categorical feature review
- Missing-value investigation
- Duplicate investigation
- Invalid and unusual value checks
- Initial class distribution
- Data-leakage discussion
- Dataset suitability conclusion

Supporting files:

```text
01_Dataset_Understanding/data_dictionary.csv
01_Dataset_Understanding/dataset_characteristics_summary.csv
```

### 2. Exploratory Data Analysis

Notebook:

```text
02_EDA/EDA.ipynb
```

Completed EDA areas:

- Member 1: Booking and customer behaviour analysis
- Member 2: Stay and reservation analysis
- Member 3: Commercial and distribution analysis

Generated figures are saved under:

```text
02_EDA/Figures/booking_customer/
02_EDA/Figures/stay_reservation/
02_EDA/Figures/commercial_distribution/
```

EDA findings were used to guide the cleaning rules, feature-engineering choices, leakage review and later ML preparation.

### 3. Data Preprocessing

Notebook:

```text
03_Data_Preprocessing/Data_Preprocessing.ipynb
```

Script:

```text
scripts/clean_hotel_bookings.py
```

Output dataset:

```text
data/processed/hotel_booking_cleaned.csv
```

Main cleaning decisions:

- Removed 4 records with missing child counts.
- Removed 180 bookings with zero total guests.
- Removed 1 record with negative ADR.
- Filled missing `country` values with `Unknown`.
- Filled missing `agent` and `company` values with `Not_recorded`.
- Corrected count, categorical ID, binary and date-related data types.
- Retained valid extreme values and unusual observations when there was no strong evidence of error.
- Preserved the target labels for retained rows.

Generated reports:

```text
03_Data_Preprocessing/data_quality_report.csv
03_Data_Preprocessing/preprocessing_summary.csv
03_Data_Preprocessing/cleaning_validation.csv
03_Data_Preprocessing/schema_report.csv
03_Data_Preprocessing/outlier_review.csv
03_Data_Preprocessing/removed_rows.csv
03_Data_Preprocessing/target_distribution_comparison.csv
```

### 4. Feature Engineering and Selection

Notebook:

```text
04_Feature_Engineering/Feature_Engineering_and_Selection.ipynb
```

Scripts:

```text
scripts/feature_engineering_hotel_bookings.py
scripts/generate_feature_engineering.py
```

Output dataset:

```text
data/processed/hotel_booking_features.csv
```

Engineered features:

- `total_stay`
- `total_guests`
- `has_children`
- `has_babies`
- `previous_total_bookings`
- `previous_cancellation_rate`

Feature-selection decisions:

- Kept useful original booking predictors.
- Kept engineered candidate predictors.
- Removed direct leakage fields such as `reservation_status` and `reservation_status_date`.
- Removed personal identifier fields such as name, email, phone number and credit card.
- Removed fields likely unavailable or unstable at initial booking time, such as `assigned_room_type`, `booking_changes` and `days_in_waiting_list`.

Final selected predictor count:

```text
32 predictors before encoding
```

Generated reports:

```text
04_Feature_Engineering/feature_type_review.csv
04_Feature_Engineering/engineered_feature_evaluation.csv
04_Feature_Engineering/feature_selection_decisions.csv
04_Feature_Engineering/final_selected_features.csv
04_Feature_Engineering/encoding_strategy.csv
04_Feature_Engineering/feature_engineering_validation.csv
```

### 5. ML Data Preparation and Final Validation

Notebook:

```text
05_ML_Data_Preparation/ML_Data_Preparation_and_Final_Validation.ipynb
```

Input dataset:

```text
data/processed/hotel_booking_features.csv
```

Current ML preparation summary:

```text
Input rows: 119,205
Selected predictors: 32
Training rows: 95,364
Test rows: 23,841
Transformed features: 116
```

Class distribution:

```text
Training not cancelled: 60,008
Training cancelled: 35,356
Test not cancelled: 15,002
Test cancelled: 8,839
```

Validation checks confirmed:

- No overlapping train/test rows.
- No non-finite transformed values.
- No excluded leakage fields among predictors.
- Train/test split and transformed feature counts were recorded.

Generated report:

```text
05_ML_Data_Preparation/data_quality_report.csv
```

## Current Status

The project is currently ready for model training and evaluation. The cleaned and feature-engineered datasets are available, and ML data preparation has been documented and validated.

The next stage should train and compare suitable classification models, such as:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbours
- Naive Bayes
- Support Vector Machine

Model performance should be evaluated using suitable classification metrics, including accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC where appropriate.

## Setup

Run all commands from the repository root folder.

Install required Python packages:

```powershell
python -m pip install -r requirements.txt
```

Required packages include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib
- nbformat
- nbclient
- ipykernel

## How to Run the Notebooks

Open these notebooks in VS Code or Jupyter and run all cells from top to bottom:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
02_EDA/EDA.ipynb
03_Data_Preprocessing/Data_Preprocessing.ipynb
04_Feature_Engineering/Feature_Engineering_and_Selection.ipynb
05_ML_Data_Preparation/ML_Data_Preparation_and_Final_Validation.ipynb
```

Use a Python environment with the packages from `requirements.txt` installed.

## How to Regenerate Outputs

Regenerate dataset-understanding files:

```powershell
python scripts/generate_dataset_understanding.py
```

Regenerate the EDA notebook and figures:

```powershell
python scripts/generate_eda.py
```

Regenerate the preprocessing notebook, cleaned dataset and cleaning reports:

```powershell
python scripts/generate_preprocessing.py
```

Regenerate the feature-engineering notebook, feature dataset and feature reports:

```powershell
python scripts/generate_feature_engineering.py
```

Run available regression checks:

```powershell
python -m unittest discover -s tests -v
```

## Team Workflow Notes

- Do not use machine-specific absolute paths in notebooks or scripts.
- Keep file paths relative to the project root.
- Use `Path` from Python's standard library when building paths.
- Keep raw data unchanged in `data/raw/`.
- Save processed outputs in `data/processed/`.
- Re-run notebooks before final submission so outputs match the latest code.
- Clear or avoid notebook outputs that show personal laptop paths.

Example portable path pattern:

```python
from pathlib import Path

ROOT = next(
    path
    for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "data/raw/hotel_booking.csv").is_file()
)

DATA_PATH = ROOT / "data" / "raw" / "hotel_booking.csv"
```

## Notes

The current repository contains an older script, `scripts/preprocess_hotel_bookings.py`, and related tests that were written for a previous preprocessing-only workflow. The current staged workflow uses `hotel_booking_features.csv` as the handoff to ML data preparation, so these older preprocessing utilities should be reviewed before final submission.
