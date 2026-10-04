# Hotel Booking Cancellation Prediction

IT3051 Fundamentals of Data Mining - Mini Project

## Project Overview

This project uses the lecturer-validated Hotel Booking dataset to study and prepare data for hotel booking cancellation prediction. The target is to predict whether a booking will be cancelled using historical booking details available before the final reservation outcome.

Completed work:

- Dataset selection and understanding
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and feature selection
- ML data preparation and final validation
- Model training and evaluation
- Final model comparison and selection

Remaining project work:

- Final report/presentation preparation

## Dataset

Raw dataset is stored in the raw data folder.

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

Feature-engineered dataset is stored in the processed data folder.

## Repository Structure

```text
IT3051-FDM-Mini-Project-2026/
|-- 01_Dataset_Understanding/
|   |-- Dataset_Selection_and_Understanding.ipynb
|   `-- generated summary tables
|-- 02_EDA/
|   |-- EDA.ipynb
|   `-- Figures/
|       |-- booking_customer/
|       |-- commercial_distribution/
|       `-- stay_reservation/
|-- 03_Data_Preprocessing/
|   |-- Data_Preprocessing.ipynb
|   `-- generated cleaning reports
|-- 04_Feature_Engineering/
|   |-- Feature_Engineering_and_Selection.ipynb
|   `-- generated feature reports
|-- 05_ML_Data_Preparation/
|   |-- ML_Data_Preparation_and_Final_Validation.ipynb
|   `-- generated validation report
|-- 06_Model_Evaluation/
|   |-- Shared_Model_Setup.ipynb
|   |-- Logistic_Regression.ipynb
|   |-- Decision_Tree.ipynb
|   |-- Random_Forest.ipynb
|   |-- Gradient_Boosting.ipynb
|   `-- Overall_Model_Comparison_and_Final_Selection.ipynb
|-- data/
|   |-- raw/
|   |   `-- raw dataset
|   `-- processed/
|       `-- processed datasets
|-- scripts/
|   |-- clean_hotel_bookings.py
|   |-- feature_engineering_hotel_bookings.py
|   |-- generate_dataset_understanding.py
|   |-- generate_eda.py
|   |-- generate_feature_engineering.py
|   |-- generate_model_evaluation.py
|   |-- generate_preprocessing.py
|   |-- model_evaluation_utils.py
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

Supporting summary tables are generated in the dataset-understanding folder.

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

Output dataset is saved in the processed data folder.

Main cleaning decisions:

- Removed 4 records with missing child counts.
- Removed 180 bookings with zero total guests.
- Removed 1 record with negative ADR.
- Filled missing `country` values with `Unknown`.
- Filled missing `agent` and `company` values with `Not_recorded`.
- Corrected count, categorical ID, binary and date-related data types.
- Retained valid extreme values and unusual observations when there was no strong evidence of error.
- Preserved the target labels for retained rows.

Generated cleaning reports are saved in the preprocessing folder.

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

Output dataset is saved in the processed data folder.

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

Generated feature reports are saved in the feature-engineering folder.

### 5. ML Data Preparation and Final Validation

Notebook:

```text
05_ML_Data_Preparation/ML_Data_Preparation_and_Final_Validation.ipynb
```

Input dataset is loaded from the processed data folder.

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

The generated validation report is saved in the ML data preparation folder.

### 6. Model Development and Evaluation

Shared setup notebook:

```text
06_Model_Evaluation/Shared_Model_Setup.ipynb
```

Model notebooks:

```text
06_Model_Evaluation/Logistic_Regression.ipynb
06_Model_Evaluation/Decision_Tree.ipynb
06_Model_Evaluation/Random_Forest.ipynb
06_Model_Evaluation/Gradient_Boosting.ipynb
06_Model_Evaluation/Overall_Model_Comparison_and_Final_Selection.ipynb
```

Shared utility script:

```text
scripts/model_evaluation_utils.py
```

Completed modelling setup includes:

- Shared train/test split from the feature-engineered dataset
- Shared preprocessing pipeline
- Shared `random_state`
- Stratified K-Fold cross-validation
- Common metrics: accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC
- Logistic Regression baseline, class-weighted model and coefficient interpretation notebook
- Decision Tree baseline and tuning notebook
- Random Forest baseline, tuning, feature-importance and feature-ablation notebook
- Gradient Boosting baseline and tuning notebook
- Overall model comparison and final model selection notebook

Final selected model:

```text
Tuned Random Forest
```

Final selection basis:

- Highest cross-validation F1-score among the compared models.
- Strong held-out test performance across F1-score, ROC-AUC and PR-AUC.
- Improved recall compared with the baseline Random Forest while keeping accuracy stable.
- Feature ablation showed that removing selected calendar fields did not improve the model, so the full selected feature set was retained.

## Current Status

The project has completed dataset preparation, model training, model evaluation, model optimization and final model selection. The remaining work is to prepare the final report and presentation using the generated notebooks, figures and comparison tables.

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

Regenerate the shared setup and model notebooks:

```powershell
python scripts/generate_model_evaluation.py
```

Regenerate and execute the model notebooks and figures:

```powershell
python scripts/generate_model_evaluation.py --execute
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
    if (path / "data" / "raw").is_dir()
)

DATA_PATH = ROOT / "data" / "raw" / "<dataset-file>"
```

## Notes

The current repository contains an older script, `scripts/preprocess_hotel_bookings.py`, and related tests that were written for a previous preprocessing-only workflow. The current staged workflow uses the feature-engineered dataset as the handoff to ML data preparation, so these older preprocessing utilities should be reviewed before final submission.
