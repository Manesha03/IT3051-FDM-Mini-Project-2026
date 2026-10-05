# Hotel Booking Cancellation Prediction System

IT3051 Fundamentals of Data Mining - Mini Project

This project builds an end-to-end hotel booking cancellation prediction system. It covers the full data mining workflow from dataset understanding and preprocessing to model evaluation, backend deployment, and a user-friendly dashboard for hotel staff.

## Project Snapshot

| Area | Status | Output |
|---|---:|---|
| Dataset understanding | Complete | Data dictionary, quality review, suitability analysis |
| EDA | Complete | Booking, stay, customer, commercial and distribution insights |
| Data preprocessing | Complete | Cleaned dataset and validation reports |
| Feature engineering | Complete | Final selected predictor set and engineered features |
| ML preparation | Complete | Shared train/test split and preprocessing pipeline |
| Model evaluation | Complete | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Final model | Complete | Tuned Random Forest pipeline |
| Backend service | Complete | Prediction API and dashboard API |
| Frontend dashboard | Complete | Staff-facing dashboard, live prediction form and recommendations |

## Problem Statement

Hotels lose revenue and operational efficiency when bookings are cancelled unexpectedly. This system estimates whether a reservation is likely to be cancelled using booking information available before the final reservation outcome.

Target variable:

```text
is_canceled
```

Target meaning:

```text
0 = Booking was not cancelled
1 = Booking was cancelled
```

## Dataset Summary

| Dataset version | Shape |
|---|---:|
| Raw dataset | 119,390 rows x 36 columns |
| Cleaned dataset | 119,205 rows x 36 columns |
| Final modelling dataset | 119,205 rows with 32 selected predictors plus target |

Raw data is stored in `data/raw/`, and processed datasets are stored in `data/processed/`.

## Final Model

The selected production model is:

```text
Tuned Random Forest
```

Selection basis:

- Highest cross-validation F1-score among compared models.
- Strong held-out test performance across F1-score, ROC-AUC and PR-AUC.
- Improved recall compared with the baseline Random Forest while keeping accuracy stable.
- Feature ablation did not improve performance, so the full selected predictor set was retained.

Saved model artifacts:

```text
06_Model_Evaluation/Models/tuned_random_forest_pipeline.joblib
06_Model_Evaluation/Models/tuned_random_forest_metadata.json
```

The saved pipeline includes the trained model and preprocessing pipeline, so backend predictions use the same transformations as model development.

## End-to-End Application

The application is split into:

```text
backend/
frontend/
```

The backend serves both the API and the frontend dashboard.

### Dashboard Features

- Dashboard KPIs for total bookings, cancellation rate, completed/cancelled bookings and average booking behaviour.
- Visual cancellation insights by hotel type, customer type, arrival month, market segment, deposit type, lead time and country.
- Two-step booking prediction form:
  - Step 1: Reservation, stay and guest details
  - Step 2: Booking details, history and price
- Live cancellation prediction result with probability and staff-friendly follow-up advice.
- Recommendations page with priority actions and daily workflow guidance.
- System Details page for model and dataset summary.

### Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Confirms the service and model are available |
| `GET` | `/api/schema` | Returns predictor columns, dropdown options and validation ranges |
| `GET` | `/api/dashboard` | Returns KPI, chart, recommendation and model-summary data |
| `POST` | `/api/predict` | Accepts booking details and returns cancellation prediction |

Prediction response includes predicted class, staff-friendly label, cancellation probability, model name/version and final feature values used by the model.

## Quick Start

Run all commands from the repository root.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the backend and dashboard:

```powershell
.\.venv\Scripts\python.exe backend\app.py 8001
```

Open the dashboard:

```text
http://127.0.0.1:8001
```

If an older version appears in the browser, hard refresh:

```text
Ctrl + F5
```

## Repository Structure

```text
IT3051-FDM-Mini-Project-2026/
|-- 01_Dataset_Understanding/
|-- 02_EDA/
|   `-- Figures/
|-- 03_Data_Preprocessing/
|-- 04_Feature_Engineering/
|-- 05_ML_Data_Preparation/
|-- 06_Model_Evaluation/
|   |-- Figures/
|   `-- Models/
|       |-- tuned_random_forest_pipeline.joblib
|       `-- tuned_random_forest_metadata.json
|-- backend/
|   `-- app.py
|-- frontend/
|   |-- assets/
|   |-- index.html
|   |-- app.js
|   `-- styles.css
|-- data/
|   |-- raw/
|   `-- processed/
|-- scripts/
|-- tests/
|-- requirements.txt
`-- README.md
```

## Workflow Deliverables

### 1. Dataset Understanding

Notebook:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

Covers problem definition, dataset source, structure, target explanation, data dictionary, missing values, duplicates, invalid values, unusual observations, class distribution, leakage review and suitability conclusion.

### 2. Exploratory Data Analysis

Notebook:

```text
02_EDA/EDA.ipynb
```

EDA areas:

- Booking and customer behaviour
- Stay and reservation patterns
- Commercial and distribution patterns

Generated figures are stored under:

```text
02_EDA/Figures/
```

### 3. Data Preprocessing

Notebook:

```text
03_Data_Preprocessing/Data_Preprocessing.ipynb
```

Script:

```text
scripts/clean_hotel_bookings.py
```

Main cleaning decisions:

- Removed records with missing child counts.
- Removed zero-guest bookings.
- Removed negative ADR record.
- Filled missing `country`, `agent` and `company` values.
- Corrected count, categorical ID, binary and date-related data types.
- Retained valid extreme values where there was no strong evidence of error.

### 4. Feature Engineering and Selection

Notebook:

```text
04_Feature_Engineering/Feature_Engineering_and_Selection.ipynb
```

Engineered features:

- `total_stay`
- `total_guests`
- `has_children`
- `has_babies`
- `previous_total_bookings`
- `previous_cancellation_rate`

Feature-selection decisions:

- Retained useful original booking predictors.
- Retained engineered candidate predictors.
- Removed direct leakage fields such as `reservation_status` and `reservation_status_date`.
- Removed personal identifier fields.
- Removed fields likely unavailable or unstable at initial booking time.

### 5. ML Data Preparation

Notebook:

```text
05_ML_Data_Preparation/ML_Data_Preparation_and_Final_Validation.ipynb
```

Preparation summary:

```text
Input rows: 119,205
Selected predictors: 32
Training rows: 95,364
Test rows: 23,841
Transformed features: 116
```

Validation confirmed no overlapping train/test rows, no non-finite transformed values and no excluded leakage fields among predictors.

### 6. Model Evaluation

Notebooks:

```text
06_Model_Evaluation/Shared_Model_Setup.ipynb
06_Model_Evaluation/Logistic_Regression.ipynb
06_Model_Evaluation/Decision_Tree.ipynb
06_Model_Evaluation/Random_Forest.ipynb
06_Model_Evaluation/Gradient_Boosting.ipynb
06_Model_Evaluation/Overall_Model_Comparison_and_Final_Selection.ipynb
```

Common modelling setup:

- Shared train/test split
- Shared preprocessing pipeline
- Stratified K-Fold cross-validation
- Common metrics: accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC
- Hyperparameter tuning and final comparison

### 7. Backend and Frontend

Backend:

```text
backend/app.py
```

Frontend:

```text
frontend/index.html
frontend/app.js
frontend/styles.css
```

Stage 9 backend requirements covered:

- Loads final trained model and metadata.
- Accepts and validates user input.
- Computes required engineered features.
- Applies the saved preprocessing pipeline.
- Generates model predictions.
- Returns meaningful JSON responses.
- Handles invalid and missing inputs.

Stage 10 frontend requirements covered:

- Provides clear booking input fields.
- Uses a two-step form for better usability.
- Validates user inputs.
- Presents prediction results and probabilities clearly.
- Provides dashboard visualizations and recommendations.
- Works end to end with the backend API.

## Regenerating Project Outputs

Regenerate dataset-understanding outputs:

```powershell
python scripts/generate_dataset_understanding.py
```

Regenerate EDA notebook and figures:

```powershell
python scripts/generate_eda.py
```

Regenerate preprocessing outputs:

```powershell
python scripts/generate_preprocessing.py
```

Regenerate feature-engineering outputs:

```powershell
python scripts/generate_feature_engineering.py
```

Regenerate model-evaluation notebooks:

```powershell
python scripts/generate_model_evaluation.py
```

Regenerate and execute model notebooks:

```powershell
python scripts/generate_model_evaluation.py --execute
```

Run regression checks:

```powershell
python -m unittest discover -s tests -v
```

## Team Workflow Notes

- Keep file paths relative to the project root.
- Do not commit machine-specific absolute paths in notebooks or scripts.
- Keep raw data unchanged in `data/raw/`.
- Save processed outputs in `data/processed/`.
- Re-run notebooks before final submission so outputs match the latest scripts.
- Avoid notebook outputs that expose personal laptop paths.

## Environment Note

The saved model metadata records scikit-learn `1.9.0`. `requirements.txt` is aligned to that version to avoid model persistence warnings when loading the saved pipeline.
