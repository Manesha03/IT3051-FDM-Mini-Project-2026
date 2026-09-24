# Hotel Booking Cancellation Prediction

IT3051 Fundamentals of Data Mining - Mini Project

## Project Overview

This project uses the lecturer-validated Hotel Booking dataset to study hotel booking cancellations. The final goal of the full project is to support cancellation prediction using historical booking information.

Current completed work:

- Dataset Selection & Understanding
- EDA - Member 1: Booking & Customer Behaviour Analysis
- EDA - Member 2: Stay & Reservation Analysis
- EDA - Member 3: Commercial & Distribution Analysis

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
├── data/
│   └── raw/
│       └── hotel_booking.csv
├── scripts/
│   ├── generate_dataset_understanding.py
│   └── generate_eda.py
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

## Setup

Run commands from the repository root folder.

Install required Python packages:

```powershell
pip install pandas numpy matplotlib seaborn
```

If running notebooks in VS Code, also install:

```powershell
pip install ipykernel
```

## How to Run the Notebooks

Open either notebook in VS Code or Jupyter:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
02_EDA/EDA.ipynb
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

## Team Workflow Notes

- Do not use machine-specific absolute paths in notebooks or documentation.
- Keep dataset paths relative to the repository, such as `../data/raw/hotel_booking.csv`.
- The current EDA stage investigates possible feature-engineering ideas but does not perform final preprocessing or model training.

## Current Scope

This repository currently covers Dataset Understanding and Member 1, Member 2, and Member 3 EDA. Final preprocessing, feature engineering, feature selection, modelling, model evaluation, and deployment are outside the current completed scope.
