# Hotel Booking Cancellation Prediction

IT3051 Fundamentals of Data Mining - Mini Project

## Overview

This repository documents the **Dataset Selection & Understanding** stage for the Hotel Booking Cancellation Prediction project.

The dataset is used to understand hotel reservation patterns and prepare for later EDA, preprocessing, feature engineering, and modelling work. The prediction target is:

```text
is_canceled
```

Target meaning:

```text
0 = Booking was not canceled
1 = Booking was canceled
```

## Current Scope

This section focuses only on Dataset Selection & Understanding. It does not perform final preprocessing, feature engineering, model training, model evaluation, or deployment.

Covered in the notebook:

| Area | Included |
| --- | --- |
| Problem understanding | Real-world hotel cancellation problem |
| Prediction objective | Binary cancellation prediction |
| Stakeholders | Hotel managers, reservation staff, revenue managers, operations teams |
| Decision support | Occupancy, revenue, staffing, follow-up, overbooking decisions |
| Expected input/output | Booking details as input, cancellation prediction as output |
| Dataset structure | Records, columns, feature types, target variable |
| Data dictionary | Attribute meanings, types, missingness, notes |
| Initial statistics | Numerical descriptive statistics |
| Categorical review | Unique values and category distributions |
| Data quality | Missing values, duplicates, unusual values |
| Leakage review | Reservation-status leakage and assigned-room investigation |
| PII review | Name, email, phone number, credit card handling |
| Limitations | Missing values, high-cardinality fields, timing concerns |
| Suitability | Why the dataset fits the prediction problem |

## Repository Structure

| Path | Purpose |
| --- | --- |
| `01_Dataset_Understanding/` | Dataset understanding deliverables |
| `01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb` | Main dataset understanding notebook |
| `01_Dataset_Understanding/data_dictionary.csv` | Attribute-level data dictionary |
| `01_Dataset_Understanding/dataset_characteristics_summary.csv` | Compact dataset summary |
| `data/raw/hotel_booking.csv` | Approved raw dataset |
| `scripts/generate_dataset_understanding.py` | Script to regenerate notebook and summary files |
| `README.md` | Project guide and run instructions |

## Main Deliverables

### Dataset Understanding Notebook

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

Main notebook for the Dataset Selection & Understanding stage.

It includes:

- Problem definition
- Prediction objective
- Stakeholders and decision support
- Expected system inputs and outputs
- Dataset source and citation
- Dataset shape and general structure
- Target variable explanation
- Feature and attribute type summary
- Data dictionary
- Numerical descriptive statistics
- Categorical value investigation
- Missing-value investigation
- Duplicate-record investigation
- Invalid/unusual value checks
- Initial class distribution
- Initial data-quality observations
- Potential data-leakage discussion
- Identifier and PII attribute discussion
- Dataset limitations and considerations
- Dataset suitability and conclusion

### Data Dictionary

```text
01_Dataset_Understanding/data_dictionary.csv
```

Contains all 36 dataset attributes with:

- Attribute name
- Description
- Attribute type
- Pandas data type
- Missing-value count and percentage
- Unique-value count
- Example values
- Notes for modelling, leakage, PII, or preprocessing relevance

### Dataset Characteristics Summary

```text
01_Dataset_Understanding/dataset_characteristics_summary.csv
```

Provides a compact summary of the dataset, including:

- Dataset file
- Dataset status
- Original source
- Period covered
- Problem type
- Prediction objective
- Target variable
- Record and column counts
- Duplicate count
- Missing-value column count
- Feature-type counts

### Raw Dataset

```text
data/raw/hotel_booking.csv
```

Approved raw dataset used for this project.

## Setup

Open a terminal from the repository root folder after cloning/downloading the project.

Install the required Python packages:

```powershell
pip install pandas numpy
```

If running the notebook in VS Code or Jupyter, also install:

```powershell
pip install ipykernel
```

## How to Run the Notebook

Open:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

Then:

1. Select a Python kernel.
2. Run all cells from top to bottom.
3. Save the notebook if outputs need to appear on GitHub.

The notebook reads the dataset from:

```text
data/raw/hotel_booking.csv
```

## Regenerating Files

To regenerate the Dataset Understanding notebook and summary CSV files, run this from the repository root:

```powershell
python scripts/generate_dataset_understanding.py
```

## Team Notes

- Use relative project paths instead of machine-specific absolute paths.
- Keep the approved raw dataset in `data/raw/hotel_booking.csv`.
- Do not edit generated outputs manually unless the generator script is also updated.
- Run notebooks from top to bottom before submission if GitHub-visible outputs are required.
- Later stages such as EDA, preprocessing, feature engineering, modelling, and evaluation should be added in separate folders/sections.
