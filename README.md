# Hotel Booking Cancellation Prediction

## Evaluation 1 - Dataset Selection & Understanding

This project currently contains only the Dataset Selection & Understanding stage for the IT3051 FDM Mini Project. The purpose of this stage is to understand the lecturer-validated Hotel Booking dataset before EDA, preprocessing, feature engineering, or modeling.

## Folder Structure

```text
IT3051-FDM-Mini-Project-2026/
+-- 01_Dataset_Understanding/
|   +-- Dataset_Selection_and_Understanding.ipynb
|   +-- data_dictionary.csv
|   +-- dataset_characteristics_summary.csv
+-- data/
|   +-- raw/
|       +-- hotel_booking.csv
+-- scripts/
|   +-- generate_dataset_understanding.py
+-- README.md
```

## Main Files

- `01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb`  
  Main notebook for Dataset Selection & Understanding. It includes the problem definition, dataset source, dataset shape, target variable, data dictionary, numerical statistics, categorical investigation, missing values, duplicates, unusual values, class distribution, potential leakage, suitability, and conclusion.

- `01_Dataset_Understanding/data_dictionary.csv`  
  Attribute-level data dictionary for all 36 columns.

- `01_Dataset_Understanding/dataset_characteristics_summary.csv`  
  Short dataset summary with record count, column count, target variable, duplicate count, missing-value column count, and feature-type counts.

- `data/raw/hotel_booking.csv`  
  Preserved copy of the approved raw dataset.

- `scripts/generate_dataset_understanding.py`  
  Script used to regenerate the notebook and CSV summary files from the approved dataset.

## How to Run

Open PowerShell in the project root:

```powershell
cd "D:\SLIIT\3Y 2S\FDM\Mini Project\FDM Mini Project\IT3051-FDM-Mini-Project-2026"
```

Regenerate the Dataset Understanding files:

```powershell
python scripts/generate_dataset_understanding.py
```

Then open and run:

```text
01_Dataset_Understanding/Dataset_Selection_and_Understanding.ipynb
```

Run the notebook cells from top to bottom. The notebook reads the raw dataset from:

```text
data/raw/hotel_booking.csv
```

## Required Python Packages

The current stage requires:

```text
pandas
numpy
```

If needed, install them using:

```powershell
pip install pandas numpy
```

## Scope

This stage does not perform preprocessing, feature engineering, model training, model evaluation, or deployment. It is limited to Dataset Selection & Understanding only.
