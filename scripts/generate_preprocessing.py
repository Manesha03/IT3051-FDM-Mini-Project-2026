"""Build and execute full data preprocessing, without derived predictor features."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "03_Data_Preprocessing_and_Feature_Engineering"


def build_notebook() -> nbformat.NotebookNode:
    cells = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text: str) -> None:
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # Data Preprocessing — No Feature Engineering
    ## Data Cleaning & Quality Preparation

    **Objective:** clean the validated raw bookings using the findings in
    `02_EDA/EDA.ipynb`, then prepare separate training and test inputs. Cleaning covers
    missingness, duplicates, invalid records, types, outlier review, range validation,
    and before/after checks.

    **Workflow:** EDA finding → quality problem → cleaning decision → justification
    → validation. The raw file remains unchanged. The target is `is_canceled`
    (0 = not canceled, 1 = canceled); no target values are imputed or recoded.

    **Scope:** cleaning, exclusion of unsuitable input columns, target separation,
    train/test preparation, categorical encoding, scaling, and validation of saved
    model inputs. No derived predictors such as `total_stay`, `total_guests`, ratios,
    presence indicators, or date features are created. One-hot columns represent
    existing categorical values.
    """)
    md("""
    ### 1. Reproducible setup

    Run all cells from the repository root or this notebook's folder. The shared Python
    module implements explicit, testable cleaning rules.
    """)
    code("""
    from pathlib import Path
    import hashlib
    import sys
    import pandas as pd
    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'data/raw/hotel_booking.csv').exists())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean_hotel_bookings import (
        RAW_PATH, REPORT_DIR, PROCESSED_PATH, COUNT_COLUMNS, DATE_COLUMNS,
        BINARY_COLUMNS, CATEGORY_COLUMNS, PII_COLUMNS, load_raw,
        remove_invalid_and_duplicates, correct_types_and_missing,
        quality_metrics, outlier_report, validate_cleaned, load_cleaned,
    )

    pd.set_option('display.max_rows', 100)
    raw_sha256 = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    raw = load_raw()
    print(f'Raw shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns')
    print(f'Raw SHA-256: {raw_sha256}')
    """)
    md("""
    ### 2. Baseline quality profile

    EDA §§1.12, 2.9 and 3.9 identify four fields with missing values and flag zero-guest
    bookings and ADR anomalies. Report only aggregate statistics; guest contact details
    are unnecessary for this analysis. Source-row indices in the removal audit are
    zero-based data-row positions (CSV line = index + 2).
    """)
    code("""
    before = quality_metrics(raw)
    missing_before = raw.isna().sum().rename('missing_count').to_frame()
    missing_before['missing_pct'] = missing_before['missing_count'] / len(raw) * 100
    display(missing_before.loc[missing_before['missing_count'].gt(0)].round(4))
    display(before.loc[~before.index.str.startswith('missing_')].to_frame('before'))
    """)
    md("""
    ### 3. Decisions and justifications

    - **Children (4 missing):** exclude these incomplete guest-composition records
      (approximately 0.0034% of the source). Unknown children are not assumed to be zero.
      This small, target-independent complete-case decision avoids a full-dataset
      imputation statistic. It may introduce selection bias; the class-impact report
      below makes its impact visible.
    - **Country:** replace missing values with `Unknown`; do not guess a country or fit
      a global mode. **Agent/company:** use `Not_recorded` and categorical ID strings
      instead of floating-point measurements. Missing IDs may mean no association or
      an unrecorded association; this label does not assert which explanation is true.
    - **Duplicates:** remove only complete, exact raw-row duplicates, keeping the first.
      The current source has none. Matching non-PII fields do not establish duplicate
      bookings; there is no reliable unique booking key. Retain these matches and report
      their counts as a data-quality observation.
    - **Zero total guests:** exclude records where adults + children + babies = 0
      under the project's guest-booking scope. Retain zero-adult records with children
      or babies. Total guests is calculated for validation only, not added as a feature.
    - **Negative ADR:** exclude the one negative-rate record under an explicit
      non-negative accommodation-rate policy. A credit adjustment is possible, but
      cannot be verified here. This is a project policy, not a claim of a confirmed typo.
      Retain zero ADR and positive extremes, including 5400; no justified upper limit
      was established by EDA.
    - **Other domain checks:** reject missing, negative, fractional or non-finite counts,
      invalid binary values, invalid arrival calendar dates/week numbers, and malformed
      status dates. Overlapping reasons are audited once per excluded row. Invalid ID
      codes or unexpected remaining missingness stop validation for review.
    - **Types:** guest/count/date-component fields become integers, binary fields become
      integer flags, nominal fields become categories, and status date becomes datetime.
      The source status dates use month/day/year; exports use ISO year-month-day.
      Contact/identifier fields stay textual. No predictive date feature is created.
    - **Unusual but unverified values:** keep zero-night stays, rare/`Undefined` categories,
      repeat/history disagreements and large valid counts. There is insufficient
      evidence to relabel them or equate `meal=Undefined` with another meal category.
    """)
    code("""
    eligible, removal_audit = remove_invalid_and_duplicates(raw)
    cleaned = correct_types_and_missing(eligible)
    print(f'Removed {len(removal_audit):,} unique rows; retained {len(cleaned):,}.')
    display(removal_audit['reason'].value_counts().rename_axis('reason').to_frame('rows'))
    display(pd.DataFrame({
        'country_Unknown': [int(cleaned['country'].eq('Unknown').sum())],
        'agent_Not_recorded': [int(cleaned['agent'].eq('Not_recorded').sum())],
        'company_Not_recorded': [int(cleaned['company'].eq('Not_recorded').sum())],
    }))
    """)
    md("""
    ### 4. Outliers and range review

    EDA §1.11 shows that several count variables have zero IQR, so a Tukey rule flags
    every positive count. EDA §3.4 also warns against removing extreme ADR values
    automatically. Below, raw-data IQR fences are **descriptive diagnostics only**;
    the same fences show before/after counts. They never determine retained rows or
    transformed values. Minima/maxima and integer checks verify the cleaned ranges.
    Positive extremes remain unresolved observations, not verified errors.
    """)
    code("""
    outliers = outlier_report(raw, cleaned)
    display(outliers.drop(columns='action').round(3))
    schema = pd.DataFrame({
        'column': raw.columns,
        'before_dtype': [str(raw[c].dtype) for c in raw.columns],
        'after_dtype': [str(cleaned[c].dtype) for c in raw.columns],
        'missing_before': [int(raw[c].isna().sum()) for c in raw.columns],
        'missing_after': [int(cleaned[c].isna().sum()) for c in raw.columns],
    })
    display(schema)
    """)
    md("""
    ### 5. Before/after quality and target preservation

    Missing-value counts below describe actual nulls. The categorical sentinel labels
    retain the fact that the original value was unknown. `review_only` metrics are
    observations, not failed validation rules. Matching non-PII rows are retained.
    Class proportions can change when invalid rows are excluded; every retained row
    must preserve its original target label. No class balancing is performed.
    """)
    code("""
    after = quality_metrics(cleaned)
    quality = pd.concat([before.rename('before'), after.rename('after')], axis=1)
    quality['change'] = quality['after'] - quality['before']
    quality.index.name = 'metric'
    display(quality.loc[~quality.index.str.startswith('missing_')].round(6))
    validation = validate_cleaned(raw, cleaned, removal_audit)
    display(validation)

    class_impact = pd.DataFrame({
        'before': raw['is_canceled'].value_counts().reindex([0, 1], fill_value=0),
        'after': cleaned['is_canceled'].value_counts().reindex([0, 1], fill_value=0),
    })
    class_impact['removed'] = class_impact['before'] - class_impact['after']
    class_impact['before_pct'] = class_impact['before'] / len(raw) * 100
    class_impact['after_pct'] = class_impact['after'] / len(cleaned) * 100
    class_impact.index.name = 'is_canceled'
    display(class_impact.round(4))
    """)
    md("""
    ### 6. Export and validate the cleaned dataset

    `data_quality_report.csv` holds before/after metrics; `preprocessing_summary.csv`
    links EDA findings to decisions and validation. Additional reports provide schema,
    outlier review, removal reasons and target impact. The cleaned CSV retains all
    original columns. The checks in this section verify the cleaning results and saved file.

    CSV does not preserve pandas categories or dates. Use `load_cleaned()` to restore
    the schema; it loads ID values as strings.
    """)
    code("""
    decisions = [
        ('children', 'EDA 2.9 / 3.9: four missing counts', 'Exclude incomplete rows',
         'Avoid inventing child counts; tiny target-independent exclusion', 'No missing/fractional/negative counts'),
        ('country', 'EDA 3.5 / 3.9: missing country', 'Fill Unknown',
         'Preserve uncertainty without a fitted global mode', 'No null country values'),
        ('agent, company', 'EDA 3.6 / 3.9: missing identifier-like fields', 'Fill Not_recorded; categorical string IDs',
         'Missing association is ambiguous; IDs are not magnitudes', 'No nulls; nominal dtypes'),
        ('duplicates', 'Dataset understanding: zero full duplicates', 'Drop exact raw duplicates only',
         'Matching booking attributes do not prove the same booking', 'Exact duplicates zero; non-PII matches reported'),
        ('guest composition', 'EDA 2.9: zero-guest observations', 'Exclude confirmed zero-total-guest rows',
         'Project scope requires at least one guest', 'Positive total guests; retain children-only bookings'),
        ('adr', 'EDA 3.4: negative, zero and extreme rates', 'Exclude negative/non-finite rates; retain zero/positive extremes',
         'Explicit non-negative rate policy; no verified upper cutoff', 'Finite non-negative ADR; retained values unchanged'),
        ('counts, binaries, dates', 'Dataset dictionary and EDA unusual-value checks', 'Apply domain checks and explicit types',
         'Counts are integers, flags are binary, dates must parse', 'Domain/type validation passes'),
        ('outliers', 'EDA 1.11 / 3.4: zero IQR and skewed distributions', 'Report only; no automatic clipping',
         'Rare observations are not necessarily errors', 'Before/after ranges; retained numeric values unchanged'),
        ('zero nights, Undefined, repeat/history', 'EDA 1.12 / 2.9 / 3.12: unusual observations', 'Retain for documented review',
         'Insufficient evidence to delete or recode', 'Review counts and category preservation'),
        ('is_canceled', 'Dataset target definition', 'Preserve retained-row labels',
         'Cleaning must not change booking outcomes', 'Exact target alignment and class-impact report'),
    ]
    summary = pd.DataFrame(decisions, columns=[
        'scope', 'eda_finding', 'cleaning_decision', 'justification', 'validation_after_cleaning'
    ])
    summary.insert(2, 'data_quality_problem', [
        'Incomplete guest composition because child counts are unknown',
        'Missing country labels',
        'Missing agency/company IDs and inappropriate floating-point ID types',
        'Potential repeated records; matching booking attributes alone are ambiguous',
        'Bookings with no recorded guests',
        'Negative rate violates the chosen non-negative accommodation-rate policy',
        'Counts, flags and dates need appropriate types and valid values',
        'Skewed and zero-heavy values make automatic IQR removal unreliable',
        'Unusual observations without evidence of a recording error',
        'Row exclusions can change class proportions; retained labels must stay unchanged',
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False, date_format='%Y-%m-%d')
    reloaded = load_cleaned()
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), reloaded)
    if hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() != raw_sha256:
        raise ValueError('The raw dataset changed during preprocessing.')
    validation = pd.concat([validation, pd.DataFrame([
        {'check': 'Saved CSV restores identical values and schema', 'passed': True},
        {'check': 'Raw file SHA-256 unchanged', 'passed': True},
    ])], ignore_index=True)
    quality.to_csv(REPORT_DIR / 'data_quality_report.csv')
    summary.to_csv(REPORT_DIR / 'preprocessing_summary.csv', index=False)
    schema.to_csv(REPORT_DIR / 'schema_report.csv', index=False)
    outliers.to_csv(REPORT_DIR / 'outlier_review.csv', index=False)
    removal_audit.to_csv(REPORT_DIR / 'removed_rows.csv', index=False)
    validation.to_csv(REPORT_DIR / 'cleaning_validation.csv', index=False)
    class_impact.to_csv(REPORT_DIR / 'target_distribution_comparison.csv')
    print(f'Saved {len(cleaned):,} rows × {len(cleaned.columns)} columns to {PROCESSED_PATH.relative_to(ROOT)}')
    print(f'All {len(validation)} cleaning/export checks passed.')
    display(summary)
    """)
    md("""
    ### 7. Cleaning results

    The current source contains **119,390 rows and 36 columns**. The documented rules
    exclude **185 rows** (4 missing child counts, 180 zero-guest rows, 1 negative ADR),
    leaving **119,205 rows and 36 columns**. There are no remaining nulls, exact duplicate
    records or violations of the declared cleaning rules. Valid extremes and ambiguous
    booking-history records remain unchanged. The executed reports above are the
    authoritative results if the source changes.

    Completed responsibilities: missing-value handling, duplicate handling, invalid-record
    handling, type correction, justified outlier decisions, value/range verification,
    before/after quality checks, and documentation of each cleaning decision.
    """)
    md("""
    ### 8. Predictor eligibility and prediction-time assumptions

    The next steps complete preprocessing using **original columns only**. The cleaned
    CSV remains an intermediate dataset. Predictor exclusions are documented below.
    `reservation_status` and `reservation_status_date` reveal the outcome; personal
    fields are excluded. Room assignment, booking changes and completed waiting time
    may be recorded after initial booking. Parking/special requests can change, and
    the recorded ADR is not verified as the original quote, so these are conservatively
    excluded from the initial-booking predictor set.

    Remaining fields are assumed to describe the booking and prior history available
    at reservation creation. The source has no historical snapshots to verify every
    field's availability. Removing confirmed leakage does not resolve that source-data
    limitation. No correlation-based selection or derived-feature evaluation is performed.
    """)
    code("""
    from scripts.preprocess_hotel_bookings import (
        MODEL_DATA_DIR, prepare_predictors, preprocess, export_preprocessed,
    )
    X, y, column_decisions = prepare_predictors(cleaned)
    display(column_decisions)
    print(f'Original predictor columns: {X.shape[1]}; derived predictors: 0')
    """)
    md("""
    ### 9. Group-aware train/test split and train-only transformations

    Use the first fold of a five-fold `StratifiedGroupKFold` with shuffle enabled and
    seed 42 for an approximately 80/20 split. All rows with identical **retained original
    predictor values** share a group and remain in one partition; target values and
    excluded columns do not define groups. Group hashes and row positions are audit
    metadata, never predictors. Stratification attempts to preserve class balance;
    group constraints can prevent exact proportions. This random grouped split does
    not measure future-period generalization or guarantee independent customers when
    no reliable customer/group key is available.

    Fit one `ColumnTransformer` on training rows only:

    - Numerical inputs: `RobustScaler` uses training medians and IQRs. A zero IQR uses
      scale 1, so valid nonzero history counts are retained; no outliers are clipped.
    - Nominal inputs: one-hot encoding preserves categories without imposing an order.
      Unseen test categories become all-zero within that column's encoded block.
    - Country/agent/company: one-hot encoding groups categories seen fewer than 50 times
      in training, with at most 30 encoded levels per field. These are fixed policy
      settings. Frequency counts and category vocabularies are learned from training
      only. Unseen values use the infrequent level when present, otherwise all-zero.
    - The existing `is_repeated_guest` binary flag passes through unchanged.

    Missing values were handled by the fixed cleaning policies above; no statistical
    imputation is fitted to the full dataset. Category dtype metadata is removed before
    fitting encoders. The test partition is transformed using the fitted object without
    refitting. The same scaling is suitable for later distance/linear models; tree models
    generally do not require scaling. No model is trained here.
    """)
    code("""
    processed = preprocess(cleaned)
    display(processed['split_summary'].round(4))
    display(processed['validation'])
    """)
    md("""
    ### 10. Save and verify the processed inputs

    `data/processed/model_inputs/` contains sparse `X_train.npz` and `X_test.npz`,
    single-column `y_train.csv` and `y_test.csv`, ordered `encoded_columns.csv`, the
    fitted `preprocessor.joblib`, and preprocessing settings/version metadata.
    `split_assignment.csv` maps matrix-row positions to cleaned positions and original
    source-row indices. It must never be included among predictors.

    The reports `preprocessing_columns.csv`, `train_test_summary.csv` and
    `preprocessing_validation.csv` document column treatment, partition shapes/class
    balance and validation. Export checks reload matrices, target files and the fitted
    preprocessor to confirm unchanged values and consistent transformations.
    """)
    code("""
    export_preprocessed(processed)
    preprocessing_checks = pd.read_csv(REPORT_DIR / 'preprocessing_validation.csv')
    display(preprocessing_checks)
    print(f'Saved processed inputs to {MODEL_DATA_DIR.relative_to(ROOT)}')
    print(f'All {len(preprocessing_checks)} preprocessing checks passed; derived predictors: 0.')
    """)
    md("""
    ### 11. Load the saved matrices for later modelling

    These arrays contain the same ordered columns in both partitions. Read only the
    `is_canceled` column as the target. The saved transformer applies the existing
    training transformations to new cleaned rows containing the original predictor
    columns listed in `preprocessing_metadata.json`.
    """)
    code("""
    from scipy.sparse import load_npz
    X_train = load_npz(MODEL_DATA_DIR / 'X_train.npz')
    X_test = load_npz(MODEL_DATA_DIR / 'X_test.npz')
    y_train = pd.read_csv(MODEL_DATA_DIR / 'y_train.csv')['is_canceled']
    y_test = pd.read_csv(MODEL_DATA_DIR / 'y_test.csv')['is_canceled']
    print(f'Training: X={X_train.shape}, y={y_train.shape}')
    print(f'Test: X={X_test.shape}, y={y_test.shape}')
    """)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Keep Jupyter's runtime/cache writes inside the workspace.
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".ipython"))
    notebook = build_notebook()
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={
        "metadata": {"path": str(ROOT)}
    }).execute()
    path = OUTPUT / "Data_Preprocessing_and_Feature_Engineering.ipynb"
    nbformat.write(notebook, path)
    print(f"Executed and saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
