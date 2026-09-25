"""Build and execute the feature engineering and selection notebook."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "04_Feature_Engineering"


def build_notebook() -> nbformat.NotebookNode:
    cells = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text: str) -> None:
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # Feature Engineering, Feature Selection & Encoding

    **Objective:** transform the cleaned hotel-booking dataset into a documented set of
    candidate predictors for the ML data-preparation stage. This notebook creates
    derived features, removes leakage/identifier/unavailable fields, records encoding
    strategy, and exports the selected feature dataset.

    **Scope boundary:** no train/test split, fitted encoder, scaler, model training or
    prediction is performed here. Learned transformations must be fitted later using
    training data only.
    """)
    md("""
    ### 1. Load cleaned dataset

    The input is the validated cleaned dataset from `03_Data_Preprocessing`. CSV dtype
    metadata is restored with the shared loader before feature decisions are made.
    """)
    code("""
    from pathlib import Path
    import sys
    import pandas as pd
    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'data/processed/hotel_booking_cleaned.csv').exists())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.clean_hotel_bookings import load_cleaned
    from scripts.feature_engineering_hotel_bookings import (
        OUTPUT_DIR, FEATURED_PATH, ENGINEERED_COLUMNS,
        prepare_feature_engineering_outputs, export_feature_engineering_outputs,
    )

    pd.set_option('display.max_rows', 120)
    cleaned = load_cleaned()
    print(f'Cleaned input shape: {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns')
    display(cleaned.head())
    """)
    md("""
    ### 2. Review available features

    The review separates target, numerical fields, categorical fields, date fields and
    personal identifiers. High-cardinality fields are not encoded here; only their
    later encoding strategy is documented.
    """)
    code("""
    result = prepare_feature_engineering_outputs(cleaned)
    display(result['feature_types'])
    """)
    md("""
    ### 3. Leakage and identifier decisions

    `reservation_status` and `reservation_status_date` are removed because they reveal
    information only known after the cancellation outcome. Personal fields are removed
    because they identify guests and do not provide appropriate modelling signal.

    The project prediction point is the moment the initial reservation is created.
    `assigned_room_type`, `booking_changes` and `days_in_waiting_list` are excluded
    because they are likely known or accumulated after that point. Parking spaces and
    special requests are retained under the team assumption that those requests are
    recorded at initial reservation time; this assumption should be revisited if the
    source-system timing is later clarified.
    """)
    code("""
    decisions = result['feature_decisions']
    display(decisions.loc[decisions['action'].isin(['exclude', 'target'])])
    """)
    md("""
    ### 4. Feature engineering

    The engineered features are based on EDA findings and business meaning:

    - `total_stay`: total booked nights.
    - `total_guests`: total party size.
    - `has_children` and `has_babies`: family-composition indicators.
    - `previous_total_bookings`: previous booking-history volume.
    - `previous_cancellation_rate`: previous cancellations divided by previous total
      bookings, with zero history mapped to 0.
    """)
    code("""
    print('Engineered columns:')
    display(pd.DataFrame({'engineered_feature': ENGINEERED_COLUMNS}))
    display(result['featured'][['is_canceled', *ENGINEERED_COLUMNS]].head())
    """)
    md("""
    ### 5. Evaluate engineered features

    Evaluation uses distribution summaries, class-wise means and absolute correlation
    with the binary target. Some engineered features have weak individual linear
    correlation with cancellation, so they are retained as candidate predictors because
    they have meaningful business interpretations and may contribute through non-linear
    relationships or interactions. Final usefulness belongs to model development.
    """)
    code("""
    display(result['engineered_evaluation'].round(6))
    """)
    md("""
    ### 6. Feature selection

    The selected dataset keeps cleaned original predictors that are available at the
    initial booking point plus retained engineered features. Confirmed leakage,
    identifiers and clearly post-booking fields are removed. Original component fields
    such as weekday/weekend nights, guest composition and previous-history counts are
    retained with their engineered summaries because the summaries do not fully replace
    the raw information.
    """)
    code("""
    display(result['feature_decisions'])
    display(result['selected_features'])
    print(f"Selected predictor count: {len(result['selected_features'])}")
    """)
    md("""
    ### 7. Encoding and high-cardinality strategy

    This section documents the intended transformation strategy only. The later ML
    data-preparation notebook must fit encoders on training rows only. `country`,
    `agent` and `company` are high-cardinality nominal fields, so rare-category grouping
    should be learned from training data rather than from the full cleaned dataset.
    """)
    code("""
    display(result['encoding_strategy'])
    """)
    md("""
    ### 8. Export selected feature dataset and reports

    `data/processed/hotel_booking_features.csv` is the handoff dataset for ML data
    preparation. It contains `is_canceled` plus selected predictor columns. Supporting
    CSV files document feature types, engineered-feature evaluation, selection decisions,
    final selected features, encoding strategy and validation.
    """)
    code("""
    export_feature_engineering_outputs(result)
    display(result['validation'])
    print(f"Saved selected feature dataset to {FEATURED_PATH.relative_to(ROOT)}")
    print(f"Saved feature-engineering reports to {OUTPUT_DIR.relative_to(ROOT)}")
    """)
    md("""
    ### 9. Conclusion

    Feature engineering and selection are complete for this stage. The next stage should
    load `hotel_booking_features.csv`, separate `X` and `y`, create train/test
    partitions, and fit encoding/scaling transformations on training data only.
    """)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".ipython"))
    notebook = build_notebook()
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={
        "metadata": {"path": str(ROOT)}
    }).execute()
    path = OUTPUT / "Feature_Engineering_and_Selection.ipynb"
    nbformat.write(notebook, path)
    print(f"Executed and saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
