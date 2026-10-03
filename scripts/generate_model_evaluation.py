"""Generate shared setup and model-specific notebooks for modelling."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "06_Model_Evaluation"


def add_markdown(cells: list, text: str) -> None:
    cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))


def add_code(cells: list, text: str) -> None:
    cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))


def notebook(cells: list) -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


def shared_import_cell() -> str:
    return """
    from pathlib import Path
    import sys

    ROOT = next(
        path
        for path in [Path.cwd(), *Path.cwd().parents]
        if (path / "data/processed/hotel_booking_features.csv").is_file()
    )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    """


def build_shared_setup() -> nbformat.NotebookNode:
    cells = []
    add_markdown(cells, """
    # Shared Model Setup

    This notebook defines the shared setup that every model notebook must use:
    the same input dataset, train/test split, preprocessing pipeline, random
    state, cross-validation strategy and evaluation metrics.

    Model tuning and model selection should use the training set with
    cross-validation. The final test set should be used only for final model
    evaluation.
    """)
    add_code(cells, shared_import_cell() + """

    from scripts.model_evaluation_utils import (
        FEATURE_DATA_PATH,
        RANDOM_STATE,
        TEST_SIZE,
        CV_SPLITS,
        COMMON_SCORING,
        load_feature_dataset,
        split_feature_dataset,
        class_distribution,
        build_preprocessor,
        get_cv,
    )
    """)
    add_markdown(cells, "## 1. Load Final Feature Dataset")
    add_code(cells, """
    data = load_feature_dataset()
    print(f"Dataset path: {FEATURE_DATA_PATH.relative_to(ROOT)}")
    print(f"Dataset shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Target values: {sorted(data['is_canceled'].unique())}")
    data.head()
    """)
    add_markdown(cells, "## 2. Shared Train/Test Split")
    add_code(cells, """
    prepared = split_feature_dataset(data)

    print(f"Random state: {RANDOM_STATE}")
    print(f"Test size: {TEST_SIZE:.0%}")
    print(f"CV splits: {CV_SPLITS}")
    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows: {len(prepared.X_test):,}")

    print("\\nTraining class distribution")
    display(class_distribution(prepared.y_train))

    print("\\nTest class distribution")
    display(class_distribution(prepared.y_test))
    """)
    add_markdown(cells, """
    ## 3. Shared Preprocessing

    Numeric fields are robust-scaled. Low-cardinality categorical fields are
    one-hot encoded. High-cardinality fields (`country`, `agent`, `company`) use
    infrequent-category grouping learned from the training data only.
    """)
    add_code(cells, """
    preprocessor = build_preprocessor(prepared.X_train)
    transformed_sample = preprocessor.fit_transform(prepared.X_train.head(100))

    print(preprocessor)
    print(f"Sample transformed shape: {transformed_sample.shape}")
    """)
    add_markdown(cells, "## 4. Cross-Validation and Metrics")
    add_code(cells, """
    cv = get_cv()
    print(cv)
    print("Scoring metrics:")
    for name, scorer in COMMON_SCORING.items():
        print(f"- {name}: {scorer}")
    """)
    return notebook(cells)


def build_logistic_regression() -> nbformat.NotebookNode:
    cells = []
    add_markdown(cells, """
    # Logistic Regression

    This notebook trains and evaluates Logistic Regression using the shared
    dataset split, preprocessing pipeline, cross-validation approach and metrics.
    Logistic Regression is used as a simple baseline because it is interpretable
    and works well with scaled numeric features and one-hot encoded categories.
    """)
    add_code(cells, shared_import_cell() + """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    from scripts.model_evaluation_utils import (
        RANDOM_STATE,
        OUTPUT_DIR,
        load_feature_dataset,
        split_feature_dataset,
        class_distribution,
        build_pipeline,
        cross_validation_summary,
        fit_and_evaluate,
        confusion_matrix_frame,
    )

    FIGURE_DIR = OUTPUT_DIR / "Figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    """)
    add_markdown(cells, "## 1. Load Shared Data Split")
    add_code(cells, """
    data = load_feature_dataset()
    prepared = split_feature_dataset(data)

    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows: {len(prepared.X_test):,}")

    print("\\nTraining class distribution")
    display(class_distribution(prepared.y_train))

    print("\\nTest class distribution")
    display(class_distribution(prepared.y_test))
    """)
    add_markdown(cells, "## 2. Class Imbalance Review")
    add_code(cells, """
    training_distribution = class_distribution(prepared.y_train)
    negative_count = int((prepared.y_train == 0).sum())
    positive_count = int((prepared.y_train == 1).sum())
    total_count = len(prepared.y_train)

    class_weights = {
        0: total_count / (2 * negative_count),
        1: total_count / (2 * positive_count),
    }

    print("Balanced class weights used by LogisticRegression(class_weight='balanced'):")
    print(class_weights)
    display(training_distribution)
    """)
    add_markdown(cells, "## 3. Baseline Model")
    add_code(cells, """
    logistic_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    logistic_pipeline = build_pipeline(logistic_model, prepared.X_train)
    logistic_pipeline
    """)
    add_markdown(cells, "## 4. Baseline Cross-Validation")
    add_code(cells, """
    logistic_cv = cross_validation_summary(
        "Logistic Regression",
        "Baseline",
        logistic_pipeline,
        prepared.X_train,
        prepared.y_train,
    )
    display(logistic_cv.round(4))
    """)
    add_markdown(cells, """
    **Interpretation:** Logistic Regression shows how well a simple linear model
    can separate cancelled and non-cancelled bookings after the shared
    preprocessing step. The cross-validation metrics are used to judge expected
    generalization before looking at the final test set. A small gap between
    training and validation scores indicates limited overfitting.
    """)
    add_markdown(cells, "## 5. Baseline Final Test Evaluation")
    add_code(cells, """
    logistic_pipeline, logistic_test = fit_and_evaluate(
        "Logistic Regression",
        "Baseline",
        logistic_pipeline,
        prepared,
    )

    logistic_test_results = pd.DataFrame([logistic_test])
    display(logistic_test_results.round(4))
    """)
    add_markdown(cells, """
    **Test-set interpretation:** The final test metrics should be compared with
    the cross-validation results. If the values are close, the model is behaving
    consistently on unseen data. For this dataset, Logistic Regression is useful
    as a stable and interpretable baseline, but its recall should be checked
    carefully because low recall means more actual cancellations are missed.
    """)
    add_markdown(cells, "## 6. Baseline Confusion Matrix")
    add_code(cells, """
    logistic_predictions = logistic_pipeline.predict(prepared.X_test)
    logistic_confusion = confusion_matrix_frame(prepared.y_test, logistic_predictions)
    display(logistic_confusion)
    """)
    add_markdown(cells, "## 7. Class-Weighted Logistic Regression")
    add_code(cells, """
    weighted_logistic_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    weighted_logistic_pipeline = build_pipeline(weighted_logistic_model, prepared.X_train)
    weighted_logistic_pipeline
    """)
    add_markdown(cells, "## 8. Class-Weighted Cross-Validation")
    add_code(cells, """
    weighted_logistic_cv = cross_validation_summary(
        "Logistic Regression",
        "Class Weighted",
        weighted_logistic_pipeline,
        prepared.X_train,
        prepared.y_train,
    )
    display(weighted_logistic_cv.round(4))
    """)
    add_markdown(cells, "## 9. Class-Weighted Final Test Evaluation")
    add_code(cells, """
    weighted_logistic_pipeline, weighted_logistic_test = fit_and_evaluate(
        "Logistic Regression",
        "Class Weighted",
        weighted_logistic_pipeline,
        prepared,
    )

    weighted_logistic_test_results = pd.DataFrame([weighted_logistic_test])
    display(weighted_logistic_test_results.round(4))
    """)
    add_markdown(cells, "## 10. Class-Weighted Confusion Matrix")
    add_code(cells, """
    weighted_logistic_predictions = weighted_logistic_pipeline.predict(prepared.X_test)
    weighted_logistic_confusion = confusion_matrix_frame(prepared.y_test, weighted_logistic_predictions)
    display(weighted_logistic_confusion)
    """)
    add_markdown(cells, "## 11. Baseline vs Class-Weighted Comparison")
    add_code(cells, """
    logistic_test_results = pd.DataFrame([
        logistic_test,
        weighted_logistic_test,
    ])
    logistic_cv_results = pd.concat([
        logistic_cv,
        weighted_logistic_cv,
    ], ignore_index=True)

    display(logistic_test_results.round(4))
    display(logistic_cv_results.round(4))
    """)
    add_markdown(cells, """
    **Class-imbalance interpretation:** The class-weighted model gives the
    cancellation class a larger penalty during training. This usually increases
    recall for cancelled bookings, but it can reduce precision because more
    bookings are predicted as cancellations. The final choice should depend on
    whether missing cancellations is more costly than extra follow-up actions.
    """)
    add_markdown(cells, "## 12. Coefficients and Odds Ratios")
    add_code(cells, """
    def coefficient_table(pipeline, version, top_n=15):
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_[0]
        frame = pd.DataFrame({
            "version": version,
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        })
        return frame.sort_values("absolute_coefficient", ascending=False).head(top_n)

    baseline_coefficients = coefficient_table(logistic_pipeline, "Baseline")
    weighted_coefficients = coefficient_table(weighted_logistic_pipeline, "Class Weighted")

    print("Top baseline coefficients")
    display(baseline_coefficients.round(4))

    print("Top class-weighted coefficients")
    display(weighted_coefficients.round(4))
    """)
    add_markdown(cells, """
    **Coefficient interpretation:** Positive coefficients increase the log-odds
    of cancellation, while negative coefficients reduce them. Odds ratios above
    1 indicate higher cancellation odds; values below 1 indicate lower
    cancellation odds. The class-weighted coefficients show how the model shifts
    when cancelled bookings receive more importance during fitting.
    """)
    add_markdown(cells, "## 13. ROC and Precision-Recall Curves")
    add_code(cells, """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[0], name="Baseline")
    RocCurveDisplay.from_estimator(weighted_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[0], name="Class Weighted")
    PrecisionRecallDisplay.from_estimator(logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[1], name="Baseline")
    PrecisionRecallDisplay.from_estimator(weighted_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[1], name="Class Weighted")
    axes[0].set_title("Logistic Regression ROC Curves")
    axes[1].set_title("Logistic Regression Precision-Recall Curves")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "logistic_regression_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    """)
    add_markdown(cells, """
    ## 14. Interpretation Notes

    Logistic Regression should remain in the final comparison as the interpretable
    linear model. The baseline version shows normal performance at the natural
    class distribution, while the class-weighted version directly addresses the
    imbalance between cancelled and non-cancelled bookings. The coefficient
    tables provide the explanation layer needed for the final report.
    """)
    return notebook(cells)


def build_decision_tree() -> nbformat.NotebookNode:
    cells = []
    add_markdown(cells, """
    # Decision Tree

    This notebook trains and evaluates a baseline Decision Tree, then tunes the
    tree using cross-validation on the training data only. The goal is to compare
    baseline and tuned performance and check whether tuning reduces overfitting.
    """)
    add_code(cells, shared_import_cell() + """

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    from scripts.model_evaluation_utils import (
        RANDOM_STATE,
        OUTPUT_DIR,
        load_feature_dataset,
        split_feature_dataset,
        class_distribution,
        build_pipeline,
        get_cv,
        cross_validation_summary,
        fit_and_evaluate,
        confusion_matrix_frame,
    )

    FIGURE_DIR = OUTPUT_DIR / "Figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    """)
    add_markdown(cells, "## 1. Load Shared Data Split")
    add_code(cells, """
    data = load_feature_dataset()
    prepared = split_feature_dataset(data)

    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows: {len(prepared.X_test):,}")

    print("\\nTraining class distribution")
    display(class_distribution(prepared.y_train))

    print("\\nTest class distribution")
    display(class_distribution(prepared.y_test))
    """)
    add_markdown(cells, "## 2. Baseline Decision Tree")
    add_code(cells, """
    tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree_pipeline = build_pipeline(tree_model, prepared.X_train)
    tree_pipeline
    """)
    add_markdown(cells, "## 3. Baseline Cross-Validation")
    add_code(cells, """
    tree_cv = cross_validation_summary(
        "Decision Tree",
        "Baseline",
        tree_pipeline,
        prepared.X_train,
        prepared.y_train,
    )
    display(tree_cv.round(4))
    """)
    add_code(cells, """
    baseline_gap = tree_cv.assign(
        train_cv_gap=tree_cv["train_mean"] - tree_cv["cv_mean"]
    )
    display(baseline_gap[["metric", "train_mean", "cv_mean", "train_cv_gap"]].round(4))
    """)
    add_markdown(cells, """
    **Baseline interpretation:** A Decision Tree can capture non-linear patterns
    and feature interactions, but an unrestricted tree can memorize the training
    data. The train-vs-CV gap table is used to check overfitting. A large gap,
    especially for F1, suggests that the baseline tree is too complex.
    """)
    add_markdown(cells, "## 4. Baseline Final Test Evaluation")
    add_code(cells, """
    tree_pipeline, tree_test = fit_and_evaluate(
        "Decision Tree",
        "Baseline",
        tree_pipeline,
        prepared,
    )

    tree_test_results = pd.DataFrame([tree_test])
    display(tree_test_results.round(4))
    """)
    add_markdown(cells, "## 5. Baseline Confusion Matrix")
    add_code(cells, """
    tree_predictions = tree_pipeline.predict(prepared.X_test)
    tree_confusion = confusion_matrix_frame(prepared.y_test, tree_predictions)
    display(tree_confusion)
    """)
    add_markdown(cells, """
    ## 6. Decision Tree Tuning

    The grid search uses F1-score because cancellation prediction should balance
    precision and recall. Tuning is performed with training data and
    cross-validation only.
    """)
    add_code(cells, """
    tuned_tree_base = build_pipeline(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        prepared.X_train,
    )

    tree_param_grid = {
        "model__max_depth": [6, 10],
        "model__min_samples_split": [20, 100],
        "model__min_samples_leaf": [10, 25],
        "model__criterion": ["gini", "entropy"],
    }

    tree_search = GridSearchCV(
        estimator=tuned_tree_base,
        param_grid=tree_param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=-1,
        return_train_score=True,
    )
    tree_search.fit(prepared.X_train, prepared.y_train)

    print("Best Decision Tree parameters:")
    print(tree_search.best_params_)
    print(f"Best CV F1: {tree_search.best_score_:.4f}")
    """)
    add_markdown(cells, """
    **Tuning choice:** F1-score is used for tuning because cancellation
    prediction needs a balance between precision and recall. False positives may
    cause unnecessary follow-up actions, while false negatives mean the hotel may
    miss bookings that are likely to cancel. ROC-AUC and PR-AUC are still
    reported to understand ranking and probability discrimination beyond one
    fixed threshold.
    """)
    add_markdown(cells, "## 7. Tuned Cross-Validation")
    add_code(cells, """
    tuned_tree_pipeline = tree_search.best_estimator_

    tuned_tree_cv = cross_validation_summary(
        "Decision Tree",
        "Tuned",
        tuned_tree_pipeline,
        prepared.X_train,
        prepared.y_train,
    )
    display(tuned_tree_cv.round(4))
    """)
    add_code(cells, """
    tuned_gap = tuned_tree_cv.assign(
        train_cv_gap=tuned_tree_cv["train_mean"] - tuned_tree_cv["cv_mean"]
    )
    display(tuned_gap[["metric", "train_mean", "cv_mean", "train_cv_gap"]].round(4))
    """)
    add_markdown(cells, """
    **Tuned model interpretation:** The tuned tree should reduce the training
    score compared with the unrestricted baseline. That is expected and usually
    desirable when the baseline has a large train-vs-CV gap. The important
    question is whether validation and test performance become more reliable.
    """)
    add_markdown(cells, "## 8. Tuned Final Test Evaluation")
    add_code(cells, """
    tuned_tree_pipeline, tuned_tree_test = fit_and_evaluate(
        "Decision Tree",
        "Tuned",
        tuned_tree_pipeline,
        prepared,
    )

    tuned_tree_test_results = pd.DataFrame([tuned_tree_test])
    display(tuned_tree_test_results.round(4))
    """)
    add_markdown(cells, "## 9. Tuned Confusion Matrix")
    add_code(cells, """
    tuned_tree_predictions = tuned_tree_pipeline.predict(prepared.X_test)
    tuned_tree_confusion = confusion_matrix_frame(prepared.y_test, tuned_tree_predictions)
    display(tuned_tree_confusion)
    """)
    add_markdown(cells, "## 10. Baseline vs Tuned Comparison")
    add_code(cells, """
    decision_tree_test_results = pd.DataFrame([
        tree_test,
        tuned_tree_test,
    ])
    decision_tree_cv_results = pd.concat([
        tree_cv,
        tuned_tree_cv,
    ], ignore_index=True)

    display(decision_tree_test_results.round(4))
    display(decision_tree_cv_results.round(4))
    """)
    add_code(cells, """
    f1_comparison = decision_tree_test_results.loc[
        :, ["version", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    display(f1_comparison.round(4))
    """)
    add_markdown(cells, """
    **Comparison interpretation:** The tuned tree should not be described as
    better on every metric unless the table supports that claim. If tuned F1 is
    lower but ROC-AUC and PR-AUC are higher, the correct conclusion is that
    tuning improved ranking/probability discrimination and reduced overfitting,
    while the default-threshold F1 did not improve. This trade-off is important
    for the final model comparison.
    """)
    add_markdown(cells, "## 11. ROC and Precision-Recall Curves")
    add_code(cells, """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(tuned_tree_pipeline, prepared.X_test, prepared.y_test, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(tuned_tree_pipeline, prepared.X_test, prepared.y_test, ax=axes[1])
    axes[0].set_title("Tuned Decision Tree ROC Curve")
    axes[1].set_title("Tuned Decision Tree Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "decision_tree_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    """)
    add_markdown(cells, "## 12. Tree Visualization")
    add_code(cells, """
    fitted_tree = tuned_tree_pipeline.named_steps["model"]

    plt.figure(figsize=(20, 8))
    plot_tree(
        fitted_tree,
        max_depth=3,
        filled=True,
        rounded=True,
        class_names=["Not Cancelled", "Cancelled"],
    )
    plt.title("Tuned Decision Tree - First Three Levels")
    plt.savefig(FIGURE_DIR / "decision_tree_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()
    """)
    add_markdown(cells, """
    ## 13. Interpretation Notes

    The baseline Decision Tree is expected to achieve very high training scores
    because it can grow deep and memorize patterns. The tuned tree is more
    constrained, so its training score may be lower, but its validation gap and
    probability-based metrics can be stronger. Keep both baseline and tuned
    results for the final comparison and explain the precision-recall trade-off
    clearly.
    """)
    return notebook(cells)


def main(execute: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".ipython"))

    notebooks = {
        OUTPUT / "Shared_Model_Setup.ipynb": build_shared_setup(),
        OUTPUT / "Logistic_Regression.ipynb": build_logistic_regression(),
        OUTPUT / "Decision_Tree.ipynb": build_decision_tree(),
    }
    for path, nb in notebooks.items():
        if execute:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={
                "metadata": {"path": str(ROOT)}
            }).execute()
        nbformat.write(nb, path)
        print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate model-evaluation notebooks.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute notebooks while generating them so notebook outputs and figures are refreshed.",
    )
    args = parser.parse_args()
    main(execute=args.execute)
