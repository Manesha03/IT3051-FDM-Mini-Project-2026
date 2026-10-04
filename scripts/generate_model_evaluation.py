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
    add_code(cells, """
    # pyright: reportUnusedImport=false
    """ + shared_import_cell() + """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
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
    add_markdown(cells, """
    ## 1. Load Shared Data Split

    This section loads the final feature-engineered dataset and applies the
    same train/test split used by the other model notebooks. The class
    distribution is shown first so the later precision, recall and F1 results
    can be interpreted in relation to the cancellation imbalance.
    """)
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
    add_markdown(cells, """
    ## 2. Class Imbalance Review

    This checks how many bookings belong to each target class. Because cancelled
    bookings are the minority class, the notebook calculates the balanced class
    weights that can make Logistic Regression pay more attention to cancelled
    cases during training.
    """)
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
    add_markdown(cells, """
    ## 3. Baseline Model

    The baseline model uses standard Logistic Regression without class
    weighting. It gives a simple linear benchmark, showing how well the shared
    predictors work before adding imbalance handling or hyperparameter tuning.
    """)
    add_code(cells, """
    logistic_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    logistic_pipeline = build_pipeline(logistic_model, prepared.X_train)
    logistic_pipeline
    """)
    add_markdown(cells, """
    ## 4. Baseline Cross-Validation

    Cross-validation evaluates the baseline on several training folds before
    touching the final test set. This helps estimate how stable the model is and
    whether its validation performance is strong enough to compare with more
    complex models.
    """)
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
    add_markdown(cells, """
    ## 5. Baseline Final Test Evaluation

    This evaluates the fitted baseline on the untouched test set. These results
    show how the simple Logistic Regression model performs on unseen bookings
    after model selection decisions were made using training data only.
    """)
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
    add_markdown(cells, """
    ## 6. Baseline Confusion Matrix

    The confusion matrix shows the actual error counts behind the summary
    metrics. It makes clear how many cancelled bookings were missed and how many
    non-cancelled bookings were incorrectly flagged as cancellations.
    """)
    add_code(cells, """
    logistic_predictions = logistic_pipeline.predict(prepared.X_test)
    logistic_confusion = confusion_matrix_frame(prepared.y_test, logistic_predictions)
    display(logistic_confusion)
    """)
    add_markdown(cells, """
    ## 7. Class-Weighted Logistic Regression

    This model adds balanced class weighting to reduce the effect of class
    imbalance. The aim is to improve recall for cancelled bookings by making
    mistakes on the minority class more costly during training.
    """)
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
    add_markdown(cells, """
    ## 8. Class-Weighted Cross-Validation

    Cross-validation checks whether class weighting consistently improves the
    training-fold results. This section is mainly used to compare recall and F1
    against the baseline model.
    """)
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
    add_markdown(cells, """
    ## 9. Class-Weighted Final Test Evaluation

    This applies the class-weighted model to the final test set. The result
    shows whether the expected recall improvement also appears on unseen data,
    and whether precision drops as a trade-off.
    """)
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
    add_markdown(cells, """
    ## 10. Class-Weighted Confusion Matrix

    This confusion matrix explains the effect of class weighting in concrete
    counts. It shows whether the model caught more actual cancellations and how
    many extra false cancellation predictions were created.
    """)
    add_code(cells, """
    weighted_logistic_predictions = weighted_logistic_pipeline.predict(prepared.X_test)
    weighted_logistic_confusion = confusion_matrix_frame(prepared.y_test, weighted_logistic_predictions)
    display(weighted_logistic_confusion)
    """)
    add_markdown(cells, """
    ## 11. Baseline vs Class-Weighted Comparison

    This table places the baseline and class-weighted models side by side. The
    purpose is to decide whether the recall gain from class weighting is worth
    the precision loss for the hotel cancellation problem.
    """)
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
    add_markdown(cells, """
    ## 12. Logistic Regression Hyperparameter Tuning

    This grid search tunes regularization strength and class weighting with
    training data only. F1-score is used for refitting because the cancellation
    class needs a practical balance between precision and recall.
    """)
    add_code(cells, """
    tuned_logistic_base = build_pipeline(
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        prepared.X_train,
    )

    logistic_param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }

    logistic_search = GridSearchCV(
        estimator=tuned_logistic_base,
        param_grid=logistic_param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=1,
        return_train_score=True,
    )
    logistic_search.fit(prepared.X_train, prepared.y_train)

    print("Best Logistic Regression parameters:")
    print(logistic_search.best_params_)
    print(f"Best CV F1: {logistic_search.best_score_:.4f}")
    """)
    add_markdown(cells, """
    **Tuning choice:** `C` controls the strength of regularization. Smaller
    values create a simpler model, while larger values allow coefficients to
    fit the training data more closely. Class weighting is included to test
    whether recall gains justify the precision trade-off for the cancellation
    class.
    """)
    add_markdown(cells, """
    ## 13. Tuned Cross-Validation

    The best grid-search model is rechecked with the common cross-validation
    summary. This confirms whether the selected settings give stable F1,
    precision and recall compared with the earlier versions.
    """)
    add_code(cells, """
    tuned_logistic_pipeline = logistic_search.best_estimator_

    tuned_logistic_cv = cross_validation_summary(
        "Logistic Regression",
        "Tuned",
        tuned_logistic_pipeline,
        prepared.X_train,
        prepared.y_train,
    )
    display(tuned_logistic_cv.round(4))
    """)
    add_markdown(cells, """
    ## 14. Tuned Final Test Evaluation

    This section evaluates the tuned model on the untouched test set. It shows
    whether the best cross-validation configuration still performs well when
    applied to bookings that were not used during tuning.
    """)
    add_code(cells, """
    tuned_logistic_pipeline, tuned_logistic_test = fit_and_evaluate(
        "Logistic Regression",
        "Tuned",
        tuned_logistic_pipeline,
        prepared,
    )

    tuned_logistic_test_results = pd.DataFrame([tuned_logistic_test])
    display(tuned_logistic_test_results.round(4))
    """)
    add_markdown(cells, """
    ## 15. Tuned Confusion Matrix

    The tuned confusion matrix shows the final error pattern for Logistic
    Regression. It helps explain whether tuning mainly changed missed
    cancellations, false alarms, or both.
    """)
    add_code(cells, """
    tuned_logistic_predictions = tuned_logistic_pipeline.predict(prepared.X_test)
    tuned_logistic_confusion = confusion_matrix_frame(prepared.y_test, tuned_logistic_predictions)
    display(tuned_logistic_confusion)
    """)
    add_markdown(cells, """
    ## 16. Baseline vs Class-Weighted vs Tuned Comparison

    This combined comparison summarizes all Logistic Regression versions. It is
    used to identify the most useful version and to explain why the selected
    version performs better or differently from the simple baseline.
    """)
    add_code(cells, """
    logistic_test_results = pd.DataFrame([
        logistic_test,
        weighted_logistic_test,
        tuned_logistic_test,
    ])
    logistic_cv_results = pd.concat([
        logistic_cv,
        weighted_logistic_cv,
        tuned_logistic_cv,
    ], ignore_index=True)

    display(logistic_test_results.round(4))
    display(logistic_cv_results.round(4))
    """)
    add_markdown(cells, """
    **Optimization interpretation:** The tuned model should be selected only if
    it improves the chosen validation objective or gives a better practical
    precision-recall trade-off on the final test set. If tuning selects
    `class_weight='balanced'`, that supports the earlier imbalance experiment.
    If it selects no class weighting, that means the extra recall was not worth
    the precision loss under the F1 objective.
    """)
    add_markdown(cells, """
    ## 17. Coefficients and Odds Ratios

    Logistic Regression is interpretable, so this section converts fitted
    coefficients into odds ratios. These values explain which encoded predictors
    increase or decrease the estimated chance of cancellation.
    """)
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
    tuned_coefficients = coefficient_table(tuned_logistic_pipeline, "Tuned")

    print("Top baseline coefficients")
    display(baseline_coefficients.round(4))

    print("Top class-weighted coefficients")
    display(weighted_coefficients.round(4))

    print("Top tuned coefficients")
    display(tuned_coefficients.round(4))
    """)
    add_markdown(cells, """
    **Coefficient interpretation:** Positive coefficients increase the log-odds
    of cancellation, while negative coefficients reduce them. Odds ratios above
    1 indicate higher cancellation odds; values below 1 indicate lower
    cancellation odds. The class-weighted coefficients show how the model shifts
    when cancelled bookings receive more importance during fitting. The tuned
    coefficient table explains which variables remain most influential after
    the selected regularization and weighting choices.
    """)
    add_markdown(cells, """
    ## 18. ROC and Precision-Recall Curves

    These curves compare the ranking ability of the Logistic Regression
    versions across different thresholds. ROC curves show overall separation,
    while precision-recall curves are especially useful for the cancelled class.
    """)
    add_code(cells, """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[0], name="Baseline")
    RocCurveDisplay.from_estimator(weighted_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[0], name="Class Weighted")
    RocCurveDisplay.from_estimator(tuned_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[0], name="Tuned")
    PrecisionRecallDisplay.from_estimator(logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[1], name="Baseline")
    PrecisionRecallDisplay.from_estimator(weighted_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[1], name="Class Weighted")
    PrecisionRecallDisplay.from_estimator(tuned_logistic_pipeline, prepared.X_test, prepared.y_test, ax=axes[1], name="Tuned")
    axes[0].set_title("Logistic Regression ROC Curves")
    axes[1].set_title("Logistic Regression Precision-Recall Curves")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "logistic_regression_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    """)
    add_markdown(cells, """
    ## 19. Interpretation Notes

    Logistic Regression should remain in the final comparison as the interpretable
    linear model. The baseline version shows normal performance at the natural
    class distribution, the class-weighted version directly addresses the
    imbalance between cancelled and non-cancelled bookings, and the tuned version
    records the best cross-validated regularization and weighting choice. The
    coefficient tables provide the explanation layer needed for the final report.
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
    add_markdown(cells, """
    ## 1. Load Shared Data Split

    This section loads the same feature-engineered dataset and train/test split
    used by the other model notebooks. The class distribution is displayed so
    the tree results can be read with the cancellation imbalance in mind.
    """)
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
    add_markdown(cells, """
    ## 2. Baseline Decision Tree

    The baseline tree is created with default depth settings. This gives an
    intentionally flexible model so we can see how well a single tree fits the
    data before pruning or limiting its complexity.
    """)
    add_code(cells, """
    tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree_pipeline = build_pipeline(tree_model, prepared.X_train)
    tree_pipeline
    """)
    add_markdown(cells, """
    ## 3. Baseline Cross-Validation

    Cross-validation tests the baseline tree on repeated training folds. The
    train-versus-validation gap is important here because unrestricted decision
    trees often overfit by memorizing detailed patterns in the training data.
    """)
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
    add_markdown(cells, """
    ## 4. Baseline Final Test Evaluation

    This evaluates the baseline tree on the untouched test set. The result
    shows whether the high flexibility of the tree actually generalizes to new
    bookings or mainly helped on the training folds.
    """)
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
    add_markdown(cells, """
    ## 5. Baseline Confusion Matrix

    The baseline confusion matrix shows the actual prediction mistakes. It
    explains how many cancellations the tree missed and how many non-cancelled
    bookings it incorrectly predicted as cancelled.
    """)
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
    add_markdown(cells, """
    ## 7. Tuned Cross-Validation

    The best tree from grid search is evaluated with the same cross-validation
    summary. The goal is to check whether limiting depth and leaf sizes reduces
    overfitting while keeping useful cancellation prediction performance.
    """)
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
    add_markdown(cells, """
    ## 8. Tuned Final Test Evaluation

    This applies the tuned tree to the final test set. It shows whether the
    simpler tree structure chosen during tuning gives more reliable performance
    on unseen bookings.
    """)
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
    add_markdown(cells, """
    ## 9. Tuned Confusion Matrix

    The tuned confusion matrix shows how pruning-style constraints changed the
    actual errors. It helps explain whether tuning reduced false positives,
    false negatives, or both.
    """)
    add_code(cells, """
    tuned_tree_predictions = tuned_tree_pipeline.predict(prepared.X_test)
    tuned_tree_confusion = confusion_matrix_frame(prepared.y_test, tuned_tree_predictions)
    display(tuned_tree_confusion)
    """)
    add_markdown(cells, """
    ## 10. Baseline vs Tuned Comparison

    This comparison places the unrestricted tree and tuned tree side by side.
    It is used to decide whether reduced overfitting and better ranking metrics
    are worth any change in default-threshold F1, precision or recall.
    """)
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
    add_markdown(cells, """
    ## 11. ROC and Precision-Recall Curves

    These curves show how the tuned tree separates cancelled and non-cancelled
    bookings across thresholds. Precision-recall is especially useful because
    the cancellation class is smaller than the non-cancellation class.
    """)
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
    add_markdown(cells, """
    ## 12. Tree Visualization

    This visualizes the first levels of the tuned decision tree. It helps make
    the model explainable by showing the early split rules that guide the most
    important prediction paths.
    """)
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


def build_random_forest() -> nbformat.NotebookNode:
    cells = []
    add_markdown(cells, """
    # Random Forest and Feature Experiment

    This notebook follows the shared model setup. It evaluates a baseline
    Random Forest, tunes hyperparameters using training data only, ranks the
    fitted model's predictors, and tests one controlled feature ablation.
    """)
    add_code(cells, shared_import_cell() + """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split

    from scripts.model_evaluation_utils import (
        RANDOM_STATE,
        OUTPUT_DIR,
        load_feature_dataset,
        split_feature_dataset,
        build_pipeline,
        get_cv,
        cross_validation_summary,
        fit_and_evaluate,
        confusion_matrix_frame,
        save_results,
    )

    FIGURE_DIR = OUTPUT_DIR / "Figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    """)
    add_markdown(cells, "## 1. Shared data split")
    add_code(cells, """
    data = load_feature_dataset()
    prepared = split_feature_dataset(data)
    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows: {len(prepared.X_test):,}")
    print(f"Predictors: {prepared.X_train.shape[1]}")
    """)
    add_markdown(cells, "## 2. Baseline Random Forest")
    add_code(cells, """
    # Unrestricted trees provide a clear baseline; fixed seed makes results reproducible.
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    baseline_pipeline = build_pipeline(baseline_model, prepared.X_train)
    baseline_cv = cross_validation_summary(
        "Random Forest", "Baseline", baseline_pipeline,
        prepared.X_train, prepared.y_train,
    )
    display(baseline_cv.round(4))
    display(baseline_cv.assign(
        train_cv_gap=baseline_cv["train_mean"] - baseline_cv["cv_mean"]
    )[["metric", "train_mean", "cv_mean", "train_cv_gap"]].round(4))
    """)
    add_markdown(cells, """
    The train/CV gap shows whether the unrestricted forest is fitting its
    training folds substantially better than held-out folds. Cross-validation
    uses only the training partition; the test partition remains untouched here.
    """)
    add_code(cells, """
    baseline_pipeline.set_params(model__n_jobs=-1)
    baseline_pipeline, baseline_test = fit_and_evaluate(
        "Random Forest", "Baseline", baseline_pipeline, prepared,
    )
    baseline_predictions = baseline_pipeline.predict(prepared.X_test)
    baseline_confusion = confusion_matrix_frame(prepared.y_test, baseline_predictions)
    display(pd.DataFrame([baseline_test]).round(4))
    display(baseline_confusion)
    save_results(baseline_confusion.rename_axis("actual_class").reset_index(),
                 "random_forest_baseline_confusion_matrix.csv")
    """)
    add_markdown(cells, "## 3. Randomized hyperparameter search")
    add_markdown(cells, """
    The search optimizes F1, balancing cancellation precision and recall. It
    uses a stratified 30% sample drawn only from the training partition to keep
    the search tractable. The selected model is scored with five-fold CV on the
    full training partition for direct comparison with baseline. `n_jobs=1`
    inside each forest avoids nested
    parallelism while the search distributes candidates/folds across workers.
    """)
    add_code(cells, """
    search_pipeline = build_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        prepared.X_train,
    )
    X_tune, _, y_tune, _ = train_test_split(
        prepared.X_train, prepared.y_train,
        train_size=0.30, random_state=RANDOM_STATE,
        stratify=prepared.y_train,
    )
    param_distributions = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 16, 28],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 3],
        "model__max_features": ["sqrt", 0.5],
    }
    search = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions=param_distributions,
        n_iter=6,
        scoring={"accuracy": "accuracy", "precision": "precision",
                 "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
                 "pr_auc": "average_precision"},
        refit="f1",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tune, y_tune)
    print("Best parameters:", search.best_params_)
    print(f"Best mean CV F1: {search.best_score_:.4f}")
    """)
    add_code(cells, """
    tuned_pipeline = search.best_estimator_
    tuned_cv = cross_validation_summary(
        "Random Forest", "Tuned", tuned_pipeline,
        prepared.X_train, prepared.y_train,
    )
    display(tuned_cv.round(4))
    display(tuned_cv.assign(
        train_cv_gap=tuned_cv["train_mean"] - tuned_cv["cv_mean"]
    )[["metric", "train_mean", "cv_mean", "train_cv_gap"]].round(4))
    """)
    add_markdown(cells, "## 4. Baseline versus tuned Random Forest")
    add_code(cells, """
    tuned_pipeline.set_params(model__n_jobs=-1)
    tuned_pipeline, tuned_test = fit_and_evaluate(
        "Random Forest", "Tuned", tuned_pipeline, prepared,
    )
    tuned_predictions = tuned_pipeline.predict(prepared.X_test)
    tuned_confusion = confusion_matrix_frame(prepared.y_test, tuned_predictions)
    test_comparison = pd.DataFrame([baseline_test, tuned_test])
    cv_comparison = pd.concat([baseline_cv, tuned_cv], ignore_index=True)
    display(test_comparison.round(4))
    display(cv_comparison.round(4))
    display(tuned_confusion)
    save_results(tuned_confusion.rename_axis("actual_class").reset_index(),
                 "random_forest_tuned_confusion_matrix.csv")
    """)
    add_markdown(cells, """
    Compare the CV and held-out test rows before concluding that tuning helped.
    A change in one metric can trade off against another; report which measures
    improved rather than claiming an across-the-board gain.
    """)
    add_markdown(cells, """
    **Observed result:** Tuning raised mean CV F1 from 0.8438 to 0.8490 and
    CV recall from 0.8101 to 0.8328, while CV ROC-AUC moved from 0.9563 to
    0.9576. On the test set, F1 rose from 0.8513 to 0.8537 and recall from
    0.8206 to 0.8379; precision declined from 0.8844 to 0.8701 and accuracy
    was essentially unchanged. The train/CV F1 gap narrowed from about 0.150
    to 0.138, so tuning modestly reduced overfitting while keeping a substantial
    gap. The tuned forest is preferable when detecting more cancellations is
    the priority, with some added false positives.
    """)
    add_markdown(cells, "## 5. Feature importance")
    add_markdown(cells, """
    Random Forest impurity importance is calculated on the tuned training fit.
    One-hot encoded columns are summed back to their source predictor so the
    ranking is interpretable at the original feature level. These scores show
    predictive use by this model, not causal effects.
    """)
    add_code(cells, """
    fitted_preprocessor = tuned_pipeline.named_steps["preprocessor"]
    fitted_forest = tuned_pipeline.named_steps["model"]
    transformed_names = fitted_preprocessor.get_feature_names_out()
    transformed_importance = pd.DataFrame({
        "transformed_feature": transformed_names,
        "importance": fitted_forest.feature_importances_,
    })

    def source_feature(name):
        name = name.split("__", 1)[-1]
        # OneHotEncoder emits source_feature_category; match the longest source
        # name prefix so underscores in source columns remain supported.
        matches = [column for column in prepared.X_train.columns
                   if name == column or name.startswith(f"{column}_")]
        return max(matches, key=len) if matches else name

    transformed_importance["feature"] = transformed_importance["transformed_feature"].map(source_feature)
    importance = (transformed_importance.groupby("feature", as_index=False)["importance"]
                  .sum().sort_values("importance", ascending=False))
    display(importance.head(20).round(5))
    top_importance = importance.head(20).sort_values("importance")
    ax = top_importance.plot.barh(x="feature", y="importance", figsize=(9, 8), legend=False)
    ax.set(title="Random Forest Feature Importance (grouped by original predictor)",
           xlabel="Summed impurity importance", ylabel="")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()
    """)
    add_markdown(cells, """
    **Importance interpretation:** `deposit_type` is the strongest predictor in
    this fitted forest, followed by `country`, `lead_time`, `agent`, `adr`, and
    `total_of_special_requests`. The model also uses prior cancellation history
    and arrival calendar details. These are patterns the forest uses to predict
    cancellations in this dataset; impurity importance does not establish that
    a factor causes cancellation, and high-cardinality fields can receive
    importance across many encoded categories.
    """)
    add_markdown(cells, "## 6. Feature ablation experiment")
    add_markdown(cells, """
    **Alternative configuration:** remove `arrival_date_week_number` and
    `arrival_date_day_of_month`, while retaining `arrival_date_month` and
    `arrival_date_year`. Week number and day-of-month encode finer calendar
    detail and overlap with the retained season/year fields. This tests whether
    that detail adds generalizable cancellation signal. The same baseline
    Random Forest, shared preprocessing, split, CV folds, and metrics are used;
    this isolates the feature-set change.
    """)
    add_code(cells, """
    removed_calendar_features = ["arrival_date_week_number", "arrival_date_day_of_month"]
    X_train_compact = prepared.X_train.drop(columns=removed_calendar_features)
    X_test_compact = prepared.X_test.drop(columns=removed_calendar_features)

    ablation_model = build_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        X_train_compact,
    )
    ablation_cv = cross_validation_summary(
        "Random Forest", "Calendar fields removed", ablation_model,
        X_train_compact, prepared.y_train,
    )
    display(ablation_cv.round(4))
    """)
    add_code(cells, """
    # Final test comparison is reported after CV to check that any CV difference
    # also appears on the untouched shared test set.
    ablation_data = type(prepared)(
        X_train=X_train_compact, X_test=X_test_compact,
        y_train=prepared.y_train, y_test=prepared.y_test,
    )
    ablation_model.set_params(model__n_jobs=-1)
    ablation_model, ablation_test = fit_and_evaluate(
        "Random Forest", "Calendar fields removed", ablation_model, ablation_data,
    )
    feature_experiment_test = pd.DataFrame([baseline_test, ablation_test])
    feature_experiment_cv = pd.concat([baseline_cv, ablation_cv], ignore_index=True)
    display(feature_experiment_test.round(4))
    display(feature_experiment_cv.round(4))
    """)
    add_markdown(cells, """
    Decide whether the feature change improved performance from the measured
    CV and test results. Small differences may be noise; look for consistent
    changes in F1, recall, ROC-AUC and PR-AUC alongside accuracy.
    """)
    add_markdown(cells, """
    **Observed result:** Removing week number and day-of-month reduced CV F1
    from 0.8438 to 0.8411, ROC-AUC from 0.9563 to 0.9548, and PR-AUC from
    0.9378 to 0.9358. Test F1 fell from 0.8513 to 0.8473 and test ROC-AUC
    from 0.9602 to 0.9585. The change did not improve this model, so retain
    the reference feature set. The difference is modest but consistent across
    CV and test measures.
    """)
    add_markdown(cells, "## 7. Save results")
    add_code(cells, """
    save_results(test_comparison, "random_forest_test_results.csv")
    save_results(cv_comparison, "random_forest_cv_results.csv")
    save_results(feature_experiment_test, "random_forest_feature_experiment_test_results.csv")
    save_results(feature_experiment_cv, "random_forest_feature_experiment_cv_results.csv")
    save_results(importance, "random_forest_feature_importance.csv")
    print("Saved Random Forest metrics and feature-importance table under 06_Model_Evaluation.")
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
        OUTPUT / "Random_Forest.ipynb": build_random_forest(),
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
