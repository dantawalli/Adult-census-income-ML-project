import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.evaluation import ModelResult, evaluate_model


POSITIVE_CLASS = ">50K"

F1_SCORER = make_scorer(
    f1_score,
    pos_label=POSITIVE_CLASS,
    zero_division=0,
)


def build_baseline_model(preprocessor) -> Pipeline:
    """
    Build a baseline classification pipeline using the most frequent class.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def compare_smote_effect(
    classifier,
    model_name: str,
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ModelResult, ModelResult]:
    """
    Compare model performance with and without SMOTE.

    SMOTE is applied only to the preprocessed training data to avoid
    data leakage. The test data remains unchanged.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    positive_class_index = int(
        np.where(label_encoder.classes_ == POSITIVE_CLASS)[0][0]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    classifier_no_smote = clone(classifier)
    classifier_no_smote.fit(X_train_preprocessed, y_train_encoded)

    predictions_no_smote = classifier_no_smote.predict(
        X_test_preprocessed
    )

    predictions_no_smote_labels = label_encoder.inverse_transform(
        predictions_no_smote
    )

    probabilities_no_smote = None

    if hasattr(classifier_no_smote, "predict_proba"):
        probabilities_no_smote = classifier_no_smote.predict_proba(
            X_test_preprocessed
        )[:, positive_class_index]

    no_smote_results = evaluate_model(
        model_name=f"{model_name} (No SMOTE)",
        y_true=y_test,
        y_pred=predictions_no_smote_labels,
        y_prob=probabilities_no_smote,
    )

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_preprocessed,
        y_train_encoded,
    )

    classifier_smote = clone(classifier)
    classifier_smote.fit(X_train_resampled, y_train_resampled)

    predictions_smote = classifier_smote.predict(X_test_preprocessed)

    predictions_smote_labels = label_encoder.inverse_transform(
        predictions_smote
    )

    probabilities_smote = None

    if hasattr(classifier_smote, "predict_proba"):
        probabilities_smote = classifier_smote.predict_proba(
            X_test_preprocessed
        )[:, positive_class_index]

    smote_results = evaluate_model(
        model_name=f"{model_name} (With SMOTE)",
        y_true=y_test,
        y_pred=predictions_smote_labels,
        y_prob=probabilities_smote,
    )

    return no_smote_results, smote_results


def get_candidate_models() -> dict[str, object]:
    """
    Return candidate classification models for comparison.
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def _get_positive_class_index(
    classifier,
    positive_class: str = POSITIVE_CLASS,
) -> int:
    """
    Get the probability-column index corresponding to the positive class.
    """
    return list(classifier.classes_).index(positive_class)


def train_and_evaluate_models(
    models: dict[str, object],
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[ModelResult], dict[str, Pipeline]]:
    """
    Train and evaluate multiple classification models.
    """
    model_results = []
    trained_models = {}

    for model_name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = None
        fitted_classifier = pipeline.named_steps["classifier"]

        if hasattr(fitted_classifier, "predict_proba"):
            positive_class_index = _get_positive_class_index(
                fitted_classifier
            )

            probabilities = pipeline.predict_proba(
                X_test
            )[:, positive_class_index]

        result = evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=predictions,
            y_prob=probabilities,
        )

        model_results.append(result)
        trained_models[model_name] = pipeline

    return model_results, trained_models


def tune_logistic_regression(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Logistic Regression using GridSearchCV.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=F1_SCORER,
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def tune_decision_tree(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Decision Tree using GridSearchCV.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 10, 20],
        "classifier__min_samples_leaf": [1, 5, 10],
        "classifier__criterion": ["gini", "entropy"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=F1_SCORER,
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def tune_random_forest(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Random Forest using GridSearchCV.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=F1_SCORER,
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def tune_gradient_boosting(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Gradient Boosting using GridSearchCV.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5],
        "classifier__subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=F1_SCORER,
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search