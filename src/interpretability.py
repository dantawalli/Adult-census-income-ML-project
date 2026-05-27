import pandas as pd
from sklearn.pipeline import Pipeline


import pandas as pd

from sklearn.pipeline import Pipeline


def extract_logistic_feature_importance(
    model: Pipeline
) -> pd.DataFrame:
    """
    Extract Logistic Regression coefficient-based feature importance.
    """
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = (
        model.named_steps["classifier"]
        .coef_[0]
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
    })

    importance_df["Absolute Coefficient"] = (
        importance_df["Coefficient"].abs()
    )

    return (
        importance_df
        .sort_values("Absolute Coefficient", ascending=False)
        .reset_index(drop=True)
    )


def extract_tree_feature_importance(
    model: Pipeline
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.

    Works for:
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    """
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    })

    return (
        importance_df
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def get_top_features(
    importance_df: pd.DataFrame,
    importance_column: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Return the top N most important features.
    """
    return importance_df.head(top_n)