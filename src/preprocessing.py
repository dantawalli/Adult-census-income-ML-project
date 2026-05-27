import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    clean_df = df.copy()

    clean_df = clean_df.replace("?", np.nan)
    clean_df = clean_df.replace(" ?", np.nan)

    clean_df = clean_df.drop_duplicates()

    return clean_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target variable.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]

    return X, y


def get_feature_types(
    X: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical features.
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include="object"
    ).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    """
    numerical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_preprocessor,
                numerical_features,
            ),
            (
                "categorical",
                categorical_preprocessor,
                categorical_features,
            ),
        ]
    )

    return preprocessor


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )