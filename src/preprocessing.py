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
    Clean the raw Adult Income dataset.

    The cleaning process standardizes column names, removes extra
    whitespace from categorical values, converts missing-value
    placeholders into NaN, removes duplicate rows, and resets the index.
    """
    clean_df = df.copy()

    clean_df.columns = (
        clean_df.columns
        .str.strip()
        .str.lower()
    )

    categorical_columns = clean_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        clean_df[column] = clean_df[column].str.strip()

    clean_df = clean_df.replace("?", np.nan)

    clean_df = clean_df.drop_duplicates()

    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into input features and target variable.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]

    return X, y


def get_feature_types(
    X: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical feature columns.
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numerical and categorical features.

    Numerical features are processed using median imputation and
    standard scaling. Categorical features are processed using
    most-frequent imputation and one-hot encoding.
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
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into stratified training and testing sets.

    Stratification preserves the target-class distribution in both
    training and testing data, which is important for imbalanced
    classification problems.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering transformations.
    """
    engineered_df = df.copy()


    # ========================================================
    # Group native-country into broader categories
    # ========================================================

    engineered_df["native-country-group"] = np.where(
        engineered_df["native-country"] == "United-States",
        "United-States",
        "Other"
    )

    # ========================================================
    # Group education levels
    # ========================================================

    education_mapping = {
        "Preschool": "Low Education",
        "1st-4th": "Low Education",
        "5th-6th": "Low Education",
        "7th-8th": "Middle School",
        "9th": "Middle School",
        "10th": "High School",
        "11th": "High School",
        "12th": "High School",
        "HS-grad": "High School",
        "Some-college": "College",
        "Assoc-acdm": "Associate",
        "Assoc-voc": "Associate",
        "Bachelors": "Bachelor",
        "Masters": "Postgraduate",
        "Doctorate": "Postgraduate",
        "Prof-school": "Postgraduate",
    }

    engineered_df["education-group"] = (
        engineered_df["education"]
        .map(education_mapping)
        .fillna("Other")
    )

    # ========================================================
    # Binary capital indicators
    # ========================================================

    engineered_df["has-capital-gain"] = (
        engineered_df["capital-gain"] > 0
    ).astype(int)

    engineered_df["has-capital-loss"] = (
        engineered_df["capital-loss"] > 0
    ).astype(int)

    # ========================================================
    # Log transformations for skewed variables
    # ========================================================

    engineered_df["capital-gain-log"] = np.log1p(
        engineered_df["capital-gain"]
    )

    engineered_df["capital-loss-log"] = np.log1p(
        engineered_df["capital-loss"]
    )

    # ========================================================
    # Remove low-interpretability sampling-weight feature
    # ========================================================

    engineered_df = engineered_df.drop(
        columns=[
            "fnlwgt",
            "capital_gain",
            "capital_loss",
        ],
        errors="ignore"
    )
    
    return engineered_df