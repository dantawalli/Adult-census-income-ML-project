from pathlib import Path

import kagglehub
import pandas as pd

from src.config import DATASET_NAME, DATA_FILE_NAME


class DatasetLoadError(Exception):
    """Custom exception raised when dataset loading fails."""


def download_dataset() -> Path:
    """
    Download the Adult Income dataset using KaggleHub.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    dataset_dir = kagglehub.dataset_download(DATASET_NAME)
    return Path(dataset_dir) / DATA_FILE_NAME


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    DatasetLoadError
        If the dataset cannot be loaded.
    """
    path = Path(path)

    try:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise DatasetLoadError("Dataset was loaded but is empty.")

        return df

    except Exception as error:
        raise DatasetLoadError(f"Failed to load dataset: {error}") from error