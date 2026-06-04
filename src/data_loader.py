from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the Adult Income dataset.

    Parameters
    ----------
    path : str | Path, optional
        Dataset path. Defaults to DATASET_PATH.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    try:
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise DatasetLoadError(
                "Dataset was loaded but contains no records."
            )

        return dataframe

    except Exception as error:
        raise DatasetLoadError(
            f"Failed to load dataset from '{path}': {error}"
        ) from error