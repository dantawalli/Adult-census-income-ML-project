from pathlib import Path

import joblib

from src.config import MODEL_DIR, BEST_MODEL_NAME


def save_model(
    model,
    filename: str = BEST_MODEL_NAME
) -> Path:
    """
    Save trained model to disk.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / filename

    joblib.dump(model, model_path)

    return model_path


def load_model(model_path: str | Path):
    """
    Load trained model from disk.
    """
    return joblib.load(model_path)