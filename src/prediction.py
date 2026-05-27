import numpy as np
import pandas as pd


def predict_income(
    model,
    input_data: pd.DataFrame
) -> np.ndarray:
    """
    Predict income class for new observations.
    """
    return model.predict(input_data)