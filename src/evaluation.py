from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelResult:
    """
    Structured container for model evaluation results.
    """
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mcc: float
    roc_auc: float | None = None

    def to_dict(self) -> dict:
        """
        Convert model result into dictionary format.
        """
        return {
            "Model": self.model,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
            "MCC": self.mcc,
            "ROC-AUC": self.roc_auc,
        }


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ModelResult:
    """
    Evaluate classification model performance.
    """
    roc_auc = None

    if y_prob is not None:
        roc_auc = roc_auc_score(
            (y_true == ">50K").astype(int),
            y_prob
        )

    return ModelResult(
        model=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(
            y_true,
            y_pred,
            pos_label=">50K",
            zero_division=0
        ),
        recall=recall_score(
            y_true,
            y_pred,
            pos_label=">50K",
           zero_division=0
        ),
        f1_score=f1_score(
            y_true,
            y_pred,
            pos_label=">50K",
            zero_division=0
        ),
        mcc=matthews_corrcoef(y_true, y_pred),
        roc_auc=roc_auc,
    )


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    """
    Convert list of ModelResult objects into a DataFrame.
    """
    return pd.DataFrame([
        result.to_dict()
        for result in results
    ])