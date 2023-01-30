from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    report: str


def evaluate_predictions(y_true: Optional[np.ndarray], y_pred: np.ndarray) -> Optional[Metrics]:
    if y_true is None:
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return Metrics(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        report=classification_report(y_true, y_pred, zero_division=0),
    )

