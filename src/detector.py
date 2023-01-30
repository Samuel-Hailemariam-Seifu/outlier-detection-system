from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ModelType = Literal["isolation_forest", "local_outlier_factor"]


@dataclass
class DetectionResult:
    predictions: np.ndarray
    scores: np.ndarray


class OutlierDetector:
    def __init__(
        self,
        model_type: ModelType = "isolation_forest",
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: IsolationForest | LocalOutlierFactor | None = None

    def fit(self, x: pd.DataFrame) -> None:
        x_scaled = self.scaler.fit_transform(x)

        if self.model_type == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
            )
            self.model.fit(x_scaled)
            return

        if self.model_type == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20,
            )
            self.model.fit(x_scaled)
            return

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, x: pd.DataFrame) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        x_scaled = self.scaler.transform(x)
        # sklearn returns 1 for inliers and -1 for outliers.
        raw_preds = self.model.predict(x_scaled)
        predictions = (raw_preds == -1).astype(int)

        # Higher score should indicate stronger anomaly.
        if hasattr(self.model, "score_samples"):
            raw_scores = self.model.score_samples(x_scaled)
            scores = -raw_scores
        else:
            raw_scores = self.model.decision_function(x_scaled)
            scores = -raw_scores

        return DetectionResult(predictions=predictions, scores=scores)

