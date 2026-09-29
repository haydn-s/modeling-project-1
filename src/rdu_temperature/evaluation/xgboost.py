"""Rolling-origin evaluation for the XGBoost temperature pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rdu_temperature.models.xgboost_model import (
    XGBoostForecastPipeline,
    _validate_features,
    _validate_target,
)


@dataclass(frozen=True)
class RollingOriginResult:
    """Predictions and per-fold metrics from a rolling-origin evaluation."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame

    def summary(self) -> pd.Series:
        """Calculate metrics across every out-of-sample prediction."""
        errors = (
            self.predictions["predicted_temperature_c"]
            - self.predictions["temperature_c"]
        )
        return pd.Series(
            {
                "mae": errors.abs().mean(),
                "rmse": np.sqrt(errors.pow(2).mean()),
                "bias": errors.mean(),
                "n_predictions": len(errors),
            },
            name="rolling_origin_summary",
        )


class XGBoostRollingOriginEvaluator:
    """Evaluate independent XGBoost fits at the real forecast horizon.

    Overlapping validation windows are safe: every fold creates a fresh model,
    and its training slice ends strictly before that fold's validation slice.
    """

    def __init__(
        self,
        pipeline_factory: Callable[[], XGBoostForecastPipeline],
        *,
        horizon: int = 336,
        step: int | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive.")
        if step is not None and step < 1:
            raise ValueError("step must be positive or None.")
        self.pipeline_factory = pipeline_factory
        self.horizon = horizon
        self.step = step or horizon

    def evaluate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        initial_train_size: int,
    ) -> RollingOriginResult:
        """Run expanding-window folds without training on a fold's future rows."""
        _validate_features(features, name="features")
        _validate_target(target, features.index, name="target")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("features must have a unique, time-ordered index.")
        if initial_train_size < 1:
            raise ValueError("initial_train_size must be positive.")
        if initial_train_size + self.horizon > len(features):
            raise ValueError("Not enough rows for one complete rolling-origin fold.")

        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        for fold, origin in enumerate(
            range(
                initial_train_size,
                len(features) - self.horizon + 1,
                self.step,
            ),
            start=1,
        ):
            validation_end = origin + self.horizon
            train_features = features.iloc[:origin]
            train_target = target.iloc[:origin]
            validation_features = features.iloc[origin:validation_end]
            validation_target = target.iloc[origin:validation_end]

            pipeline = self.pipeline_factory()
            predictions = pipeline.fit(train_features, train_target).predict(
                validation_features
            )
            errors = predictions - validation_target
            fold_start = validation_features.index[0]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "forecast_origin": fold_start,
                        "lead_hour": np.arange(self.horizon),
                        "timestamp": validation_features.index,
                        "temperature_c": validation_target.to_numpy(),
                        "predicted_temperature_c": predictions.to_numpy(),
                    }
                )
            )
            metric_rows.append(
                {
                    "fold": fold,
                    "forecast_origin": fold_start,
                    "train_rows": origin,
                    "validation_rows": self.horizon,
                    "mae": errors.abs().mean(),
                    "rmse": np.sqrt(errors.pow(2).mean()),
                    "bias": errors.mean(),
                }
            )

        return RollingOriginResult(
            predictions=pd.concat(prediction_frames, ignore_index=True),
            fold_metrics=pd.DataFrame(metric_rows),
        )
