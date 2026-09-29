"""Leakage-aware XGBoost components for RDU temperature forecasting.

This module starts at the modeling boundary: callers provide a numeric feature
matrix whose values were available at the relevant forecast origin and a
measured RDU temperature target.  Downloading weather data, aligning stations,
and building historical GFS forecast/observation pairs belong in the data and
feature pipelines rather than in the estimator classes below.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "importance_type": "gain",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# These columns reveal the answer or describe how the answer was assembled.
# Safe historical observations should be explicitly lagged and renamed, for
# example ``rdu_temperature_lag_1h``.
DEFAULT_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"temperature_c", "temperature_source"}
)


def _validate_features(
    features: pd.DataFrame,
    *,
    name: str,
    forbidden_features: frozenset[str] = DEFAULT_FORBIDDEN_FEATURES,
) -> None:
    """Validate the common contract consumed by every model component."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if features.empty:
        raise ValueError(f"{name} cannot be empty.")
    if not features.columns.is_unique:
        raise ValueError(f"{name} contains duplicate feature names.")

    forbidden = sorted(forbidden_features.intersection(features.columns))
    if forbidden:
        raise ValueError(
            f"{name} contains target-derived columns: {forbidden}. "
            "Use forecast variables or explicitly lagged observations instead."
        )

    non_numeric = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError(
            f"{name} contains non-numeric columns: {non_numeric}. "
            "Encode timestamps and categories before fitting the model."
        )

    infinite = [
        column
        for column in features.columns
        if np.isinf(features[column].to_numpy(dtype="float64", na_value=np.nan)).any()
    ]
    if infinite:
        raise ValueError(f"{name} contains infinite values in columns: {infinite}.")


def _validate_target(
    target: pd.Series,
    expected_index: pd.Index,
    *,
    name: str,
) -> None:
    """Require a complete numeric target aligned row-for-row with features."""
    if not isinstance(target, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if len(target) != len(expected_index):
        raise ValueError(f"{name} must have the same number of rows as its features.")
    if not target.index.equals(expected_index):
        raise ValueError(f"{name} index must exactly match its feature index.")
    if not pd.api.types.is_numeric_dtype(target):
        raise TypeError(f"{name} must be numeric.")
    if target.isna().any():
        raise ValueError(f"{name} cannot contain missing values.")
    if np.isinf(target.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ValueError(f"{name} cannot contain infinite values.")


def _validate_temporal_holdout(
    training_index: pd.Index,
    validation_index: pd.Index,
) -> None:
    """Require a disjoint, forward-looking validation block."""
    if not training_index.is_unique or not validation_index.is_unique:
        raise ValueError("Training and validation indices must be unique.")
    if not training_index.is_monotonic_increasing:
        raise ValueError("Training rows must be ordered by time.")
    if not validation_index.is_monotonic_increasing:
        raise ValueError("Validation rows must be ordered by time.")
    if len(training_index.intersection(validation_index)):
        raise ValueError("Training and validation indices cannot overlap.")

    try:
        validation_is_forward = training_index[-1] < validation_index[0]
    except TypeError:
        validation_is_forward = True
    if not validation_is_forward:
        raise ValueError("Validation rows must occur after all training rows.")


class XGBoostFeatureSelector:
    """Rank candidate features with XGBoost and retain a reproducible subset.

    The selector must be fitted on training data only.  ``max_features`` and
    ``min_importance`` are tuning choices that should be compared with
    time-ordered validation folds, never with the final test window.
    """

    def __init__(
        self,
        *,
        max_features: int | None = None,
        min_importance: float = 0.0,
        model_params: Mapping[str, Any] | None = None,
        forbidden_features: frozenset[str] = DEFAULT_FORBIDDEN_FEATURES,
    ) -> None:
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None.")
        if min_importance < 0:
            raise ValueError("min_importance cannot be negative.")

        self.max_features = max_features
        self.min_importance = min_importance
        self.model_params = dict(model_params or {})
        self.forbidden_features = forbidden_features
        self.feature_importances_: pd.Series | None = None
        self.selected_features_: list[str] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostFeatureSelector:
        """Fit an embedded selector using training rows only."""
        _validate_features(
            features,
            name="features",
            forbidden_features=self.forbidden_features,
        )
        _validate_target(target, features.index, name="target")

        params = {**DEFAULT_MODEL_PARAMS, **self.model_params}
        params.pop("early_stopping_rounds", None)
        selector_model = XGBRegressor(**params)
        selector_model.fit(features, target, verbose=False)

        importances = pd.Series(
            selector_model.feature_importances_,
            index=features.columns,
            name="importance",
            dtype="float64",
        ).sort_values(ascending=False, kind="stable")

        selected = importances[importances > self.min_importance]
        if self.max_features is not None:
            selected = selected.head(self.max_features)
        if selected.empty:
            # A constant or very small synthetic dataset can give every feature
            # zero importance.  Keeping the highest-ranked column leaves the
            # selector usable and makes the fallback deterministic.
            selected = importances.head(1)

        self.feature_importances_ = importances
        self.selected_features_ = selected.index.tolist()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return selected columns in the exact order learned during fitting."""
        if self.selected_features_ is None:
            raise RuntimeError("The feature selector must be fitted first.")
        _validate_features(
            features,
            name="features",
            forbidden_features=self.forbidden_features,
        )
        missing = [
            feature
            for feature in self.selected_features_
            if feature not in features.columns
        ]
        if missing:
            raise ValueError(f"features is missing selected columns: {missing}.")
        return features.loc[:, self.selected_features_].copy()

    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Fit the selector and return the selected training columns."""
        return self.fit(features, target).transform(features)


class XGBoostForecaster:
    """Train, validate, predict, and save an XGBoost temperature model."""

    def __init__(
        self,
        *,
        model_params: Mapping[str, Any] | None = None,
        early_stopping_rounds: int | None = 50,
        forbidden_features: frozenset[str] = DEFAULT_FORBIDDEN_FEATURES,
    ) -> None:
        if early_stopping_rounds is not None and early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive or None.")

        self.model_params = dict(model_params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.forbidden_features = forbidden_features
        self.model_: XGBRegressor | None = None
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> XGBoostForecaster:
        """Fit the model, optionally monitoring a time-ordered validation set."""
        _validate_features(
            features,
            name="features",
            forbidden_features=self.forbidden_features,
        )
        _validate_target(target, features.index, name="target")

        validation_supplied = (
            validation_features is not None or validation_target is not None
        )
        if validation_supplied and (
            validation_features is None or validation_target is None
        ):
            raise ValueError(
                "validation_features and validation_target must be supplied together."
            )

        params = {**DEFAULT_MODEL_PARAMS, **self.model_params}
        fit_options: dict[str, Any] = {"verbose": False}
        if validation_features is not None and validation_target is not None:
            _validate_features(
                validation_features,
                name="validation_features",
                forbidden_features=self.forbidden_features,
            )
            _validate_target(
                validation_target,
                validation_features.index,
                name="validation_target",
            )
            if validation_features.columns.tolist() != features.columns.tolist():
                raise ValueError(
                    "Validation columns must exactly match training columns and order."
                )
            _validate_temporal_holdout(features.index, validation_features.index)
            fit_options["eval_set"] = [(validation_features, validation_target)]
            if self.early_stopping_rounds is not None:
                params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model_ = XGBRegressor(**params)
        self.model_.fit(features, target, **fit_options)
        self.feature_names_ = features.columns.tolist()
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict Celsius temperatures while preserving the input index."""
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("The forecaster must be fitted before prediction.")
        _validate_features(
            features,
            name="features",
            forbidden_features=self.forbidden_features,
        )
        if features.columns.tolist() != self.feature_names_:
            raise ValueError(
                "Prediction columns must exactly match training columns and order."
            )

        return pd.Series(
            self.model_.predict(features),
            index=features.index,
            name="predicted_temperature_c",
        )

    def feature_importance(self) -> pd.Series:
        """Return fitted model importance values ordered from largest to smallest."""
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("The forecaster must be fitted first.")
        return pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_,
            name="importance",
            dtype="float64",
        ).sort_values(ascending=False, kind="stable")

    def save(self, output_path: Path) -> Path:
        """Save a fitted XGBoost model beneath the requested path."""
        if self.model_ is None:
            raise RuntimeError("The forecaster must be fitted before saving.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(output_path))
        return output_path

    def load(self, input_path: Path, feature_names: list[str]) -> XGBoostForecaster:
        """Load a saved booster and restore its feature contract."""
        if not input_path.is_file():
            raise FileNotFoundError(f"Saved XGBoost model not found: {input_path}")
        model = XGBRegressor()
        model.load_model(str(input_path))
        self.model_ = model
        self.feature_names_ = feature_names.copy()
        return self


class XGBoostForecastPipeline:
    """Compose model-specific feature selection with XGBoost forecasting."""

    def __init__(
        self,
        *,
        selector: XGBoostFeatureSelector | None = None,
        forecaster: XGBoostForecaster | None = None,
    ) -> None:
        self.selector = selector or XGBoostFeatureSelector()
        self.forecaster = forecaster or XGBoostForecaster()

    @property
    def selected_features_(self) -> list[str]:
        """Expose the selected feature names after fitting."""
        if self.selector.selected_features_ is None:
            raise RuntimeError("The pipeline must be fitted first.")
        return self.selector.selected_features_.copy()

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        validation_features: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> XGBoostForecastPipeline:
        """Select on training rows, then fit on the selected feature subset."""
        selected_train = self.selector.fit_transform(features, target)
        selected_validation = None
        if validation_features is not None:
            selected_validation = self.selector.transform(validation_features)

        self.forecaster.fit(
            selected_train,
            target,
            validation_features=selected_validation,
            validation_target=validation_target,
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Apply the learned selection before generating predictions."""
        selected = self.selector.transform(features)
        return self.forecaster.predict(selected)

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a submission-friendly frame with timestamps and predictions."""
        prediction = self.predict(features)
        index_name = features.index.name or "timestamp"
        return prediction.rename_axis(index_name).reset_index()

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Persist the booster and feature-selection contract together."""
        if self.selector.selected_features_ is None:
            raise RuntimeError("The pipeline must be fitted before saving.")
        if self.selector.feature_importances_ is None:
            raise RuntimeError("Selector feature importances are unavailable.")
        if self.forecaster.feature_names_ is None:
            raise RuntimeError("Forecaster feature names are unavailable.")

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.ubj"
        metadata_path = output_dir / "metadata.json"
        self.forecaster.save(model_path)

        metadata = {
            "artifact_version": 1,
            "selected_features": self.selector.selected_features_,
            "selector_importances": self.selector.feature_importances_.to_dict(),
            "selector_max_features": self.selector.max_features,
            "selector_min_importance": self.selector.min_importance,
            "forbidden_features": sorted(self.selector.forbidden_features),
            "forecaster_features": self.forecaster.feature_names_,
            "early_stopping_rounds": self.forecaster.early_stopping_rounds,
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)
        return {"model": model_path, "metadata": metadata_path}

    @classmethod
    def load(cls, input_dir: Path) -> XGBoostForecastPipeline:
        """Restore a fitted pipeline without re-running feature selection."""
        metadata_path = input_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Pipeline metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("artifact_version") != 1:
            raise ValueError("Unsupported XGBoost pipeline artifact version.")

        forbidden = frozenset(metadata["forbidden_features"])
        selector = XGBoostFeatureSelector(
            max_features=metadata["selector_max_features"],
            min_importance=metadata["selector_min_importance"],
            forbidden_features=forbidden,
        )
        selector.selected_features_ = list(metadata["selected_features"])
        selector.feature_importances_ = pd.Series(
            metadata["selector_importances"],
            name="importance",
            dtype="float64",
        ).sort_values(ascending=False, kind="stable")

        forecaster = XGBoostForecaster(
            early_stopping_rounds=metadata["early_stopping_rounds"],
            forbidden_features=forbidden,
        ).load(input_dir / "model.ubj", list(metadata["forecaster_features"]))
        return cls(selector=selector, forecaster=forecaster)
