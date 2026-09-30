from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("xgboost")

from rdu_temperature.evaluation.xgboost import XGBoostRollingOriginEvaluator
from rdu_temperature.models.xgboost_model import (
    XGBoostFeatureSelector,
    XGBoostForecaster,
    XGBoostForecastPipeline,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "gfs_temperature_c": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "gfs_relative_humidity_pct": [80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
            "forecast_lead_hour": [24, 48, 72, 96, 120, 144],
            "target_hour_sin": [0.0, 0.5, 1.0, 0.5, 0.0, -0.5],
        }
    )
    target = pd.Series(
        [9.5, 11.2, 13.6, 15.1, 17.0, 18.7],
        name="temperature_c",
    )
    return features, target


@pytest.fixture
def small_model_params() -> dict[str, object]:
    return {
        "n_estimators": 20,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
    }


def test_selector_keeps_a_bounded_feature_subset(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    selector = XGBoostFeatureSelector(
        max_features=2,
        model_params=small_model_params,
    )

    selected = selector.fit_transform(features, target)

    assert selected.columns.tolist() == selector.selected_features_
    assert 1 <= selected.shape[1] <= 2
    assert selector.feature_importances_ is not None


def test_pipeline_returns_indexed_temperature_predictions(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    pipeline = XGBoostForecastPipeline(
        selector=XGBoostFeatureSelector(
            max_features=3,
            model_params=small_model_params,
        ),
        forecaster=XGBoostForecaster(
            model_params=small_model_params,
            early_stopping_rounds=None,
        ),
    )

    predictions = pipeline.fit(features, target).predict(features)

    assert predictions.name == "predicted_temperature_c"
    assert predictions.index.equals(features.index)
    assert len(predictions) == len(features)


def test_forecaster_rejects_target_leakage(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    leaking = features.assign(temperature_c=target)
    forecaster = XGBoostForecaster(
        model_params=small_model_params,
        early_stopping_rounds=None,
    )

    with pytest.raises(ValueError, match="target-derived"):
        forecaster.fit(leaking, target)


def test_forecaster_requires_the_training_feature_order(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    forecaster = XGBoostForecaster(
        model_params=small_model_params,
        early_stopping_rounds=None,
    ).fit(features, target)

    with pytest.raises(ValueError, match="exactly match"):
        forecaster.predict(features.loc[:, list(reversed(features.columns))])


def test_validation_features_and_target_are_required_together(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    forecaster = XGBoostForecaster(
        model_params=small_model_params,
        early_stopping_rounds=None,
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        forecaster.fit(features, target, validation_features=features)


def test_forecaster_rejects_overlapping_validation_rows(
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    forecaster = XGBoostForecaster(
        model_params=small_model_params,
        early_stopping_rounds=None,
    )

    with pytest.raises(ValueError, match="cannot overlap"):
        forecaster.fit(
            features.iloc[:4],
            target.iloc[:4],
            validation_features=features.iloc[3:],
            validation_target=target.iloc[3:],
        )


def test_pipeline_round_trip_preserves_predictions(
    tmp_path: Path,
    training_data: tuple[pd.DataFrame, pd.Series],
    small_model_params: dict[str, object],
) -> None:
    features, target = training_data
    pipeline = XGBoostForecastPipeline(
        selector=XGBoostFeatureSelector(
            max_features=3,
            model_params=small_model_params,
        ),
        forecaster=XGBoostForecaster(
            model_params=small_model_params,
            early_stopping_rounds=None,
        ),
    ).fit(features, target)
    expected = pipeline.predict(features)

    paths = pipeline.save(tmp_path / "xgboost")
    restored = XGBoostForecastPipeline.load(tmp_path / "xgboost")

    assert paths["model"].is_file()
    assert paths["metadata"].is_file()
    pd.testing.assert_series_equal(restored.predict(features), expected)
    assert restored.selected_features_ == pipeline.selected_features_


def test_rolling_origin_evaluation_allows_overlapping_test_windows(
    small_model_params: dict[str, object],
) -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "gfs_temperature_c": range(10),
            "forecast_lead_hour": [1] * 10,
        },
        index=index,
    )
    target = pd.Series(range(10), index=index, dtype="float64")

    def pipeline_factory() -> XGBoostForecastPipeline:
        return XGBoostForecastPipeline(
            selector=XGBoostFeatureSelector(
                max_features=2,
                model_params=small_model_params,
            ),
            forecaster=XGBoostForecaster(
                model_params=small_model_params,
                early_stopping_rounds=None,
            ),
        )

    result = XGBoostRollingOriginEvaluator(
        pipeline_factory,
        horizon=3,
        step=2,
    ).evaluate(features, target, initial_train_size=5)

    assert result.fold_metrics["fold"].tolist() == [1, 2]
    assert len(result.predictions) == 6
    assert result.predictions["timestamp"].duplicated().any()
    assert result.summary()["n_predictions"] == 6
