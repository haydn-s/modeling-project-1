import numpy as np
import pandas as pd
import pytest

from rdu_temperature.features import prophet_frame
from rdu_temperature.models.linear_model import (
    LinearTemperatureModel,
    calendar_features,
)


def _history(days: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2022-01-01", periods=days * 24, freq="h")
    hour = ds.hour.to_numpy()
    doy = ds.dayofyear.to_numpy()
    y = (
        15
        - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        + 5 * np.sin(2 * np.pi * (hour - 14) / 24)
        + rng.normal(0, 0.5, len(ds))
    )
    return pd.DataFrame({prophet_frame.DS: ds, prophet_frame.Y: y})


def test_calendar_features_have_no_missing_values():
    ds = pd.date_range("2024-01-01", periods=48, freq="h")
    features = calendar_features(ds, ds[0])
    assert not features.isna().any(axis=None)
    assert "trend_years" in features.columns
    assert features["trend_years"].iloc[0] == 0


def test_forecast_has_backtest_columns_and_length():
    history = _history()
    cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    forecast = LinearTemperatureModel().fit(history).forecast(cutoff, 336)
    assert list(forecast.columns) == [
        prophet_frame.DS,
        "timestamp_local",
        "yhat",
        "yhat_lower",
        "yhat_upper",
    ]
    assert len(forecast) == 336
    assert forecast[prophet_frame.DS].iloc[0] == cutoff
    assert (forecast["yhat_lower"] < forecast["yhat"]).all()
    assert (forecast["yhat"] < forecast["yhat_upper"]).all()


def test_recovers_a_seasonal_signal():
    history = _history()
    cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    model = LinearTemperatureModel(use_anomaly=False).fit(history)
    future = _history(days=800 + 14).iloc[-336:]
    forecast = model.forecast(cutoff, 336)
    error = forecast["yhat"].to_numpy() - future[prophet_frame.Y].to_numpy()
    assert np.abs(error).mean() < 1.0


def test_anomaly_correction_fades_with_lead_time():
    history = _history()
    history.loc[history.index[-72:], prophet_frame.Y] += 5.0
    cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    with_anomaly = LinearTemperatureModel().fit(history).forecast(cutoff, 336)
    without = (
        LinearTemperatureModel(use_anomaly=False).fit(history).forecast(cutoff, 336)
    )
    gap = (with_anomaly["yhat"] - without["yhat"]).to_numpy()
    assert gap[0] > 1.0
    assert abs(gap[-1]) < 0.1


def test_missing_targets_are_ignored():
    history = _history()
    history.loc[history.index[:100], prophet_frame.Y] = np.nan
    cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    forecast = LinearTemperatureModel().fit(history).forecast(cutoff, 24)
    assert forecast["yhat"].notna().all()


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearTemperatureModel().forecast(pd.Timestamp("2024-01-01"), 24)
