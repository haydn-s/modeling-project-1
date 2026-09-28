"""A real Prophet fit over a short synthetic series with a known shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.models.prophet_model import ProphetConfig, TemperatureProphet
from rdu_temperature.pipeline import schema

CUTOFF = pd.Timestamp("2024-01-01 00:00:00")
DIURNAL_AMPLITUDE_C = 8.0
BASELINE_C = 15.0


def _synthetic_target(hours: int = 24 * 400) -> pd.DataFrame:
    """A clean diurnal cycle on an annual swing, with no weather on top.

    The signal is exactly what Prophet decomposes into, so a fit that cannot
    recover it points at the wiring rather than at the weather.
    """
    timestamps = pd.date_range(
        CUTOFF - pd.Timedelta(hours=hours), periods=hours, freq="h"
    )
    hour_of_day = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    temperature = (
        BASELINE_C
        + DIURNAL_AMPLITUDE_C * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
        + 10.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    )
    return pd.DataFrame(
        {schema.TIMESTAMP_UTC: timestamps, schema.TEMPERATURE_C: temperature}
    )


@pytest.fixture(scope="module")
def fitted() -> TemperatureProphet:
    frame = pf.build(_synthetic_target())
    return TemperatureProphet(ProphetConfig()).fit(frame)


def test_forecast_covers_the_horizon_in_both_clocks(
    fitted: TemperatureProphet,
) -> None:
    forecast = fitted.forecast(CUTOFF, hours=48)

    assert len(forecast) == 48
    assert list(forecast.columns) == [
        pf.DS,
        "timestamp_local",
        "yhat",
        "yhat_lower",
        "yhat_upper",
    ]
    assert forecast[pf.DS].iloc[0] == CUTOFF
    assert forecast["timestamp_local"].iloc[0] == pd.Timestamp(
        "2023-12-31 19:00:00-05:00"
    )


def test_forecast_recovers_a_clean_diurnal_cycle(fitted: TemperatureProphet) -> None:
    forecast = fitted.forecast(CUTOFF, hours=24 * 3)

    swing = forecast["yhat"].max() - forecast["yhat"].min()

    # The synthetic series swings 16 C peak to trough; the fit should land
    # near that rather than flattening it away.
    assert swing == pytest.approx(2 * DIURNAL_AMPLITUDE_C, abs=3.0)


def test_intervals_bracket_the_point_forecast(fitted: TemperatureProphet) -> None:
    forecast = fitted.forecast(CUTOFF, hours=48)

    assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
    assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


def test_seeding_makes_the_sampled_interval_reproducible(
    fitted: TemperatureProphet,
) -> None:
    first = fitted.forecast(CUTOFF, hours=48)
    second = fitted.forecast(CUTOFF, hours=48)

    # Prophet samples its interval, so without a seed these would differ.
    pd.testing.assert_frame_equal(first, second)


def test_conditional_daily_adds_one_seasonality_per_season(
    fitted: TemperatureProphet,
) -> None:
    names = set(fitted.components())

    assert {"daily_winter", "daily_spring", "daily_summer", "daily_autumn"} <= names
    # The global daily seasonality is replaced, not supplemented.
    assert "daily" not in names


def test_fit_refuses_a_series_with_nothing_observed() -> None:
    frame = pf.build(
        pd.DataFrame(
            {
                schema.TIMESTAMP_UTC: pd.date_range(
                    "2024-01-01T00:00Z", periods=5, freq="h"
                ),
                schema.TEMPERATURE_C: [None] * 5,
            }
        )
    )

    with pytest.raises(ValueError, match="at least two observations"):
        TemperatureProphet(ProphetConfig()).fit(frame)


def test_forecast_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TemperatureProphet(ProphetConfig()).forecast(CUTOFF)
