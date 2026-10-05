"""A real fit of Prophet-with-GFS over a synthetic panel of known bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.models.covariate_prophet import (
    ALL_REGRESSORS,
    CovariateProphet,
    RawGfs,
)
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED

CUTOFF = pd.Timestamp("2025-01-01 00:00:00")
HORIZON = 48
# The forecast runs this much too warm everywhere, which a correction should
# recover and an uncorrected reading should not.
GFS_BIAS_C = 3.0


def _truth(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """A clean diurnal cycle on an annual swing."""
    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofyear.to_numpy()
    return (
        15.0
        + 8.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        + 10.0 * np.sin(2 * np.pi * (day - 105) / 365)
    )


def _history(hours: int = 24 * 200) -> pd.DataFrame:
    timestamps = pd.date_range(
        CUTOFF - pd.Timedelta(hours=hours), periods=hours, freq="h"
    )
    return pf.build(
        pd.DataFrame(
            {
                schema.TIMESTAMP_UTC: timestamps.tz_localize("UTC"),
                schema.TEMPERATURE_C: _truth(timestamps),
            }
        )
    )


def _panel(first: pd.Timestamp, hours: int, init_offset_hours: int = 16):
    """A run covering ``hours`` from ``first``, biased warm by a known amount."""
    init_time = first - pd.Timedelta(hours=init_offset_hours)
    valid = pd.date_range(first, periods=hours, freq="h")
    return pd.DataFrame(
        {
            schema.SOURCE: "noaa_gfs",
            schema.STATION_ID: "KRDU",
            schema.INIT_TIME_UTC: init_time,
            schema.VALID_TIME_UTC: valid,
            schema.LEAD_HOURS: ((valid - init_time).total_seconds() // 3600).astype(
                int
            ),
            schema.TEMPERATURE_C: _truth(valid) + GFS_BIAS_C,
            schema.DEWPOINT_C: _truth(valid) - 5.0,
            schema.WIND_SPEED_MS: 2.0,
            schema.WIND_DIRECTION_DEG: 180.0,
            schema.CLOUD_COVER_PCT: 50.0,
            schema.PRECIPITATION_MM: 0.0,
            INTERPOLATED: False,
        }
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Covariates over a stretch of training history, plus the test window."""
    training = _panel(CUTOFF - pd.Timedelta(hours=24 * 60), 24 * 60)
    window = _panel(CUTOFF, HORIZON)
    return pd.concat([training, window], ignore_index=True)


@pytest.fixture(scope="module")
def fitted(panel: pd.DataFrame) -> CovariateProphet:
    return CovariateProphet(panel=panel).fit(_history())


def test_training_is_restricted_to_covered_hours(
    fitted: CovariateProphet,
) -> None:
    # The history is 4,800 hours; the panel covers 1,440 of them. A regressor
    # must exist on every fitted row, so the rest cannot be used.
    assert fitted.training_hours == 24 * 60
    assert fitted.training_hours < len(_history())


def test_the_correction_removes_a_known_bias(fitted: CovariateProphet) -> None:
    forecast = fitted.forecast(CUTOFF, hours=HORIZON)

    truth = _truth(pd.DatetimeIndex(forecast[pf.DS]))
    bias = float((forecast["yhat"] - truth).mean())

    # The forecast it was handed is three degrees warm everywhere; having seen
    # that on the training hours, the fit should take it back out.
    assert abs(bias) < 1.0


def test_the_uncorrected_forecast_keeps_the_bias(panel: pd.DataFrame) -> None:
    raw = RawGfs(panel=panel).fit(_history())

    forecast = raw.forecast(CUTOFF, hours=HORIZON)

    truth = _truth(pd.DatetimeIndex(forecast[pf.DS]))
    bias = float((forecast["yhat"] - truth).mean())
    # The baseline reads the number straight off the model, bias included.
    assert bias == pytest.approx(GFS_BIAS_C, abs=0.1)


def test_the_correction_beats_the_uncorrected_forecast(
    fitted: CovariateProphet, panel: pd.DataFrame
) -> None:
    corrected = fitted.forecast(CUTOFF, hours=HORIZON)
    raw = RawGfs(panel=panel).fit(_history()).forecast(CUTOFF, hours=HORIZON)

    truth = _truth(pd.DatetimeIndex(corrected[pf.DS]))
    corrected_mae = float((corrected["yhat"] - truth).abs().mean())
    raw_mae = float((raw["yhat"] - truth).abs().mean())

    assert corrected_mae < raw_mae


def test_the_forecast_covers_the_whole_horizon(fitted: CovariateProphet) -> None:
    forecast = fitted.forecast(CUTOFF, hours=HORIZON)

    assert len(forecast) == HORIZON
    assert forecast["yhat"].notna().all()
    assert forecast[pf.DS].iloc[0] == CUTOFF


def test_every_covariate_can_be_used_as_a_regressor(panel: pd.DataFrame) -> None:
    model = CovariateProphet(panel=panel, regressors=ALL_REGRESSORS)

    model.fit(_history())
    forecast = model.forecast(CUTOFF, hours=HORIZON)

    assert forecast["yhat"].notna().all()


def test_seasonality_can_be_switched_off(panel: pd.DataFrame) -> None:
    # With no trend and no seasonality the model is a pure correction of GFS,
    # which is the comparison that shows whether Prophet's own components add
    # anything once the regressor is present.
    model = CovariateProphet(panel=panel, seasonality=False)

    model.fit(_history())
    forecast = model.forecast(CUTOFF, hours=HORIZON)

    truth = _truth(pd.DatetimeIndex(forecast[pf.DS]))
    assert abs(float((forecast["yhat"] - truth).mean())) < 1.5


def test_fitting_without_covered_hours_is_refused() -> None:
    # A panel whose runs all post-date the training window: legitimate to
    # forecast with, useless to train on.
    future_only = _panel(CUTOFF, HORIZON)

    with pytest.raises(ValueError, match="fetch the runs"):
        CovariateProphet(panel=future_only).fit(_history())


def test_forecast_before_fit_is_refused(panel: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CovariateProphet(panel=panel).forecast(CUTOFF)
