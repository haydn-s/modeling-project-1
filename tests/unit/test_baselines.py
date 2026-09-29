from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.models.baselines import Climatology, SeasonalPersistence
from rdu_temperature.pipeline import schema

CUTOFF = pd.Timestamp("2025-01-01 00:00:00")


def _history(hours: int, values) -> pd.DataFrame:
    """A Prophet frame ending exactly at the cutoff."""
    timestamps = pd.date_range(
        CUTOFF - pd.Timedelta(hours=hours), periods=hours, freq="h"
    )
    return pf.build(
        pd.DataFrame(
            {
                schema.TIMESTAMP_UTC: timestamps.tz_localize("UTC"),
                schema.TEMPERATURE_C: values,
            }
        )
    )


def _diurnal(hours: int, amplitude: float = 10.0, baseline: float = 15.0):
    index = np.arange(hours)
    return baseline + amplitude * np.sin(2 * np.pi * (index % 24) / 24)


def test_climatology_recovers_a_repeating_daily_shape() -> None:
    hours = 24 * 400
    history = _history(hours, _diurnal(hours))

    forecast = Climatology().fit(history).forecast(CUTOFF, hours=48)

    # Every day of the training series is identical, so the climatology should
    # reproduce it rather than flatten it.
    expected = _diurnal(48 + int(CUTOFF.hour))[:48]
    assert forecast["yhat"].to_numpy() == pytest.approx(expected, abs=0.5)


def test_climatology_averages_across_the_day_window() -> None:
    hours = 24 * 400
    history = _history(hours, _diurnal(hours))

    narrow = Climatology(window_days=0).fit(history).forecast(CUTOFF, hours=24)
    wide = Climatology(window_days=14).fit(history).forecast(CUTOFF, hours=24)

    # With a perfectly repeating signal the window changes nothing, which is
    # what shows the smoothing is averaging like with like.
    assert narrow["yhat"].to_numpy() == pytest.approx(wide["yhat"].to_numpy(), abs=0.5)


def test_climatology_reports_no_interval() -> None:
    hours = 24 * 400
    history = _history(hours, _diurnal(hours))

    forecast = Climatology().fit(history).forecast(CUTOFF, hours=24)

    # It estimated no spread, so it must not appear to have predicted one.
    assert forecast["yhat_lower"].isna().all()
    assert forecast["yhat_upper"].isna().all()


def test_climatology_needs_an_observation() -> None:
    history = _history(48, [None] * 48)

    with pytest.raises(ValueError, match="at least one observation"):
        Climatology().fit(history)


def test_persistence_repeats_the_last_observed_day() -> None:
    hours = 24 * 5
    values = list(_diurnal(hours))
    history = _history(hours, values)

    forecast = SeasonalPersistence().fit(history).forecast(CUTOFF, hours=48)

    # Hour of day drives the lookup, so the horizon tiles the final day twice.
    first_day = forecast["yhat"].to_numpy()[:24]
    second_day = forecast["yhat"].to_numpy()[24:]
    assert first_day == pytest.approx(second_day)


def test_persistence_fills_a_gap_from_the_previous_day() -> None:
    hours = 24 * 3
    values = list(_diurnal(hours))
    # Blank the final hour, whose slot must come from the day before rather
    # than shifting the whole profile along by one.
    values[-1] = None
    history = _history(hours, values)

    forecast = SeasonalPersistence().fit(history).forecast(CUTOFF, hours=24)

    expected_hour = (CUTOFF - pd.Timedelta(hours=1)).hour
    recovered = forecast.loc[forecast[pf.DS].dt.hour == expected_hour, "yhat"].iloc[0]
    assert recovered == pytest.approx(values[-25])


def test_persistence_needs_an_observation_in_the_lookback() -> None:
    history = _history(48, [None] * 48)

    with pytest.raises(ValueError, match="lookback window"):
        SeasonalPersistence().fit(history)


def test_forecast_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Climatology().forecast(CUTOFF)
    with pytest.raises(RuntimeError, match="fit"):
        SeasonalPersistence().forecast(CUTOFF)
