from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import (
    INTERPOLATED,
    GfsCleaner,
    derive_wind,
    load_runs,
    normalize,
    to_hourly,
)

INIT = pd.Timestamp("2026-09-16 12:00:00")


def _raw(leads: list[int], **columns) -> pd.DataFrame:
    count = len(leads)
    defaults = {
        "TMP": [293.15] * count,
        "DPT": [283.15] * count,
        "UGRD": [1.0] * count,
        "VGRD": [0.0] * count,
        "TCDC": [50.0] * count,
        "PRATE": [0.0] * count,
    }
    defaults.update(columns)
    return pd.DataFrame(
        {
            schema.SOURCE: ["noaa_gfs"] * count,
            schema.STATION_ID: ["KRDU"] * count,
            schema.INIT_TIME_UTC: [INIT] * count,
            schema.VALID_TIME_UTC: [INIT + pd.Timedelta(hours=h) for h in leads],
            schema.LEAD_HOURS: leads,
            **defaults,
        }
    )


def _components(bearing_degrees: float, speed: float = 1.0) -> tuple[float, float]:
    """The components a wind arriving from ``bearing_degrees`` would have."""
    angle = np.radians(270.0 - bearing_degrees)
    return speed * np.cos(angle), speed * np.sin(angle)


def test_normalize_converts_kelvin_to_celsius() -> None:
    frame = normalize(_raw([0], TMP=[300.15], DPT=[290.15]))

    assert frame[schema.TEMPERATURE_C].iloc[0] == pytest.approx(27.0)
    assert frame[schema.DEWPOINT_C].iloc[0] == pytest.approx(17.0)


def test_normalize_converts_precipitation_rate_to_millimetres_per_hour() -> None:
    # One kilogram per square metre per second is 3600 mm in an hour.
    frame = normalize(_raw([0], PRATE=[1.0 / 3600.0]))

    assert frame[schema.PRECIPITATION_MM].iloc[0] == pytest.approx(1.0)


def test_normalize_keeps_both_clocks() -> None:
    frame = normalize(_raw([0, 24]))

    assert frame[schema.INIT_TIME_UTC].nunique() == 1
    assert frame[schema.VALID_TIME_UTC].nunique() == 2
    assert frame[schema.LEAD_HOURS].tolist() == [0, 24]


def test_to_hourly_fills_the_three_hourly_tail() -> None:
    # Published at 0 and 3 hours only, as the product does past 120 hours.
    raw = normalize(_raw([0, 3], TMP=[293.15, 299.15]))

    hourly = to_hourly(raw)

    assert hourly[schema.LEAD_HOURS].tolist() == [0, 1, 2, 3]
    assert hourly[schema.TEMPERATURE_C].tolist() == pytest.approx(
        [20.0, 22.0, 24.0, 26.0]
    )


def test_to_hourly_records_which_hours_were_filled() -> None:
    raw = normalize(_raw([0, 3]))

    hourly = to_hourly(raw)

    # A result that leans on the filled two thirds must be distinguishable
    # from one that does not.
    assert hourly[INTERPOLATED].tolist() == [False, True, True, False]


def test_to_hourly_does_not_extrapolate_past_the_last_lead() -> None:
    raw = normalize(_raw([0, 3]))

    hourly = to_hourly(raw)

    # Extrapolating would invent a forecast the model never issued.
    assert hourly[schema.LEAD_HOURS].max() == 3


def test_to_hourly_keeps_runs_separate() -> None:
    first = normalize(_raw([0, 3]))
    second = normalize(_raw([0, 3]))
    second[schema.INIT_TIME_UTC] = INIT + pd.Timedelta(days=1)
    second[schema.VALID_TIME_UTC] += pd.Timedelta(days=1)

    hourly = to_hourly(pd.concat([first, second], ignore_index=True))

    # Two runs must not be interpolated into one another; each is a separate
    # forecast with its own initialisation.
    assert hourly[schema.INIT_TIME_UTC].nunique() == 2
    assert len(hourly) == 8


def test_wind_is_interpolated_as_components_not_as_a_bearing() -> None:
    east_a, north_a = _components(350.0)
    east_b, north_b = _components(10.0)
    raw = normalize(_raw([0, 2], UGRD=[east_a, east_b], VGRD=[north_a, north_b]))

    panel = derive_wind(to_hourly(raw))

    midpoint = panel.loc[panel[schema.LEAD_HOURS] == 1, schema.WIND_DIRECTION_DEG]
    # Halfway between 350 and 10 degrees is north. Interpolating the bearing
    # itself would average to 180 and report a southerly.
    assert float(midpoint.iloc[0]) == pytest.approx(0.0, abs=0.5)


def test_derive_wind_converts_components_to_the_observation_convention() -> None:
    # A wind blowing toward the east arrives from the west: 270 degrees.
    raw = normalize(_raw([0], UGRD=[5.0], VGRD=[0.0]))

    panel = derive_wind(to_hourly(raw))

    assert panel[schema.WIND_SPEED_MS].iloc[0] == pytest.approx(5.0)
    assert panel[schema.WIND_DIRECTION_DEG].iloc[0] == pytest.approx(270.0)


def test_derive_wind_drops_the_component_columns() -> None:
    panel = derive_wind(to_hourly(normalize(_raw([0]))))

    assert "wind_eastward_ms" not in panel.columns
    assert "wind_northward_ms" not in panel.columns


def test_cleaner_screens_an_implausible_decode() -> None:
    # A misread message would land far outside anything the Piedmont produces.
    raw = _raw([0, 1, 2], TMP=[293.15, 9999.0, 294.15])

    panel, report = GfsCleaner().clean(raw)

    assert panel[schema.TEMPERATURE_C].isna().tolist() == [False, True, False]
    assert int(report["masked"].sum()) == 1


def test_cleaner_keeps_a_settled_forecast() -> None:
    # Forty-eight identical hours is a model predicting settled weather, not a
    # sensor that has stopped reporting, so the flatline rule must not fire.
    raw = _raw(list(range(48)))

    panel, report = GfsCleaner().clean(raw)

    assert panel[schema.TEMPERATURE_C].notna().all()
    assert report.empty


def test_cleaner_orders_the_panel_columns() -> None:
    panel, _ = GfsCleaner().clean(_raw([0, 1]))

    assert list(panel.columns) == [
        schema.SOURCE,
        schema.STATION_ID,
        schema.INIT_TIME_UTC,
        schema.VALID_TIME_UTC,
        schema.LEAD_HOURS,
        schema.TEMPERATURE_C,
        schema.DEWPOINT_C,
        schema.WIND_SPEED_MS,
        schema.WIND_DIRECTION_DEG,
        schema.CLOUD_COVER_PCT,
        schema.PRECIPITATION_MM,
        INTERPOLATED,
    ]


def test_load_runs_reports_an_empty_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest_gfs"):
        load_runs(tmp_path)
