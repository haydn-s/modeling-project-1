from __future__ import annotations

import pandas as pd

from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.quality import Range, Screener, flatline_mask


def _panel(**columns: list[float | None]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(
        {
            schema.SOURCE: ["noaa_ghcnh"] * length,
            schema.STATION_ID: ["KRDU"] * length,
            schema.TIMESTAMP_UTC: pd.date_range(
                "2026-01-01T00:00Z", periods=length, freq="h"
            ),
            **columns,
        }
    )


def test_range_ignores_missing_readings() -> None:
    limit = Range(0.0, 100.0)

    outside = limit.outside(pd.Series([-1.0, 0.0, 50.0, 100.0, 101.0, None]))

    assert outside.tolist() == [True, False, False, False, True, False]


def test_flatline_mask_needs_an_unbroken_run() -> None:
    values = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0])
    broken = pd.Series([1.0, 1.0, None, 1.0, 1.0])

    assert flatline_mask(values, 3).tolist() == [True, True, True, False, False]
    assert not flatline_mask(values, 4).any()
    # An outage must not join two stretches that happen to share a value.
    assert not flatline_mask(broken, 3).any()


def test_screening_masks_readings_rather_than_rows() -> None:
    panel = _panel(
        temperature_c=[20.0, 999.0, 22.0],
        wind_speed_ms=[1.0, 2.0, 3.0],
    )

    result = Screener().screen(panel)

    assert len(result.frame) == 3
    assert result.frame[schema.TEMPERATURE_C].isna().tolist() == [False, True, False]
    # The rest of the flagged observation survives.
    assert result.frame[schema.WIND_SPEED_MS].tolist() == [1.0, 2.0, 3.0]


def test_screening_masks_the_dew_point_not_the_temperature() -> None:
    panel = _panel(temperature_c=[20.0, 20.0, 20.0], dewpoint_c=[15.0, 20.2, 25.0])

    result = Screener().screen(panel)

    # A dew point a touch above the temperature is rounding, not an error.
    assert result.frame[schema.DEWPOINT_C].isna().tolist() == [False, False, True]
    assert result.frame[schema.TEMPERATURE_C].tolist() == [20.0, 20.0, 20.0]


def test_screening_masks_a_stuck_sensor() -> None:
    panel = _panel(temperature_c=[5.0] * 30)

    result = Screener(flatline_hours=24).screen(panel)

    assert result.frame[schema.TEMPERATURE_C].isna().all()
    assert result.report["rule"].tolist() == ["flatline"]


def test_screening_tolerates_whole_degree_stations() -> None:
    # KJNX reports whole degrees, so a long flat stretch is quantization.
    panel = _panel(temperature_c=[2.0] * 21 + [3.0, 4.0, 5.0])

    result = Screener(flatline_hours=24).screen(panel)

    assert result.frame[schema.TEMPERATURE_C].notna().all()
    assert result.masked == 0


def test_screening_report_counts_each_rule_by_station() -> None:
    panel = _panel(temperature_c=[20.0, 999.0, -999.0], dewpoint_c=[30.0, 1.0, 1.0])

    result = Screener().screen(panel)
    report = result.report

    assert list(report.columns) == [
        "rule",
        schema.SOURCE,
        schema.STATION_ID,
        "column",
        "masked",
    ]
    temperature = report[report["column"] == schema.TEMPERATURE_C]
    assert temperature["masked"].sum() == 2
    assert temperature[schema.STATION_ID].tolist() == ["KRDU"]
    assert result.masked == int(report["masked"].sum())


def test_screening_leaves_clean_data_untouched() -> None:
    panel = _panel(temperature_c=[20.0, 21.0, 22.0], dewpoint_c=[15.0, 16.0, 17.0])

    result = Screener().screen(panel)

    assert result.masked == 0
    assert result.report.empty
    pd.testing.assert_frame_equal(result.frame, panel)
