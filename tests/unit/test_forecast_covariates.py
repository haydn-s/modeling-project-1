from __future__ import annotations

import pandas as pd
import pytest

from rdu_temperature.features import forecast_covariates as fc
from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED

CUTOFF = pd.Timestamp("2026-09-17 04:00:00")


def _panel(init: str, first_valid: str, hours: int, offset: float = 0.0):
    init_time = pd.Timestamp(init)
    valid = pd.date_range(pd.Timestamp(first_valid), periods=hours, freq="h")
    return pd.DataFrame(
        {
            schema.SOURCE: "noaa_gfs",
            schema.STATION_ID: "KRDU",
            schema.INIT_TIME_UTC: init_time,
            schema.VALID_TIME_UTC: valid,
            schema.LEAD_HOURS: ((valid - init_time).total_seconds() // 3600).astype(
                int
            ),
            schema.TEMPERATURE_C: [20.0 + offset] * hours,
            schema.DEWPOINT_C: [10.0] * hours,
            schema.WIND_SPEED_MS: [2.0] * hours,
            schema.WIND_DIRECTION_DEG: [180.0] * hours,
            schema.CLOUD_COVER_PCT: [50.0] * hours,
            schema.PRECIPITATION_MM: [0.0] * hours,
            INTERPOLATED: [False] * hours,
        }
    )


def _target(first: str, hours: int, value: float = 18.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.TIMESTAMP_UTC: pd.date_range(
                pd.Timestamp(first), periods=hours, freq="h"
            ),
            schema.TEMPERATURE_C: [value] * hours,
        }
    )


def test_covariates_are_prefixed_so_they_cannot_collide() -> None:
    panel = _panel("2026-09-16 12:00", "2026-09-17 04:00", 48)

    covariates = fc.forecast_covariates(panel, CUTOFF, hours=48)

    assert "gfs_temperature_c" in covariates.columns
    # An unprefixed temperature_c beside an observed one would be a trap.
    assert schema.TEMPERATURE_C not in covariates.columns


def test_covariates_carry_lead_and_the_interpolation_flag() -> None:
    panel = _panel("2026-09-16 12:00", "2026-09-17 04:00", 24)

    covariates = fc.forecast_covariates(panel, CUTOFF, hours=24)

    # A twelve-hour forecast and a twelve-day one are not the same covariate.
    assert covariates[schema.LEAD_HOURS].iloc[0] == 16
    assert INTERPOLATED in covariates.columns


def test_covariates_span_exactly_the_requested_window() -> None:
    panel = _panel("2026-09-16 12:00", "2026-09-17 04:00", 400)

    covariates = fc.forecast_covariates(panel, CUTOFF, hours=336)

    assert len(covariates) == 336
    assert covariates[schema.VALID_TIME_UTC].min() == CUTOFF
    # Half-open: the last hour needing a prediction is one before the end.
    assert covariates[schema.VALID_TIME_UTC].max() == CUTOFF + pd.Timedelta(hours=335)


def test_a_run_initialised_after_the_cutoff_is_not_used() -> None:
    late = _panel("2026-09-18 12:00", "2026-09-19 00:00", 24)

    with pytest.raises(ValueError, match="initialised before"):
        fc.forecast_covariates(late, CUTOFF, hours=24)


def test_the_freshest_legitimate_run_wins_an_overlap() -> None:
    older = _panel("2026-09-15 12:00", "2026-09-17 04:00", 24, offset=0.0)
    newer = _panel("2026-09-16 12:00", "2026-09-17 04:00", 24, offset=5.0)

    covariates = fc.forecast_covariates(
        pd.concat([older, newer], ignore_index=True), CUTOFF, hours=24
    )

    # The later run is the more skilful forecast available at the cutoff.
    assert covariates["gfs_temperature_c"].iloc[0] == pytest.approx(25.0)
    assert covariates[schema.INIT_TIME_UTC].iloc[0] == pd.Timestamp("2026-09-16 12:00")


def test_training_pairs_never_reach_the_forecast_period() -> None:
    # A run before the cutoff whose horizon crosses it, as every real run does.
    panel = _panel("2026-09-16 00:00", "2026-09-16 12:00", 48)
    target = _target("2026-09-16 12:00", 48)

    pairs = fc.training_pairs(panel, target, CUTOFF)

    # An observation at or after the cutoff is the answer being predicted.
    assert len(pairs) == 16
    assert pairs[schema.VALID_TIME_UTC].max() < CUTOFF


def test_training_pairs_drop_hours_with_no_observation() -> None:
    panel = _panel("2026-09-15 00:00", "2026-09-15 06:00", 6)
    target = _target("2026-09-15 06:00", 6)
    target.loc[2, schema.TEMPERATURE_C] = None

    pairs = fc.training_pairs(panel, target, CUTOFF)

    # The target is a measurement; an hour without one cannot be learned from.
    assert len(pairs) == 5


def test_training_pairs_join_forecast_to_the_matching_hour() -> None:
    panel = _panel("2026-09-15 00:00", "2026-09-15 06:00", 3, offset=4.0)
    target = _target("2026-09-15 06:00", 3, value=18.0)

    pairs = fc.training_pairs(panel, target, CUTOFF)

    assert pairs["gfs_temperature_c"].tolist() == pytest.approx([24.0] * 3)
    assert pairs[fc.OBSERVED_TEMPERATURE_C].tolist() == pytest.approx([18.0] * 3)


def test_forecast_error_is_forecast_minus_observed() -> None:
    panel = _panel("2026-09-15 00:00", "2026-09-15 06:00", 3, offset=4.0)
    target = _target("2026-09-15 06:00", 3, value=18.0)
    pairs = fc.training_pairs(panel, target, CUTOFF)

    error = fc.forecast_error(pairs)

    # Positive means too warm, matching the backtest's convention.
    assert error.tolist() == pytest.approx([6.0] * 3)


def test_attaching_to_a_prophet_frame_matches_on_ds() -> None:
    panel = _panel("2026-09-16 12:00", "2026-09-17 04:00", 24)
    covariates = fc.forecast_covariates(panel, CUTOFF, hours=24)
    frame = pf.add_season_flags(
        pd.DataFrame(
            {
                pf.DS: pd.date_range(CUTOFF, periods=24, freq="h"),
                pf.Y: [float("nan")] * 24,
            }
        )
    )

    joined = fc.attach_to_prophet_frame(frame, covariates)

    assert len(joined) == 24
    assert joined["gfs_temperature_c"].notna().all()


def test_attaching_refuses_a_regressor_with_gaps() -> None:
    panel = _panel("2026-09-16 12:00", "2026-09-17 04:00", 12)
    covariates = fc.forecast_covariates(panel, CUTOFF, hours=12)
    frame = pf.add_season_flags(
        pd.DataFrame(
            {
                pf.DS: pd.date_range(CUTOFF, periods=24, freq="h"),
                pf.Y: [float("nan")] * 24,
            }
        )
    )

    # Prophet rejects a regressor with holes at fit time with a far less
    # informative message, so this fails early and says what to do.
    with pytest.raises(ValueError, match="no GFS covariate"):
        fc.attach_to_prophet_frame(frame, covariates)


def test_load_panel_reports_a_missing_panel(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="clean_gfs"):
        fc.load_panel(tmp_path / "absent.parquet")
