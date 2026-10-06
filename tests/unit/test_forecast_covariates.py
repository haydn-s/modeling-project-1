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
    """The target as the cleaning pipeline writes it: aware, on UTC.

    This fixture was naive until real data proved otherwise, and the join
    silently passed because both sides happened to agree. It is aware here so
    that the test exercises the conversion the real parquet requires.
    """
    return pd.DataFrame(
        {
            schema.TIMESTAMP_UTC: pd.date_range(
                pd.Timestamp(first), periods=hours, freq="h", tz="UTC"
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


def test_covered_cutoffs_keep_only_fully_covered_folds() -> None:
    covered = pd.Timestamp("2025-09-17 04:00:00")
    panel = _panel("2025-09-16 12:00", "2025-09-17 04:00", 24)
    uncovered = pd.Timestamp("2024-09-17 04:00:00")

    kept = fc.covered_cutoffs(panel, [uncovered, covered], hours=24)

    # A model cannot score a fold it has no forecast for, and a paired
    # comparison needs every model on the same folds.
    assert kept == [covered]


def test_covered_cutoffs_reject_a_partial_horizon() -> None:
    cutoff = pd.Timestamp("2025-09-17 04:00:00")
    # Twelve hours of covariate against a twenty-four hour horizon.
    panel = _panel("2025-09-16 12:00", "2025-09-17 04:00", 12)

    assert fc.covered_cutoffs(panel, [cutoff], hours=24) == []


def test_covered_cutoffs_are_empty_without_a_panel() -> None:
    empty = _panel("2026-09-16 12:00", "2026-09-17 04:00", 0)

    assert fc.covered_cutoffs(empty, [CUTOFF], hours=24) == []


def _observed(first: str, hours: int) -> pd.DataFrame:
    """A Prophet frame of observations, as the backtest hands one over."""
    return pf.add_season_flags(
        pd.DataFrame(
            {
                pf.DS: pd.date_range(pd.Timestamp(first), periods=hours, freq="h"),
                pf.Y: [18.0] * hours,
            }
        )
    )


def test_covariate_history_counts_only_hours_with_both() -> None:
    frame = _observed("2025-01-01 00:00", 24 * 40)
    # A run covering a single day inside the observed stretch.
    panel = _panel("2025-01-09 12:00", "2025-01-10 00:00", 24)
    cutoff = pd.Timestamp("2025-02-10 00:00")

    usable = fc.covariate_history(frame, panel, cutoff)

    # The frame offers 960 hours; the panel covers 24 of them, and a regressor
    # has to exist on every row Prophet fits.
    assert len(usable) == 24
    assert usable["gfs_temperature_c"].notna().all()


def test_covariate_history_excludes_the_forecast_period() -> None:
    frame = _observed("2025-01-01 00:00", 24 * 40)
    panel = _panel("2025-01-09 12:00", "2025-01-10 00:00", 24)
    # A cutoff before the run: nothing it says is legitimately known yet.
    cutoff = pd.Timestamp("2025-01-05 00:00")

    assert fc.covariate_history(frame, panel, cutoff).empty


def test_trainable_cutoffs_require_a_year_of_covariate_span() -> None:
    frame = _observed("2024-01-01 00:00", 24 * 800)
    cutoff = pd.Timestamp("2025-06-01 00:00")
    # Covariates for the fold's own horizon, plus two earlier days a month
    # apart. The fold can be forecast, but a month is not an annual cycle.
    panel = pd.concat(
        [
            _panel("2025-05-31 12:00", "2025-06-01 00:00", 24),
            _panel("2025-04-01 12:00", "2025-04-02 00:00", 24),
            _panel("2025-05-01 12:00", "2025-05-02 00:00", 24),
        ],
        ignore_index=True,
    )

    assert fc.covered_cutoffs(panel, [cutoff], hours=24) == [cutoff]
    assert fc.trainable_cutoffs(frame, panel, [cutoff], hours=24) == []


def test_trainable_cutoffs_keep_a_fold_whose_history_spans_the_year() -> None:
    frame = _observed("2024-01-01 00:00", 24 * 800)
    cutoff = pd.Timestamp("2025-06-01 00:00")
    # The same fold with the same number of covariate hours, the earlier of
    # the two days moved back beyond a year. Only the span changed, which is
    # the whole point of measuring span rather than counting hours.
    panel = pd.concat(
        [
            _panel("2025-05-31 12:00", "2025-06-01 00:00", 24),
            _panel("2024-04-01 12:00", "2024-04-02 00:00", 24),
            _panel("2025-05-01 12:00", "2025-05-02 00:00", 24),
        ],
        ignore_index=True,
    )

    assert fc.trainable_cutoffs(frame, panel, [cutoff], hours=24) == [cutoff]


def test_trainable_cutoffs_drop_a_fold_with_no_history_at_all() -> None:
    frame = _observed("2024-01-01 00:00", 24 * 800)
    cutoff = pd.Timestamp("2025-06-01 00:00")
    # The earliest covered fold: its run covers the horizon and nothing
    # before it, so there is nothing to learn the correction from.
    panel = _panel("2025-05-31 12:00", "2025-06-01 00:00", 24)

    assert fc.trainable_cutoffs(frame, panel, [cutoff], hours=24) == []


def test_load_panel_reports_a_missing_panel(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="clean_gfs"):
        fc.load_panel(tmp_path / "absent.parquet")
