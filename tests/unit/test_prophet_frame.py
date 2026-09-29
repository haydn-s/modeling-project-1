from __future__ import annotations

import pandas as pd
import pytest

from rdu_temperature.evaluation.backtest import (
    FoldResult,
    seasonal_cutoffs,
    split,
    summarize,
)
from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.pipeline import schema


def _target(
    start: str = "2026-01-01T00:00Z", hours: int = 48, **columns
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame(
        {
            schema.TIMESTAMP_UTC: timestamps,
            schema.TEMPERATURE_C: columns.get(
                "temperature_c", [float(hour) for hour in range(hours)]
            ),
        }
    )


def _frame(start: str = "2021-09-17T04:00Z", hours: int = 48) -> pd.DataFrame:
    return pf.build(_target(start, hours))


def test_ds_is_naive_utc_wall_time() -> None:
    frame = pf.build(_target("2026-07-01T12:00Z", hours=3))

    # Prophet rejects a tz-aware ds, so the offset must be gone while the
    # clock left behind is still UTC rather than local.
    assert frame[pf.DS].dt.tz is None
    assert frame[pf.DS].iloc[0] == pd.Timestamp("2026-07-01 12:00:00")


def test_season_flags_are_exclusive_and_exhaustive() -> None:
    frame = pf.build(_target("2026-01-01T00:00Z", hours=24 * 400))

    flags = frame.loc[:, list(pf.SEASON_COLUMNS)]

    assert (flags.sum(axis=1) == 1).all()
    assert flags.to_numpy().dtype == bool


def test_season_flags_follow_the_meteorological_calendar() -> None:
    frame = pf.build(_target("2026-03-01T00:00Z", hours=1))
    winter = pf.build(_target("2026-02-28T00:00Z", hours=1))

    assert bool(frame["is_spring"].iloc[0])
    assert bool(winter["is_winter"].iloc[0])


def test_missing_observations_are_kept_rather_than_interpolated() -> None:
    target = _target(hours=4, temperature_c=[10.0, None, None, 13.0])

    frame = pf.build(target)

    # The target is a measurement; a gap stays a gap for Prophet to drop.
    assert len(frame) == 4
    assert frame[pf.Y].isna().tolist() == [False, True, True, False]


def test_build_sorts_by_timestamp() -> None:
    target = _target(hours=3).iloc[::-1]

    frame = pf.build(target)

    assert frame[pf.DS].is_monotonic_increasing


def test_future_frame_carries_the_condition_columns() -> None:
    future = pf.future_frame(pd.Timestamp("2026-09-17 04:00:00"), hours=336)

    # Prophet needs every condition column present on the frame it predicts.
    assert set(pf.SEASON_COLUMNS) <= set(future.columns)
    assert len(future) == 336
    assert future[pf.DS].iloc[0] == pd.Timestamp("2026-09-17 04:00:00")
    assert future[pf.DS].iloc[-1] == pd.Timestamp("2026-10-01 03:00:00")


def test_future_frame_covers_the_scored_local_window() -> None:
    future = pf.future_frame(pd.Timestamp("2026-09-17 04:00:00"))

    local = pf.to_local(future[pf.DS])

    # The project is scored from midnight on the 17th through 11 p.m. on the
    # 30th, local time.
    assert str(local.iloc[0]) == "2026-09-17 00:00:00-04:00"
    assert str(local.iloc[-1]) == "2026-09-30 23:00:00-04:00"


def test_load_reports_a_missing_target(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="clean_weather"):
        pf.load(tmp_path / "absent.parquet")


def test_split_is_half_open_at_the_cutoff() -> None:
    frame = _frame(hours=6)
    cutoff = frame[pf.DS].iloc[3]

    history, held_out = split(frame, cutoff)

    assert len(history) == 3
    assert len(held_out) == 3
    # The cutoff hour is the first hour forecast, never the last hour trained.
    assert history[pf.DS].max() < cutoff
    assert held_out[pf.DS].min() == cutoff


def test_seasonal_cutoffs_land_on_the_forecast_anniversary() -> None:
    frame = _frame(hours=24 * 365 * 5)
    cutoff = frame[pf.DS].max() + pd.Timedelta(hours=1)

    folds = seasonal_cutoffs(frame, cutoff)

    assert folds == sorted(folds)
    assert all(fold.month == cutoff.month and fold.day == cutoff.day for fold in folds)


def test_seasonal_cutoffs_require_a_full_horizon_and_a_year_of_training() -> None:
    frame = _frame(hours=24 * 400)
    cutoff = frame[pf.DS].max() + pd.Timedelta(hours=1)

    folds = seasonal_cutoffs(frame, cutoff)

    # 400 days cannot hold a year of training plus a 336-hour horizon.
    assert folds == []


def test_seasonal_cutoffs_never_reach_the_forecast_period() -> None:
    frame = _frame(hours=24 * 365 * 5)
    cutoff = frame[pf.DS].max() + pd.Timedelta(hours=1)

    folds = seasonal_cutoffs(frame, cutoff)

    horizon = pd.Timedelta(hours=pf.FORECAST_HOURS)
    assert all(fold + horizon <= cutoff for fold in folds)


def _fold(cutoff: str, errors: list[float]) -> FoldResult:
    timestamps = pd.date_range(cutoff, periods=len(errors), freq="h")
    truth = [20.0] * len(errors)
    predictions = pd.DataFrame(
        {
            pf.DS: timestamps,
            "yhat": [t + e for t, e in zip(truth, errors)],
            "yhat_lower": [t - 5.0 for t in truth],
            "yhat_upper": [t + 5.0 for t in truth],
            pf.Y: truth,
        }
    )
    return FoldResult(
        cutoff=pd.Timestamp(cutoff), training_hours=8760, predictions=predictions
    )


def test_fold_metrics_separate_bias_from_magnitude() -> None:
    metrics = _fold("2025-09-17 04:00:00", [2.0, 2.0, 2.0, 2.0]).metrics()

    # A uniformly warm forecast carries its whole error as bias.
    assert metrics["mae_c"] == pytest.approx(2.0)
    assert metrics["bias_c"] == pytest.approx(2.0)
    assert metrics["scored_hours"] == 4


def test_fold_metrics_skip_hours_with_no_observation() -> None:
    fold = _fold("2025-09-17 04:00:00", [1.0, 1.0, 1.0, 1.0])
    fold.predictions.loc[1, pf.Y] = None

    metrics = fold.metrics()

    assert metrics["scored_hours"] == 3
    assert metrics["mae_c"] == pytest.approx(1.0)


def test_summary_weights_folds_by_the_hours_they_scored() -> None:
    long_fold = _fold("2024-09-17 04:00:00", [1.0] * 8)
    short_fold = _fold("2025-09-17 04:00:00", [5.0] * 2)

    summary = summarize([long_fold, short_fold])
    total = summary.iloc[-1]

    assert total["cutoff"] == "all"
    assert total["scored_hours"] == 10
    # (8 * 1 + 2 * 5) / 10, not the unweighted mean of 3.0.
    assert total["mae_c"] == pytest.approx(1.8)
