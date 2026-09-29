from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.evaluation.backtest import (
    MODELS,
    Backtest,
    FoldResult,
    rolling_cutoffs,
    summarize,
)
from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.pipeline import schema

START = "2021-09-17T04:00Z"


def _frame(hours: int) -> pd.DataFrame:
    timestamps = pd.date_range(START, periods=hours, freq="h")
    return pf.build(
        pd.DataFrame(
            {
                schema.TIMESTAMP_UTC: timestamps,
                schema.TEMPERATURE_C: np.linspace(10.0, 20.0, hours),
            }
        )
    )


def test_rolling_cutoffs_step_through_the_history() -> None:
    frame = _frame(24 * 365 * 3)

    folds = rolling_cutoffs(frame, step_days=14)

    assert folds == sorted(folds)
    assert len(folds) > 40
    gaps = {(b - a).days for a, b in pairwise(folds)}
    assert gaps == {14}


def test_rolling_cutoffs_leave_a_full_year_of_training() -> None:
    frame = _frame(24 * 365 * 3)
    first = frame[pf.DS].min()

    folds = rolling_cutoffs(frame)

    assert (folds[0] - first).days == 365


def test_rolling_cutoffs_leave_a_full_horizon_to_score() -> None:
    frame = _frame(24 * 365 * 3)
    frame_end = frame[pf.DS].max() + pd.Timedelta(hours=1)

    folds = rolling_cutoffs(frame)

    horizon = pd.Timedelta(hours=pf.FORECAST_HOURS)
    assert all(fold + horizon <= frame_end for fold in folds)


def test_rolling_cutoffs_are_empty_when_history_is_too_short() -> None:
    assert rolling_cutoffs(_frame(24 * 300)) == []


def test_rolling_cutoffs_give_far_more_folds_than_seasonal() -> None:
    frame = _frame(24 * 365 * 5)

    # The whole point of the rolling set: enough folds to resolve an effect
    # that four September folds cannot.
    assert len(rolling_cutoffs(frame)) > 20 * 4


def _fold(errors: list[float], *, bounded: bool) -> FoldResult:
    cutoff = pd.Timestamp("2025-09-17 04:00:00")
    timestamps = pd.date_range(cutoff, periods=len(errors), freq="h")
    truth = [20.0] * len(errors)
    predictions = pd.DataFrame(
        {
            pf.DS: timestamps,
            "yhat": [t + e for t, e in zip(truth, errors)],
            "yhat_lower": [t - 5.0 for t in truth] if bounded else np.nan,
            "yhat_upper": [t + 5.0 for t in truth] if bounded else np.nan,
            pf.Y: truth,
        }
    )
    return FoldResult(cutoff=cutoff, training_hours=8760, predictions=predictions)


def test_coverage_is_missing_when_a_model_predicted_no_interval() -> None:
    metrics = _fold([1.0] * 4, bounded=False).metrics()

    # Absent bounds must not score as nought per cent, which would read as a
    # badly calibrated interval rather than no interval at all.
    assert np.isnan(metrics["interval_coverage"])
    assert metrics["mae_c"] == pytest.approx(1.0)


def test_coverage_is_reported_when_bounds_exist() -> None:
    metrics = _fold([1.0] * 4, bounded=True).metrics()

    assert metrics["interval_coverage"] == pytest.approx(1.0)


def test_summary_total_skips_folds_missing_a_metric() -> None:
    summary = summarize([_fold([1.0] * 4, bounded=False)])

    total = summary.iloc[-1]

    # A weighted mean over nothing is missing, not zero.
    assert np.isnan(total["interval_coverage"])
    assert total["mae_c"] == pytest.approx(1.0)


def test_backtest_runs_any_model_the_factory_returns() -> None:
    frame = _frame(24 * 400)
    cutoff = frame[pf.DS].min() + pd.Timedelta(days=366)

    result = Backtest(MODELS["persistence"]).run_fold(frame, cutoff)

    assert len(result.predictions) == pf.FORECAST_HOURS
    assert result.training_hours == 366 * 24
    # The held-out truth is joined on, so the fold is scorable.
    assert result.scored[pf.Y].notna().all()


def test_every_registered_model_fits_and_forecasts() -> None:
    frame = _frame(24 * 400)
    cutoff = frame[pf.DS].min() + pd.Timedelta(days=366)

    for name, factory in MODELS.items():
        result = Backtest(factory).run_fold(frame, cutoff)

        assert len(result.predictions) == pf.FORECAST_HOURS, name
        assert result.predictions["yhat"].notna().all(), name
