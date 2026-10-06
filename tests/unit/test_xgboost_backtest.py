from __future__ import annotations

import pandas as pd

from rdu_temperature.features import forecast_covariates, xgboost_frame
from rdu_temperature.models.xgboost_backtest import dedupe_cutoff_aware
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED


def _training_data() -> xgboost_frame.XGBoostTrainingData:
    """Two runs covering the same three valid hours."""
    valid = pd.DatetimeIndex(
        [
            "2024-09-14 00:00",
            "2024-09-14 01:00",
            "2024-09-14 02:00",
            "2024-09-14 00:00",
            "2024-09-14 01:00",
            "2024-09-14 02:00",
        ],
        name="valid_time_utc",
    )
    features = pd.DataFrame({"gfs_temperature_c": [20.0] * 6}, index=valid)
    target = pd.Series([21.0] * 6, index=valid)
    run_times = pd.Series(
        pd.to_datetime(["2024-09-13 12:00"] * 3 + ["2024-09-13 18:00"] * 3),
        index=valid,
    )
    return xgboost_frame.XGBoostTrainingData(features, target, run_times)


def test_dedupe_keeps_latest_init_per_valid_hour() -> None:
    deduped = dedupe_cutoff_aware(_training_data())

    assert len(deduped.features) == 3
    assert deduped.features.index.is_unique
    assert deduped.features.index.is_monotonic_increasing
    assert (deduped.run_times == pd.Timestamp("2024-09-13 18:00")).all()
    assert deduped.target.index.equals(deduped.features.index)


def test_training_pairs_never_cross_cutoff() -> None:
    """The leakage guard the backtest adapter rests on."""
    cutoff = pd.Timestamp("2024-09-14 00:00")
    panel = pd.DataFrame(
        {
            schema.INIT_TIME_UTC: pd.to_datetime(
                ["2024-09-13 12:00", "2024-09-14 12:00", "2024-09-13 12:00"]
            ),
            schema.VALID_TIME_UTC: pd.to_datetime(
                ["2024-09-13 18:00", "2024-09-14 18:00", "2024-09-14 06:00"]
            ),
            schema.LEAD_HOURS: [6, 6, 18],
            INTERPOLATED: [False, False, False],
            schema.TEMPERATURE_C: [20.0, 21.0, 22.0],
            schema.DEWPOINT_C: [10.0, 11.0, 12.0],
            schema.WIND_SPEED_MS: [3.0, 3.0, 3.0],
            schema.WIND_DIRECTION_DEG: [350.0, 350.0, 350.0],
            schema.CLOUD_COVER_PCT: [25.0, 25.0, 25.0],
            schema.PRECIPITATION_MM: [0.0, 0.0, 0.0],
        }
    )
    target = pd.DataFrame(
        {
            schema.TIMESTAMP_UTC: pd.to_datetime(
                ["2024-09-13 18:00", "2024-09-14 06:00", "2024-09-14 18:00"], utc=True
            ),
            schema.TEMPERATURE_C: [20.5, 21.5, 22.5],
        }
    )

    pairs = forecast_covariates.training_pairs(panel, target, cutoff)

    # Only the first row is legitimate: the second is initialized after the
    # cutoff and the third scores an hour inside the forecast period.
    assert len(pairs) == 1
    assert (pairs[schema.INIT_TIME_UTC] < cutoff).all()
    assert (pairs[schema.VALID_TIME_UTC] < cutoff).all()
