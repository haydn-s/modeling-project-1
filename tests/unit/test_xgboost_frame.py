from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.features import xgboost_frame
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED


def _gfs_frame(direction: float = 350.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.INIT_TIME_UTC: [pd.Timestamp("2025-09-16 12:00:00")],
            schema.VALID_TIME_UTC: [pd.Timestamp("2025-09-17 04:00:00")],
            schema.LEAD_HOURS: [16],
            INTERPOLATED: [False],
            xgboost_frame.GFS_TEMPERATURE: [20.0],
            xgboost_frame.GFS_DEWPOINT: [12.0],
            xgboost_frame.GFS_WIND_SPEED: [3.0],
            xgboost_frame.GFS_WIND_DIRECTION: [direction],
            xgboost_frame.GFS_CLOUD_COVER: [25.0],
            xgboost_frame.GFS_PRECIPITATION: [0.0],
        }
    )


def test_build_features_produces_only_complete_numeric_columns() -> None:
    features = xgboost_frame.build_features(_gfs_frame())

    assert features.index.tolist() == [pd.Timestamp("2025-09-17 04:00:00")]
    assert features.columns.is_unique
    assert all(pd.api.types.is_numeric_dtype(features[column]) for column in features)
    assert not features.isna().any(axis=None)
    assert "forecast_lead_hour" in features
    assert xgboost_frame.GFS_WIND_DIRECTION not in features


def test_circular_wind_features_keep_north_bearings_close() -> None:
    almost_north = xgboost_frame.build_features(_gfs_frame(359.0)).iloc[0]
    just_past_north = xgboost_frame.build_features(_gfs_frame(1.0)).iloc[0]

    first = almost_north[["gfs_wind_direction_sin", "gfs_wind_direction_cos"]]
    second = just_past_north[["gfs_wind_direction_sin", "gfs_wind_direction_cos"]]
    assert np.linalg.norm(first - second) < 0.04


def test_build_features_reports_missing_gfs_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        xgboost_frame.build_features(_gfs_frame().drop(columns=[schema.LEAD_HOURS]))
