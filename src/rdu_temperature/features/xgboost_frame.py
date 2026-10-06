"""Turn matched GFS forecasts into the numeric matrix XGBoost consumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdu_temperature.features import forecast_covariates
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.25

GFS_TEMPERATURE = f"{forecast_covariates.PREFIX}{schema.TEMPERATURE_C}"
GFS_DEWPOINT = f"{forecast_covariates.PREFIX}{schema.DEWPOINT_C}"
GFS_WIND_SPEED = f"{forecast_covariates.PREFIX}{schema.WIND_SPEED_MS}"
GFS_WIND_DIRECTION = f"{forecast_covariates.PREFIX}{schema.WIND_DIRECTION_DEG}"
GFS_CLOUD_COVER = f"{forecast_covariates.PREFIX}{schema.CLOUD_COVER_PCT}"
GFS_PRECIPITATION = f"{forecast_covariates.PREFIX}{schema.PRECIPITATION_MM}"

REQUIRED_COLUMNS: tuple[str, ...] = (
    schema.INIT_TIME_UTC,
    schema.VALID_TIME_UTC,
    schema.LEAD_HOURS,
    INTERPOLATED,
    GFS_TEMPERATURE,
    GFS_DEWPOINT,
    GFS_WIND_SPEED,
    GFS_WIND_DIRECTION,
    GFS_CLOUD_COVER,
    GFS_PRECIPITATION,
)


@dataclass(frozen=True)
class XGBoostTrainingData:
    """Numeric features, measured target, and the run behind every row."""

    features: pd.DataFrame
    target: pd.Series
    run_times: pd.Series


@dataclass(frozen=True)
class XGBoostForecastData:
    """Numeric features and timestamps for the final prediction window."""

    features: pd.DataFrame
    valid_times: pd.Series


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build small, interpretable features available at forecast time.

    Wind direction and clock/calendar position are circular, so each becomes a
    sine/cosine pair. This prevents 359 and 1 degrees, or 23:00 and 00:00, from
    looking far apart to a tree only because their raw numbers wrap around.
    """
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"GFS frame is missing required columns: {missing}.")

    valid_times = pd.to_datetime(frame[schema.VALID_TIME_UTC], utc=True)
    wind_radians = np.deg2rad(frame[GFS_WIND_DIRECTION].astype(float))
    hour_radians = 2 * np.pi * valid_times.dt.hour / HOURS_PER_DAY
    year_radians = 2 * np.pi * valid_times.dt.dayofyear / DAYS_PER_YEAR

    features = pd.DataFrame(
        {
            GFS_TEMPERATURE: frame[GFS_TEMPERATURE].astype(float).to_numpy(),
            GFS_DEWPOINT: frame[GFS_DEWPOINT].astype(float).to_numpy(),
            GFS_WIND_SPEED: frame[GFS_WIND_SPEED].astype(float).to_numpy(),
            GFS_CLOUD_COVER: frame[GFS_CLOUD_COVER].astype(float).to_numpy(),
            GFS_PRECIPITATION: frame[GFS_PRECIPITATION].astype(float).to_numpy(),
            "gfs_wind_direction_sin": np.sin(wind_radians).to_numpy(),
            "gfs_wind_direction_cos": np.cos(wind_radians).to_numpy(),
            "forecast_lead_hour": frame[schema.LEAD_HOURS].astype(float).to_numpy(),
            "gfs_interpolated": frame[INTERPOLATED].astype(int).to_numpy(),
            "target_hour_sin": np.sin(hour_radians).to_numpy(),
            "target_hour_cos": np.cos(hour_radians).to_numpy(),
            "target_day_of_year_sin": np.sin(year_radians).to_numpy(),
            "target_day_of_year_cos": np.cos(year_radians).to_numpy(),
        },
        index=pd.DatetimeIndex(valid_times.dt.tz_localize(None), name="valid_time_utc"),
    )
    if features.isna().any(axis=None):
        columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"XGBoost features contain missing values: {columns}.")
    return features


def build_training_data(
    panel: pd.DataFrame,
    target: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> XGBoostTrainingData:
    """Pair historical forecasts with observations and build X/y."""
    pairs = forecast_covariates.training_pairs(panel, target, cutoff)
    if pairs.empty:
        raise ValueError("No historical GFS/temperature pairs are available.")

    features = build_features(pairs)
    observed = pd.Series(
        pairs[forecast_covariates.OBSERVED_TEMPERATURE_C].to_numpy(dtype=float),
        index=features.index,
        name=schema.TEMPERATURE_C,
    )
    run_times = pd.Series(
        pd.to_datetime(pairs[schema.INIT_TIME_UTC]).to_numpy(),
        index=features.index,
        name=schema.INIT_TIME_UTC,
    )
    return XGBoostTrainingData(features, observed, run_times)


def build_forecast_data(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    hours: int,
) -> XGBoostForecastData:
    """Build the same feature contract for a future GFS window."""
    covariates = forecast_covariates.forecast_covariates(panel, cutoff, hours)
    features = build_features(covariates)
    valid_times = pd.Series(
        pd.to_datetime(covariates[schema.VALID_TIME_UTC]).to_numpy(),
        index=features.index,
        name=schema.VALID_TIME_UTC,
    )
    return XGBoostForecastData(features, valid_times)
