"""Shape the cleaned target series into the frame Prophet fits on.

Prophet takes two columns — ``ds`` for the timestamp and ``y`` for the value —
so feature engineering here means shaping time itself rather than adding
covariates. Three decisions carry the module.

``ds`` holds UTC wall time. Prophet rejects timezone-aware timestamps outright,
so the offset has to come off; the question is which clock is left behind.
Local time would put an hour-wide discontinuity into the diurnal cycle at every
daylight saving transition, and the training window spans ten of them. UTC runs
at a fixed offset from solar time all year, which leaves one continuous daily
shape to fit. Predictions are converted back to local time for scoring.

The diurnal cycle changes shape through the year. The Piedmont swings much
wider between night and afternoon in spring and autumn than it does under
summer humidity or winter cloud, and one daily seasonality can only average
those shapes together. The season flags built here let Prophet fit a separate
daily curve per season instead.

Measured over the seasonal backtest, that split does not pay for itself: it
moves mean absolute error by less than 0.02 degrees against a single global
daily curve, and loses on three folds of four. The diurnal shape is simply not
where the error lives. A fortnight forecast is dominated by whether the model
guessed the period's synoptic pattern, which shows up as a bias of two to
three degrees and swamps any refinement of the daily curve. The flags are kept
because the comparison is worth stating in the writeup, and because they cost
nothing once the frame is built.

Hours with no observation stay missing. The pipeline treats the target as a
measurement rather than an estimate, and Prophet drops null ``y`` rows when it
fits, so the gap costs a handful of rows and invents nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from rdu_temperature.pipeline import schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "weather_sources.json"
DEFAULT_TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "rdu_hourly_target.parquet"

DS = "ds"
Y = "y"

# Meteorological seasons, keyed by the condition column handed to Prophet.
SEASON_MONTHS: Mapping[str, tuple[int, ...]] = {
    "is_winter": (12, 1, 2),
    "is_spring": (3, 4, 5),
    "is_summer": (6, 7, 8),
    "is_autumn": (9, 10, 11),
}

SEASON_COLUMNS: tuple[str, ...] = tuple(SEASON_MONTHS)
PROPHET_COLUMNS: tuple[str, ...] = (DS, Y, *SEASON_COLUMNS)

# The forecast window runs from the data cutoff through September 30 at 11 p.m.
# local time: fourteen days, and the horizon every backtest fold reproduces.
FORECAST_HOURS = 336


def add_season_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach one boolean column per meteorological season, keyed off ``ds``.

    Seasons are taken from the UTC month so that the flags agree with ``ds``.
    A boundary hour can therefore sit one season over from its local date,
    which is immaterial against a three-month block.

    Prophet requires a condition column on the future frame as well as the
    history, so forecasting calls this on the frame it is about to predict.
    """
    months = frame[DS].dt.month
    return frame.assign(
        **{name: months.isin(span) for name, span in SEASON_MONTHS.items()}
    )


def build(target: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned target series as a Prophet frame.

    The full hourly grid is kept, missing observations included, so the frame
    stays a faithful picture of what was measured. Prophet discards the null
    rows itself at fit time.
    """
    timestamps = pd.to_datetime(target[schema.TIMESTAMP_UTC], utc=True)
    frame = pd.DataFrame(
        {
            DS: timestamps.dt.tz_localize(None),
            Y: pd.to_numeric(target[schema.TEMPERATURE_C], errors="coerce"),
        }
    )
    return (
        add_season_flags(frame)
        .sort_values(DS, kind="stable")
        .reset_index(drop=True)
        .loc[:, list(PROPHET_COLUMNS)]
    )


def load(path: Path = DEFAULT_TARGET_PATH) -> pd.DataFrame:
    """Read the cleaned target parquet and shape it for Prophet."""
    if not path.exists():
        raise FileNotFoundError(
            f"No cleaned target at {path}; run rdu_temperature.pipeline.clean_weather."
        )
    return build(pd.read_parquet(path))


def future_frame(cutoff: pd.Timestamp, hours: int = FORECAST_HOURS) -> pd.DataFrame:
    """Return the hours to predict, starting at ``cutoff`` and carrying flags.

    The cutoff is the first hour forecast, not the last hour observed: the
    ingestion window is half-open, so the data ends where the horizon begins.
    """
    timestamps = pd.date_range(cutoff, periods=hours, freq="h", name=DS)
    return add_season_flags(pd.DataFrame({DS: timestamps}))


def forecast_cutoff(config_path: Path = DEFAULT_CONFIG_PATH) -> pd.Timestamp:
    """Return the first forecast hour as naive UTC, matching ``ds``.

    The pipeline's ingestion window ends exactly where the forecast period
    starts, so the cutoff is read from the same configuration rather than
    restated here and left to drift.
    """
    with config_path.open(encoding="utf-8") as config_file:
        history = json.load(config_file)["history"]
    return pd.Timestamp(history["end_utc"]).tz_convert("UTC").tz_localize(None)


def to_local(
    timestamps: pd.Series, local_timezone: str = "America/New_York"
) -> pd.Series:
    """Convert naive UTC ``ds`` values back to the local clock used for scoring."""
    return timestamps.dt.tz_localize("UTC").dt.tz_convert(local_timezone)
