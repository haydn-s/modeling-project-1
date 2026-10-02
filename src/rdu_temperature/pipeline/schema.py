"""Canonical column vocabulary and unit conversions for cleaned weather data.

Each raw source publishes its own column names and units. This module defines
the single vocabulary every source is converted into, so downstream code can
treat the sources identically. Conversions operate on :class:`pandas.Series`
and propagate missing values rather than filling them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = "source"
STATION_ID = "station_id"
TIMESTAMP_UTC = "timestamp_utc"

IDENTITY_COLUMNS: tuple[str, ...] = (SOURCE, STATION_ID, TIMESTAMP_UTC)

TEMPERATURE_C = "temperature_c"
DEWPOINT_C = "dewpoint_c"
RELATIVE_HUMIDITY_PCT = "relative_humidity_pct"
WIND_SPEED_MS = "wind_speed_ms"
WIND_DIRECTION_DEG = "wind_direction_deg"
WIND_GUST_MS = "wind_gust_ms"
PRESSURE_HPA = "pressure_hpa"
SEA_LEVEL_PRESSURE_HPA = "sea_level_pressure_hpa"
PRECIPITATION_MM = "precipitation_mm"
CLOUD_COVER_PCT = "cloud_cover_pct"
VISIBILITY_M = "visibility_m"
SHORTWAVE_RADIATION_WM2 = "shortwave_radiation_wm2"

MEASUREMENT_COLUMNS: tuple[str, ...] = (
    TEMPERATURE_C,
    DEWPOINT_C,
    RELATIVE_HUMIDITY_PCT,
    WIND_SPEED_MS,
    WIND_DIRECTION_DEG,
    WIND_GUST_MS,
    PRESSURE_HPA,
    SEA_LEVEL_PRESSURE_HPA,
    PRECIPITATION_MM,
    CLOUD_COVER_PCT,
    VISIBILITY_M,
    SHORTWAVE_RADIATION_WM2,
)

CANONICAL_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *MEASUREMENT_COLUMNS)

# The target series is derived rather than observed, so it carries its own
# columns: the local clock hour the project is scored on, and the source each
# reading came from.
TIMESTAMP_LOCAL = "timestamp_local"
TEMPERATURE_SOURCE = "temperature_source"

TARGET_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_UTC,
    TIMESTAMP_LOCAL,
    STATION_ID,
    TEMPERATURE_C,
    TEMPERATURE_SOURCE,
)

# A forecast carries two clocks where an observation carries one: when the run
# that produced it started, and which hour it describes. Both have to survive
# into the cleaned data. The initialisation time is what makes a prediction for
# the forecast period legitimate to train on -- a run that started before the
# cutoff knew nothing after it -- and collapsing the pair to a single timestamp
# would throw that guarantee away. Lead time is carried too, although it is
# derivable, because forecast skill is read against it.
INIT_TIME_UTC = "init_time_utc"
VALID_TIME_UTC = "valid_time_utc"
LEAD_HOURS = "lead_hours"

FORECAST_IDENTITY_COLUMNS: tuple[str, ...] = (
    SOURCE,
    STATION_ID,
    INIT_TIME_UTC,
    VALID_TIME_UTC,
    LEAD_HOURS,
)

FORECAST_COLUMNS: tuple[str, ...] = (
    *FORECAST_IDENTITY_COLUMNS,
    *MEASUREMENT_COLUMNS,
)

# METAR sky cover abbreviations expressed as a percentage of the sky. Reported
# layers are cumulative, so a layer's code describes total cover up to and
# including that layer. Ranged codes use the midpoint of their okta range.
SKY_COVER_PCT: dict[str, float] = {
    "SKC": 0.0,  # sky clear, human observer
    "CLR": 0.0,  # clear below 12,000 ft, automated station
    "NCD": 0.0,  # no cloud detected
    "NSC": 0.0,  # no significant cloud
    "FEW": 18.75,  # 1-2 oktas
    "SCT": 43.75,  # 3-4 oktas
    "BKN": 75.0,  # 5-7 oktas
    "OVC": 100.0,  # 8 oktas
    "VV": 100.0,  # sky obscured, vertical visibility only
}

KNOTS_TO_METERS_PER_SECOND = 1852.0 / 3600.0
INCHES_OF_MERCURY_TO_HECTOPASCALS = 33.863886667
INCHES_TO_MILLIMETERS = 25.4
MILES_TO_METERS = 1609.344
KILOMETERS_TO_METERS = 1000.0
KELVIN_AT_ZERO_CELSIUS = 273.15
SECONDS_PER_HOUR = 3600.0


def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    """Convert degrees Fahrenheit to degrees Celsius."""
    return (values - 32.0) * 5.0 / 9.0


def kelvin_to_celsius(values: pd.Series) -> pd.Series:
    """Convert kelvin to degrees Celsius."""
    return values - KELVIN_AT_ZERO_CELSIUS


def precipitation_rate_to_millimetres_per_hour(values: pd.Series) -> pd.Series:
    """Convert a precipitation rate in kg/m^2/s to millimetres per hour.

    One kilogram of water per square metre is one millimetre deep, so the
    conversion is the seconds in an hour and nothing else.
    """
    return values * SECONDS_PER_HOUR


def wind_components_to_speed(eastward: pd.Series, northward: pd.Series) -> pd.Series:
    """Combine eastward and northward wind components into a speed."""
    return pd.Series(
        np.hypot(eastward.to_numpy(), northward.to_numpy()), index=eastward.index
    )


def wind_components_to_direction(
    eastward: pd.Series, northward: pd.Series
) -> pd.Series:
    """Return the meteorological direction the wind blows *from*, in degrees.

    Gridded models publish wind as components pointing in the direction of
    travel; observations report the bearing the wind arrives from. Converting
    to the observation convention is what lets a model column sit in the same
    vocabulary as a station column, so 270 degrees means a westerly in both.
    """
    bearing = 270.0 - np.degrees(np.arctan2(northward.to_numpy(), eastward.to_numpy()))
    return pd.Series(np.mod(bearing, 360.0), index=eastward.index)


def knots_to_meters_per_second(values: pd.Series) -> pd.Series:
    """Convert knots to meters per second."""
    return values * KNOTS_TO_METERS_PER_SECOND


def inches_of_mercury_to_hectopascals(values: pd.Series) -> pd.Series:
    """Convert inches of mercury to hectopascals."""
    return values * INCHES_OF_MERCURY_TO_HECTOPASCALS


def inches_to_millimeters(values: pd.Series) -> pd.Series:
    """Convert inches to millimeters."""
    return values * INCHES_TO_MILLIMETERS


def miles_to_meters(values: pd.Series) -> pd.Series:
    """Convert statute miles to meters."""
    return values * MILES_TO_METERS


def kilometers_to_meters(values: pd.Series) -> pd.Series:
    """Convert kilometers to meters."""
    return values * KILOMETERS_TO_METERS


def sky_cover_to_percent(codes: pd.Series) -> pd.Series:
    """Map METAR sky cover abbreviations onto a percentage of the sky.

    Unrecognized abbreviations become missing values rather than zero, so that
    an unreadable report is never mistaken for a clear sky.
    """
    normalized = codes.astype("string").str.strip().str.upper()
    return normalized.map(SKY_COVER_PCT).astype("Float64").astype("float64")
