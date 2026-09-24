"""Normalize the raw weather sources into one canonical hourly vocabulary.

Run from the repository root with:

    python -m rdu_temperature.pipeline.clean_weather

Each source publishes different column names, units, and quality conventions.
The subclasses below express those differences declaratively; the shared base
class handles reading, station identity, and column ordering.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from rdu_temperature.pipeline import schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "weather_sources.json"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# A variable maps onto a canonical column, optionally through a unit conversion.
VariableMap = Mapping[str, tuple[str, Callable[[pd.Series], pd.Series] | None]]

# GHCNh carries an ISD quality code beside each variable. Codes 1 and 5 mean the
# value passed every check; 0, 4, and 9 mean it passed the gross limits check.
# Anything else is suspect or erroneous, and a missing code means no check ran.
ACCEPTED_QUALITY_CODES = frozenset({"0", "1", "4", "5", "9"})
QUALITY_CODE_SUFFIX = "_Quality_Code"

# ECONet grades every observation. Scores above this threshold carry the literal
# string "QCF" in place of a reading, so coercion alone would also drop them.
MAX_ECONET_SCORE = 1


@dataclass(frozen=True)
class StationRegistry:
    """Canonical station identifiers and the per-source aliases mapping to them.

    Airport stations are keyed by ICAO identifier so that the three airport
    sources agree; ECONet stations keep their own identifiers.
    """

    canonical_ids: tuple[str, ...]
    aliases: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> StationRegistry:
        airports = config["airport_stations"]
        econet = config["econet_stations"]
        return cls(
            canonical_ids=tuple(
                [station["icao"] for station in airports]
                + [station["id"] for station in econet]
            ),
            aliases={
                "ghcnh": {s["ghcnh_id"]: s["icao"] for s in airports},
                "iem": {s["iem_id"]: s["icao"] for s in airports},
                "econet": {s["id"]: s["id"] for s in econet},
            },
        )

    @classmethod
    def from_path(cls, config_path: Path) -> StationRegistry:
        with config_path.open(encoding="utf-8") as config_file:
            return cls.from_config(json.load(config_file))

    def resolve(self, values: pd.Series, namespace: str) -> pd.Series:
        """Translate source-specific station identifiers to canonical ones."""
        return values.astype("string").str.strip().map(self.aliases[namespace])


@dataclass(frozen=True)
class HourlyGrid:
    """The half-open hourly UTC index that every aligned source is placed on."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HourlyGrid:
        history = config["history"]
        return cls(
            start=pd.Timestamp(history["start_utc"]).tz_convert("UTC"),
            end=pd.Timestamp(history["end_utc"]).tz_convert("UTC"),
        )

    @classmethod
    def from_path(cls, config_path: Path) -> HourlyGrid:
        with config_path.open(encoding="utf-8") as config_file:
            return cls.from_config(json.load(config_file))

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.start, self.end, freq="h", inclusive="left", name=schema.TIMESTAMP_UTC
        )

    def align(self, frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Place one normalized source on the hourly grid.

        Observation times are floored, so the 23:51 report describes hour 23.
        Where a station reports more than once in an hour the last non-missing
        reading of each variable wins, which keeps corrected reports and the
        routine report closest to the top of the next hour. Every station then
        carries every hour, with absent hours left missing rather than filled.
        """
        measurements = [
            column for column in frame.columns if column in schema.MEASUREMENT_COLUMNS
        ]
        floored = frame.assign(
            **{
                schema.TIMESTAMP_UTC: pd.to_datetime(
                    frame[schema.TIMESTAMP_UTC], utc=True
                ).dt.floor("h")
            }
        ).sort_values(schema.TIMESTAMP_UTC, kind="stable")
        collapsed = floored.groupby(
            [schema.STATION_ID, schema.TIMESTAMP_UTC], as_index=False
        )[measurements].last()

        stations = sorted(collapsed[schema.STATION_ID].unique())
        panel = pd.MultiIndex.from_product(
            [stations, self.index], names=[schema.STATION_ID, schema.TIMESTAMP_UTC]
        )
        aligned = (
            collapsed.set_index([schema.STATION_ID, schema.TIMESTAMP_UTC])
            .reindex(panel)
            .reset_index()
        )
        aligned.insert(0, schema.SOURCE, source_name)
        return aligned


class SourceNormalizer(ABC):
    """Shared normalization lifecycle for one raw source."""

    source_name: ClassVar[str]
    station_namespace: ClassVar[str]
    read_options: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, registry: StationRegistry) -> None:
        self.registry = registry

    def locate(self, input_dir: Path) -> Path:
        """Return the single raw CSV for this source."""
        candidates = sorted((input_dir / self.source_name).glob("weather_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No raw CSV for {self.source_name} beneath {input_dir}."
            )
        return candidates[-1]

    def read(self, input_dir: Path) -> pd.DataFrame:
        return pd.read_csv(
            self.locate(input_dir),
            parse_dates=[schema.TIMESTAMP_UTC],
            low_memory=False,
            **self.read_options,
        )

    @abstractmethod
    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the raw frame expressed in canonical columns and units."""

    def load(self, input_dir: Path) -> pd.DataFrame:
        return self.normalize(self.read(input_dir))

    def align(self, input_dir: Path, grid: HourlyGrid) -> pd.DataFrame:
        return grid.align(self.load(input_dir), self.source_name)

    def _apply_variables(
        self, frame: pd.DataFrame, variables: VariableMap
    ) -> dict[str, pd.Series]:
        columns: dict[str, pd.Series] = {}
        for variable, (canonical, convert) in variables.items():
            if variable not in frame.columns:
                continue
            values = pd.to_numeric(frame[variable], errors="coerce")
            columns[canonical] = convert(values) if convert else values
        return columns

    def _finalize(
        self, frame: pd.DataFrame, columns: Mapping[str, pd.Series], aliases: pd.Series
    ) -> pd.DataFrame:
        """Attach identity columns, drop unknown stations, and order the result."""
        station_ids = self.registry.resolve(aliases, self.station_namespace)
        normalized = pd.DataFrame(
            {
                schema.SOURCE: self.source_name,
                schema.STATION_ID: station_ids,
                schema.TIMESTAMP_UTC: pd.to_datetime(
                    frame[schema.TIMESTAMP_UTC], utc=True
                ),
                **columns,
            }
        )
        normalized = normalized.loc[normalized[schema.STATION_ID].notna()]
        normalized[schema.STATION_ID] = normalized[schema.STATION_ID].astype("object")
        ordered = [c for c in schema.CANONICAL_COLUMNS if c in normalized.columns]
        return (
            normalized.loc[:, ordered]
            .sort_values([schema.STATION_ID, schema.TIMESTAMP_UTC], kind="stable")
            .reset_index(drop=True)
        )


class NoaaGhcnhNormalizer(SourceNormalizer):
    """GHCN-hourly airport observations, already in SI units."""

    source_name = "noaa_ghcnh"
    station_namespace = "ghcnh"

    variables: ClassVar[VariableMap] = {
        "temperature": (schema.TEMPERATURE_C, None),
        "dew_point_temperature": (schema.DEWPOINT_C, None),
        "relative_humidity": (schema.RELATIVE_HUMIDITY_PCT, None),
        "wind_speed": (schema.WIND_SPEED_MS, None),
        "wind_direction": (schema.WIND_DIRECTION_DEG, None),
        "wind_gust": (schema.WIND_GUST_MS, None),
        "station_level_pressure": (schema.PRESSURE_HPA, None),
        "sea_level_pressure": (schema.SEA_LEVEL_PRESSURE_HPA, None),
        "precipitation": (schema.PRECIPITATION_MM, None),
        "visibility": (schema.VISIBILITY_M, schema.kilometers_to_meters),
    }

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self._apply_variables(frame, self.variables)
        for variable, (canonical, _) in self.variables.items():
            code_column = f"{variable}{QUALITY_CODE_SUFFIX}"
            if canonical not in columns or code_column not in frame.columns:
                continue
            codes = frame[code_column].astype("string").str.strip()
            suspect = codes.notna() & ~codes.isin(ACCEPTED_QUALITY_CODES)
            columns[canonical] = columns[canonical].mask(suspect.to_numpy())
        return self._finalize(frame, columns, frame[schema.STATION_ID])


class IemAsosNormalizer(SourceNormalizer):
    """METAR observations reported in imperial units."""

    source_name = "iem_asos"
    station_namespace = "iem"

    sky_columns: ClassVar[tuple[str, ...]] = ("skyc1", "skyc2", "skyc3", "skyc4")
    variables: ClassVar[VariableMap] = {
        "tmpf": (schema.TEMPERATURE_C, schema.fahrenheit_to_celsius),
        "dwpf": (schema.DEWPOINT_C, schema.fahrenheit_to_celsius),
        "relh": (schema.RELATIVE_HUMIDITY_PCT, None),
        "sknt": (schema.WIND_SPEED_MS, schema.knots_to_meters_per_second),
        "drct": (schema.WIND_DIRECTION_DEG, None),
        "gust": (schema.WIND_GUST_MS, schema.knots_to_meters_per_second),
        "mslp": (schema.SEA_LEVEL_PRESSURE_HPA, None),
        "p01i": (schema.PRECIPITATION_MM, schema.inches_to_millimeters),
        "vsby": (schema.VISIBILITY_M, schema.miles_to_meters),
    }

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self._apply_variables(frame, self.variables)
        present = [column for column in self.sky_columns if column in frame.columns]
        if present:
            # METAR sky layers are cumulative, so the greatest reported cover is
            # the total cover. A row with no readable layer stays missing.
            layers = pd.DataFrame(
                {
                    column: schema.sky_cover_to_percent(frame[column])
                    for column in present
                }
            )
            columns[schema.CLOUD_COVER_PCT] = layers.max(axis=1, skipna=True)
        return self._finalize(frame, columns, frame[schema.STATION_ID])


class OpenMeteoNormalizer(SourceNormalizer):
    """ERA5-Seamless reanalysis already requested in the project's units."""

    source_name = "open_meteo"
    station_namespace = "ghcnh"

    variables: ClassVar[VariableMap] = {
        "temperature_2m": (schema.TEMPERATURE_C, None),
        "dew_point_2m": (schema.DEWPOINT_C, None),
        "relative_humidity_2m": (schema.RELATIVE_HUMIDITY_PCT, None),
        "wind_speed_10m": (schema.WIND_SPEED_MS, None),
        "wind_direction_10m": (schema.WIND_DIRECTION_DEG, None),
        "wind_gusts_10m": (schema.WIND_GUST_MS, None),
        "pressure_msl": (schema.SEA_LEVEL_PRESSURE_HPA, None),
        "precipitation": (schema.PRECIPITATION_MM, None),
        "cloud_cover": (schema.CLOUD_COVER_PCT, None),
        "shortwave_radiation": (schema.SHORTWAVE_RADIATION_WM2, None),
    }

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self._apply_variables(frame, self.variables)
        return self._finalize(frame, columns, frame[schema.STATION_ID])


class EconetNormalizer(SourceNormalizer):
    """ECONet observations, published one variable per row."""

    source_name = "ncsco_econet"
    station_namespace = "econet"
    read_options: ClassVar[Mapping[str, Any]] = {"dtype": {"value": str}}

    variables: ClassVar[VariableMap] = {
        "airtemp2m|C": (schema.TEMPERATURE_C, None),
        "dewtemp2m|C": (schema.DEWPOINT_C, None),
        "windspeed10m|ms": (schema.WIND_SPEED_MS, None),
    }

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        graded = frame.loc[
            pd.to_numeric(frame["score"], errors="coerce") <= MAX_ECONET_SCORE
        ].copy()
        graded["value"] = pd.to_numeric(graded["value"], errors="coerce")
        wide = graded.pivot_table(
            index=[schema.STATION_ID, schema.TIMESTAMP_UTC],
            columns="var",
            values="value",
            aggfunc="mean",
        ).reset_index()
        wide.columns.name = None
        columns = self._apply_variables(wide, self.variables)
        return self._finalize(wide, columns, wide[schema.STATION_ID])


NORMALIZERS: tuple[type[SourceNormalizer], ...] = (
    NoaaGhcnhNormalizer,
    IemAsosNormalizer,
    OpenMeteoNormalizer,
    EconetNormalizer,
)
