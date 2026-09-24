"""Download raw historical weather data from the project's selected sources.

Run from the repository root with:

    python -m rdu_temperature.pipeline.ingest_weather

The module intentionally keeps source-specific behavior in small subclasses. HTTP
retries, cutoff enforcement, provenance fields, and CSV output are implemented once
in :class:`WeatherSource`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "weather_sources.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DateWindow:
    """A half-open UTC interval used consistently by every source."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_strings(cls, start: str, end: str) -> DateWindow:
        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)
        if start_timestamp.tzinfo is None or end_timestamp.tzinfo is None:
            raise ValueError("The ingestion window must include UTC offsets.")
        return cls(start_timestamp.tz_convert("UTC"), end_timestamp.tz_convert("UTC"))

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("The ingestion start must be earlier than the end.")

    def calendar_years(self) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        cursor = self.start
        while cursor < self.end:
            boundary = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz="UTC")
            chunk_end = min(boundary, self.end)
            yield cursor, chunk_end
            cursor = chunk_end

    def fixed_days(self, days: int) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        if days < 1:
            raise ValueError("Chunk size must be at least one day.")
        cursor = self.start
        while cursor < self.end:
            chunk_end = min(cursor + pd.Timedelta(days=days), self.end)
            yield cursor, chunk_end
            cursor = chunk_end


class WeatherSource(ABC):
    """Shared ingestion lifecycle for an HTTP weather source."""

    source_name: ClassVar[str]

    def __init__(
        self,
        window: DateWindow,
        output_dir: Path,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.window = window
        self.output_dir = output_dir
        self.session = session or self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "rdu-temperature-forecasting/1.0 (academic project)"}
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> requests.Response:
        try:
            response = self.session.get(url, params=params, timeout=(15, 180))
            response.raise_for_status()
        except requests.RequestException:
            raise RuntimeError(
                f"{self.source_name} request failed after retries."
            ) from None
        return response

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return source records with station_id and timestamp_utc columns."""

    def ingest(self, *, overwrite: bool = False) -> Path:
        source_dir = self.output_dir / self.source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        start = self.window.start.strftime("%Y%m%dT%H%M%SZ")
        end = self.window.end.strftime("%Y%m%dT%H%M%SZ")
        output_path = source_dir / f"weather_{start}_{end}.csv"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it."
            )

        frame = self.fetch()
        required = {"station_id", "timestamp_utc"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{self.source_name} omitted required columns: {missing}")

        frame = frame.copy()
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame.loc[
            (frame["timestamp_utc"] >= self.window.start)
            & (frame["timestamp_utc"] < self.window.end)
        ]
        frame = frame.drop_duplicates()
        frame.insert(0, "source", self.source_name)
        frame = frame.sort_values(["station_id", "timestamp_utc"], kind="stable")
        temporary_path = output_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        temporary_path.replace(output_path)
        return output_path


class NoaaGhcnhSource(WeatherSource):
    """NOAA GHCN-hourly annual Parquet files for airport stations."""

    source_name = "noaa_ghcnh"
    url_template = (
        "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
        "hourly/access/by-year/{year}/parquet/GHCNh_{station_id}_{year}.parquet"
    )
    attribute_suffixes = (
        "_Measurement_Code",
        "_Quality_Code",
        "_Report_Type",
        "_Source_Code",
        "_Source_Station_ID",
    )

    def __init__(
        self,
        *args: Any,
        stations: Sequence[Mapping[str, Any]],
        variables: Sequence[str],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.stations = stations
        self.variables = variables

    def fetch(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        years = {chunk_start.year for chunk_start, _ in self.window.calendar_years()}
        for station in self.stations:
            for year in sorted(years):
                url = self.url_template.format(
                    year=year, station_id=station["ghcnh_id"]
                )
                response = self._get(url)
                frame = pd.read_parquet(io.BytesIO(response.content))
                frame["timestamp_utc"] = pd.to_datetime(frame["DATE"], utc=True)
                frame = frame.loc[
                    (frame["timestamp_utc"] >= self.window.start)
                    & (frame["timestamp_utc"] < self.window.end)
                    & frame["temperature_Report_Type"].eq("FM15")
                ]
                identity_columns = [
                    "DATE",
                    "STATION",
                    "NAME",
                    "LATITUDE",
                    "LONGITUDE",
                    "ELEVATION",
                    "timestamp_utc",
                ]
                configured_columns = {
                    column
                    for variable in self.variables
                    for column in (
                        variable,
                        *(f"{variable}{suffix}" for suffix in self.attribute_suffixes),
                    )
                }
                variable_columns = [
                    column for column in frame.columns if column in configured_columns
                ]
                selected_columns = [
                    column
                    for column in (*identity_columns, *variable_columns)
                    if column in frame.columns
                ]
                frame = frame.loc[:, list(dict.fromkeys(selected_columns))].copy()
                frame["station_id"] = station["ghcnh_id"]
                frame["station_name_config"] = station["name"]
                frame["station_icao"] = station["icao"]
                frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class IemAsosSource(WeatherSource):
    """Iowa Environmental Mesonet routine METAR archive."""

    source_name = "iem_asos"
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    def __init__(
        self, *args: Any, stations: Sequence[Mapping[str, Any]], **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.stations = stations

    def fetch(self) -> pd.DataFrame:
        params: list[tuple[str, Any]] = [
            ("station", station["iem_id"]) for station in self.stations
        ]
        params.extend(
            [
                ("data", "all"),
                ("sts", self.window.start.isoformat().replace("+00:00", "Z")),
                ("ets", self.window.end.isoformat().replace("+00:00", "Z")),
                ("tz", "Etc/UTC"),
                ("format", "onlycomma"),
                ("latlon", "yes"),
                ("elev", "yes"),
                ("missing", "empty"),
                ("trace", "0.0001"),
                ("report_type", "3"),
            ]
        )
        response = self._get(self.url, params=params)
        frame = pd.read_csv(io.StringIO(response.text), low_memory=False)
        frame["timestamp_utc"] = pd.to_datetime(frame["valid"], utc=True)
        frame["station_id"] = frame["station"]
        names = {station["iem_id"]: station["name"] for station in self.stations}
        frame["station_name_config"] = frame["station_id"].map(names)
        return frame


class OpenMeteoSource(WeatherSource):
    """Open-Meteo historical API pinned to a consistent reanalysis model."""

    source_name = "open_meteo"
    url = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        *args: Any,
        stations: Sequence[Mapping[str, Any]],
        variables: Sequence[str],
        model: str,
        request_interval_seconds: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stations = stations
        self.variables = variables
        self.model = model
        self.request_interval_seconds = request_interval_seconds

    def fetch(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        latitudes = ",".join(str(station["latitude"]) for station in self.stations)
        longitudes = ",".join(str(station["longitude"]) for station in self.stations)

        for request_number, (chunk_start, chunk_end) in enumerate(
            self.window.calendar_years()
        ):
            if request_number:
                time.sleep(self.request_interval_seconds)
            params = {
                "latitude": latitudes,
                "longitude": longitudes,
                "start_date": chunk_start.date().isoformat(),
                "end_date": (chunk_end - pd.Timedelta(nanoseconds=1))
                .date()
                .isoformat(),
                "hourly": ",".join(self.variables),
                "models": self.model,
                "timezone": "UTC",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            }
            payload = self._get(self.url, params=params).json()
            responses = payload if isinstance(payload, list) else [payload]
            if len(responses) != len(self.stations):
                raise ValueError("Open-Meteo returned an unexpected location count.")

            for station, location in zip(self.stations, responses, strict=True):
                frame = pd.DataFrame(location["hourly"])
                frame["timestamp_utc"] = pd.to_datetime(frame["time"], utc=True)
                frame["station_id"] = station["ghcnh_id"]
                frame["station_name_config"] = station["name"]
                frame["requested_latitude"] = station["latitude"]
                frame["requested_longitude"] = station["longitude"]
                frame["grid_latitude"] = location["latitude"]
                frame["grid_longitude"] = location["longitude"]
                frame["grid_elevation_m"] = location.get("elevation")
                frame["model"] = self.model
                frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class EconetSource(WeatherSource):
    """North Carolina State Climate Office CLOUDS API for ECONet stations."""

    source_name = "ncsco_econet"
    url = "https://api.climate.ncsu.edu/data.php"

    def __init__(
        self,
        *args: Any,
        stations: Sequence[Mapping[str, Any]],
        variables: Sequence[str],
        api_hash: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stations = stations
        self.variables = variables
        self.api_hash = api_hash

    def fetch(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        station_ids = ",".join(station["id"] for station in self.stations)
        station_names = {station["id"]: station["name"] for station in self.stations}

        for chunk_start, chunk_end in self.window.fixed_days(90):
            params = {
                "hash": self.api_hash,
                "loc": f"location={station_ids}",
                "var": ",".join(self.variables),
                "start": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                "int": "1 hour",
                "obtype": "H",
                "output": "csv_long",
                "timezone": "UTC",
                "metadata": "no",
                "qclimit": "1",
                "date_partial": "no",
                "attr": "location,datetime,var,value,unit,score",
            }
            response = self._get(self.url, params=params)
            data_lines = [
                line for line in response.text.splitlines() if not line.startswith("##")
            ]
            frame = pd.read_csv(io.StringIO("\n".join(data_lines)))
            frame["timestamp_utc"] = pd.to_datetime(frame["datetime"], utc=True)
            frame["station_id"] = frame["location"]
            frame["station_name_config"] = frame["station_id"].map(station_names)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class WeatherIngestionApp:
    """Load project configuration and coordinate selected source ingestors."""

    source_names = ("noaa-ghcnh", "ncsco-econet", "open-meteo", "iem-asos")

    def __init__(self, config_path: Path, output_dir: Path) -> None:
        with config_path.open(encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        history = self.config["history"]
        self.window = DateWindow.from_strings(history["start_utc"], history["end_utc"])
        self.output_dir = output_dir

    def run(self, selected: Sequence[str], *, overwrite: bool = False) -> list[Path]:
        requested = list(self.source_names) if "all" in selected else list(selected)
        api_hash = os.getenv("NCSCO_API_HASH")
        if "ncsco-econet" in requested and not api_hash:
            raise ValueError(
                "NCSCO_API_HASH is required for ECONet. Add it to .env or select "
                "only public sources with --source."
            )

        common = {"window": self.window, "output_dir": self.output_dir}
        airport_stations = self.config["airport_stations"]
        source_map: dict[str, WeatherSource] = {
            "noaa-ghcnh": NoaaGhcnhSource(
                **common,
                stations=airport_stations,
                variables=self.config["noaa_ghcnh"]["variables"],
            ),
            "ncsco-econet": EconetSource(
                **common,
                stations=self.config["econet_stations"],
                variables=self.config["econet"]["variables"],
                api_hash=api_hash or "",
            ),
            "open-meteo": OpenMeteoSource(
                **common,
                stations=airport_stations,
                variables=self.config["open_meteo"]["variables"],
                model=self.config["open_meteo"]["model"],
                request_interval_seconds=self.config["open_meteo"][
                    "request_interval_seconds"
                ],
            ),
            "iem-asos": IemAsosSource(**common, stations=airport_stations),
        }

        outputs: list[Path] = []
        for source_name in requested:
            print(f"Ingesting {source_name}...", flush=True)
            output_path = source_map[source_name].ingest(overwrite=overwrite)
            print(f"Wrote {output_path}", flush=True)
            outputs.append(output_path)
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=("all", *WeatherIngestionApp.source_names),
        default=None,
        help="Source to ingest; repeat for multiple sources. Defaults to all.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    app = WeatherIngestionApp(args.config, args.output_dir)
    app.run(args.source or ["all"], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
