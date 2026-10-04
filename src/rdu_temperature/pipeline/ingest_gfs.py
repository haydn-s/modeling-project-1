"""Ingest archived GFS forecasts as covariates the forecast period can use.

Run from the repository root with:

    python -m rdu_temperature.pipeline.ingest_gfs --runs forecast

Every other source in this project is an observation: one timestamp, one
reading, and nothing legitimate to say about a period that has not happened.
GFS is different, and the difference is the entire point. A model run records
the time it was initialised, so a run that started before the data cutoff knew
nothing after it, and its predictions for the forecast period are fair to use.
That is what makes covariates possible at all here.

Two consequences shape this module.

It does not reuse ``WeatherSource``. That base class filters rows to a single
half-open window on one timestamp, which would discard exactly the records this
source exists to fetch, and it has no place to keep an initialisation time. A
forecast needs both clocks: drop the initialisation time and the leakage
guarantee goes with it.

It fetches by byte range. A GFS 0.25-degree file is about 520 MB and this
project wants six fields at one grid point. Each file publishes a ``.idx``
sidecar giving the byte offset of every message, so a ranged request pulls a
single field in about half a megabyte -- a thousandfold saving that is the
difference between a feasible download and an infeasible one.

Runs are written one file per initialisation. A full training fetch is tens of
thousands of requests, so it has to survive being interrupted; an existing run
file is skipped rather than refetched, which also lets a teammate build the
archive up in stages instead of in one sitting.

What GFS cannot do is worth stating plainly. The 0.25-degree product is hourly
only to 120 hours and three-hourly after that, so the back nine days of a
fourteen-day horizon arrive at a third of the resolution the project is scored
on. Those hours are interpolated when the panel is built, and the interpolation
is recorded rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rdu_temperature.pipeline import schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "weather_sources.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

SOURCE_NAME = "noaa_gfs"

# The 0.25-degree product publishes hourly out to this lead and three-hourly
# beyond it, which is a property of the product rather than a choice here.
HOURLY_LEAD_LIMIT = 120
COARSE_LEAD_STEP = 3


@dataclass(frozen=True)
class GribField:
    """One GRIB message to pull, named as the product names it.

    Raw files keep the publisher's names and units, as every other source in
    this project does: kelvin stays kelvin and wind stays a pair of components
    until the cleaning stage converts them. Writing a column called
    ``temperature_c`` holding kelvin would be worse than writing ``TMP``.
    """

    name: str
    level: str

    @property
    def key(self) -> str:
        return f"{self.name}:{self.level}"


# Instantaneous fields only. APCP is published over accumulation windows that
# change with lead -- six-hourly in places, daily in others -- and cannot be
# placed on an hourly grid without inventing a disaggregation. PRATE is the
# rate at the valid hour, which converts to millimetres in that hour directly.
#
# Wind arrives as eastward and northward components because the product
# publishes no single speed field at ten metres; cleaning combines them.
GRIB_FIELDS: tuple[GribField, ...] = (
    GribField("TMP", "2 m above ground"),
    GribField("DPT", "2 m above ground"),
    GribField("UGRD", "10 m above ground"),
    GribField("VGRD", "10 m above ground"),
    GribField("TCDC", "entire atmosphere"),
    GribField("PRATE", "surface"),
)


@dataclass(frozen=True)
class GfsRun:
    """One model initialisation and the leads wanted from it."""

    init_time: pd.Timestamp
    leads: tuple[int, ...]

    @property
    def slug(self) -> str:
        return self.init_time.strftime("%Y%m%dT%H%MZ")

    def url(self, lead: int) -> str:
        day = self.init_time.strftime("%Y%m%d")
        cycle = self.init_time.strftime("%H")
        return (
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{day}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}"
        )


def available_leads(first: int, last: int) -> tuple[int, ...]:
    """Return the leads the product actually publishes between two bounds.

    Hourly to the hourly limit, three-hourly after it. Asking for a lead the
    archive does not hold returns a 404 rather than an error worth retrying,
    so the leads are worked out here instead of discovered.
    """
    hourly = range(first, min(last, HOURLY_LEAD_LIMIT) + 1)
    coarse_start = max(first, HOURLY_LEAD_LIMIT + COARSE_LEAD_STEP)
    coarse_start += -coarse_start % COARSE_LEAD_STEP
    coarse = range(coarse_start, last + 1, COARSE_LEAD_STEP)
    return tuple(sorted(set(hourly) | set(coarse)))


def leads_covering(
    init_time: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[int, ...]:
    """Leads from ``init_time`` that cover the half-open window ``[start, end)``.

    The first lead is rounded down and the last up, so the window is spanned
    rather than merely touched; interpolation onto the hourly grid then has a
    value on both sides of every gap.
    """
    first = math.floor((start - init_time) / pd.Timedelta(hours=1))
    last = math.ceil((end - init_time) / pd.Timedelta(hours=1))
    return available_leads(max(first, 0), last)


@dataclass(frozen=True)
class IndexEntry:
    """Where one GRIB message sits inside its file."""

    key: str
    start: int
    end: int | None

    @property
    def byte_range(self) -> str:
        return f"{self.start}-" if self.end is None else f"{self.start}-{self.end}"


def parse_index(text: str) -> dict[str, IndexEntry]:
    """Turn a ``.idx`` sidecar into a lookup from field key to byte range.

    Each line is ``record:offset:date:name:level:forecast:``. A message runs
    from its own offset to the byte before the next one, and the final message
    runs to the end of the file. Only the first match for a key is kept: the
    averaged and accumulated variants that share a name sit later in the file
    and are not what this module asks for.
    """
    offsets: list[tuple[int, str]] = []
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 6:
            continue
        offsets.append((int(parts[1]), f"{parts[3]}:{parts[4]}"))
    offsets.sort()

    entries: dict[str, IndexEntry] = {}
    for position, (start, key) in enumerate(offsets):
        following = (
            offsets[position + 1][0] - 1 if position + 1 < len(offsets) else None
        )
        entries.setdefault(key, IndexEntry(key=key, start=start, end=following))
    return entries


class PointExtractor:
    """Pull the grid point nearest a station out of a single GRIB message."""

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude

    def value(self, message: bytes) -> float:
        # Imported here so that the module can be imported, and its pure
        # functions tested, on a machine without the GRIB libraries.
        import eccodes

        handle = eccodes.codes_new_from_message(message)
        try:
            nearest = eccodes.codes_grib_find_nearest(
                handle, self.latitude, self.longitude
            )[0]
            return float(nearest.value)
        finally:
            eccodes.codes_release(handle)


class GfsIngestion:
    """Fetch the configured fields for a set of runs, one file per run."""

    source_name = SOURCE_NAME

    def __init__(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        output_dir: Path,
        *,
        fields: Sequence[GribField] = GRIB_FIELDS,
        session: requests.Session | None = None,
    ) -> None:
        self.station_id = station_id
        self.extractor = PointExtractor(latitude, longitude)
        self.output_dir = output_dir / self.source_name
        self.fields = tuple(fields)
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

    def _get(self, url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
        try:
            response = self.session.get(
                url, headers=dict(headers or {}), timeout=(15, 180)
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"GFS request failed: {url}") from error
        return response.content

    def fetch_lead(self, run: GfsRun, lead: int) -> dict[str, float]:
        """Return one row of field values for a single initialisation and lead."""
        url = run.url(lead)
        index = parse_index(self._get(f"{url}.idx").decode("utf-8"))
        values: dict[str, float] = {}
        for field in self.fields:
            entry = index.get(field.key)
            if entry is None:
                continue
            message = self._get(url, headers={"Range": f"bytes={entry.byte_range}"})
            values[field.name] = self.extractor.value(message)
        return values

    def fetch_run(self, run: GfsRun) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for lead in run.leads:
            rows.append(self.fetch_row(run, lead))
        return pd.DataFrame(rows)

    def fetch_row(self, run: GfsRun, lead: int) -> dict[str, Any]:
        """Fetch one lead and attach the two timestamps that prevent leakage."""
        values = self.fetch_lead(run, lead)
        return {
            schema.SOURCE: self.source_name,
            schema.STATION_ID: self.station_id,
            schema.INIT_TIME_UTC: run.init_time,
            schema.VALID_TIME_UTC: run.init_time + pd.Timedelta(hours=lead),
            schema.LEAD_HOURS: lead,
            **values,
        }

    def run_path(self, run: GfsRun) -> Path:
        return self.output_dir / f"gfs_{run.slug}.csv"

    def checkpoint_path(self, run: GfsRun) -> Path:
        """Path holding completed leads until the run is fully downloaded."""
        return self.output_dir / f"gfs_{run.slug}.partial.csv"

    @staticmethod
    def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        frame.to_csv(temporary_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        temporary_path.replace(path)

    def ingest_run(self, run: GfsRun, *, overwrite: bool = False) -> Path:
        """Fetch one run, resuming after the last successfully saved lead.

        NOAA serves one index plus one byte-range request per field for every
        lead. A fourteen-day run therefore makes more than a thousand HTTP
        requests. Saving only after the final request meant one late timeout
        discarded the entire run. The partial CSV is updated atomically after
        each lead so a retry loses at most the request currently in flight.
        """
        path = self.run_path(run)
        if path.exists() and not overwrite:
            return path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_path(run)
        if overwrite:
            checkpoint.unlink(missing_ok=True)

        if checkpoint.exists():
            frame = pd.read_csv(
                checkpoint,
                parse_dates=[schema.INIT_TIME_UTC, schema.VALID_TIME_UTC],
            )
            completed = set(frame[schema.LEAD_HOURS].astype(int))
        else:
            frame = pd.DataFrame()
            completed = set()

        total = len(run.leads)
        for position, lead in enumerate(run.leads, start=1):
            if lead in completed:
                continue
            row = pd.DataFrame([self.fetch_row(run, lead)])
            frame = pd.concat([frame, row], ignore_index=True)
            frame = frame.sort_values(schema.LEAD_HOURS).reset_index(drop=True)
            self._write_csv_atomic(frame, checkpoint)
            print(
                f"    {run.slug}: saved lead {lead:03d} "
                f"({position}/{total})",
                flush=True,
            )

        # Renaming a complete checkpoint is atomic. A failed or interrupted
        # run deliberately leaves the partial file in place for the next call.
        checkpoint.replace(path)
        return path

    def ingest(
        self, runs: Sequence[GfsRun], *, overwrite: bool = False
    ) -> Iterator[tuple[GfsRun, Path, bool]]:
        for run in runs:
            path = self.run_path(run)
            already = path.exists() and not overwrite
            yield run, self.ingest_run(run, overwrite=overwrite), already


@dataclass(frozen=True)
class RunPlan:
    """Which initialisations to fetch, and the leads each should supply.

    The cutoff is the contract. Every run here starts strictly before it, so
    nothing fetched can carry information from the forecast period, and the
    plan refuses to build a run that would.
    """

    cutoff: pd.Timestamp
    horizon_hours: int
    cycle_hour: int = 12
    # How long before a cutoff the run used for it was initialised. The
    # operational run closest to a cutoff is the most skilful one legitimately
    # available, and training folds have to be given the same advantage or the
    # covariate they learn from is better than the one they will be used with.
    lead_in_hours: int = 16

    def run_for(self, cutoff: pd.Timestamp) -> GfsRun:
        """The latest run that starts before ``cutoff`` and covers its horizon."""
        init_time = (cutoff - pd.Timedelta(hours=self.lead_in_hours)).normalize()
        init_time += pd.Timedelta(hours=self.cycle_hour)
        while init_time >= cutoff:
            init_time -= pd.Timedelta(days=1)
        if init_time >= self.cutoff:
            raise ValueError(
                f"A run initialised {init_time} is not before the data cutoff "
                f"{self.cutoff}; it could carry forecast-period information."
            )
        end = cutoff + pd.Timedelta(hours=self.horizon_hours)
        return GfsRun(init_time, leads_covering(init_time, cutoff, end))

    def for_cutoffs(self, cutoffs: Sequence[pd.Timestamp]) -> list[GfsRun]:
        runs = {run.init_time: run for run in (self.run_for(c) for c in cutoffs)}
        return [runs[key] for key in sorted(runs)]


def load_station(config_path: Path, station_id: str) -> tuple[float, float]:
    """Return a configured station's coordinates."""
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    for station in config["airport_stations"]:
        if station["icao"] == station_id:
            return float(station["latitude"]), float(station["longitude"])
    raise ValueError(f"No airport station {station_id!r} in {config_path}.")


def load_cutoff(config_path: Path) -> pd.Timestamp:
    with config_path.open(encoding="utf-8") as config_file:
        history = json.load(config_file)["history"]
    return pd.Timestamp(history["end_utc"]).tz_convert("UTC").tz_localize(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station", default="KRDU")
    parser.add_argument(
        "--runs",
        choices=("forecast", "seasonal", "rolling"),
        default="forecast",
        help=(
            "forecast: the single run covering the forecast period. "
            "seasonal: that run plus one per September backtest fold. "
            "rolling: that run plus one per sampled rolling fold."
        ),
    )
    parser.add_argument(
        "--fold-step",
        type=int,
        default=4,
        help="With --runs rolling, take every Nth fold.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    latitude, longitude = load_station(args.config, args.station)
    cutoff = load_cutoff(args.config)
    plan = RunPlan(cutoff=cutoff, horizon_hours=336)

    cutoffs = [cutoff]
    if args.runs != "forecast":
        # Imported lazily: the fold definitions live with the evaluation code,
        # and ingestion should not need it to fetch the forecast run alone.
        from rdu_temperature.evaluation.backtest import (
            rolling_cutoffs,
            seasonal_cutoffs,
        )
        from rdu_temperature.features import prophet_frame

        frame = prophet_frame.load()
        if args.runs == "seasonal":
            cutoffs += seasonal_cutoffs(frame, cutoff)
        else:
            cutoffs += rolling_cutoffs(frame)[:: args.fold_step]

    runs = plan.for_cutoffs(cutoffs)
    fetches = sum(len(run.leads) for run in runs) * (len(GRIB_FIELDS) + 1)
    print(
        f"{len(runs)} run(s), {sum(len(r.leads) for r in runs)} lead(s), "
        f"about {fetches:,} request(s).",
        flush=True,
    )

    ingestion = GfsIngestion(args.station, latitude, longitude, args.output_dir)
    for run, path, already in ingestion.ingest(runs, overwrite=args.overwrite):
        state = "skipped (already present)" if already else f"{len(run.leads)} leads"
        print(f"  {run.slug}  {state}  -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
