"""Normalize archived GFS runs into one hourly forecast panel.

Run from the repository root with:

    python -m rdu_temperature.pipeline.clean_gfs

This is the forecast counterpart to ``clean_weather``, kept separate because
the shape genuinely differs. An observation is a station and an hour; a
forecast is a run and an hour, and the run's initialisation time is what makes
a prediction for the forecast period legitimate to use. ``HourlyGrid`` aligns
stations onto one timestamp axis and has nowhere to put a second clock, so
forcing GFS through it would mean discarding the guarantee the covariate rests
on.

The output is deliberately model-agnostic. Prophet wants named regressor
columns beside its ``ds``; XGBoost wants a wide numeric matrix. Both are a join
away from one panel keyed by initialisation and valid time, so the panel is
what this writes and neither model's preferences leak into it.

Two decisions carry the module.

**Interpolation order.** The 0.25-degree product is hourly to 120 hours and
three-hourly beyond, so two thirds of a fourteen-day horizon arrives coarser
than the grid the project is scored on. Those hours are filled by linear
interpolation between published leads -- and the wind is interpolated as
eastward and northward components, *before* being turned into a speed and a
bearing. Degrees do not interpolate: halfway between 350 and 10 is 180 by
arithmetic and 0 in reality. Components have no such discontinuity.

**Interpolation is recorded.** Every row carries whether its values came from a
published lead or were filled between two, so a model can weight them, and so
a result that depends on the interpolated two thirds can be told apart from one
that does not.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.quality import PLAUSIBLE_RANGES, Screener

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "noaa_gfs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Raw GRIB names, and the canonical column each becomes once converted.
EASTWARD_MS = "wind_eastward_ms"
NORTHWARD_MS = "wind_northward_ms"

DIRECT_CONVERSIONS: Mapping[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "TMP": (schema.TEMPERATURE_C, schema.kelvin_to_celsius),
    "DPT": (schema.DEWPOINT_C, schema.kelvin_to_celsius),
    "PRATE": (
        schema.PRECIPITATION_MM,
        schema.precipitation_rate_to_millimetres_per_hour,
    ),
    "TCDC": (schema.CLOUD_COVER_PCT, None),
    "UGRD": (EASTWARD_MS, None),
    "VGRD": (NORTHWARD_MS, None),
}

# Columns that interpolate linearly and so may be filled between leads. Wind
# is here as components; speed and bearing are derived after the fill.
INTERPOLATED_COLUMNS: tuple[str, ...] = (
    schema.TEMPERATURE_C,
    schema.DEWPOINT_C,
    schema.CLOUD_COVER_PCT,
    schema.PRECIPITATION_MM,
    EASTWARD_MS,
    NORTHWARD_MS,
)

INTERPOLATED = "interpolated"

PANEL_COLUMNS: tuple[str, ...] = (
    schema.SOURCE,
    schema.STATION_ID,
    schema.INIT_TIME_UTC,
    schema.VALID_TIME_UTC,
    schema.LEAD_HOURS,
    schema.TEMPERATURE_C,
    schema.DEWPOINT_C,
    schema.WIND_SPEED_MS,
    schema.WIND_DIRECTION_DEG,
    schema.CLOUD_COVER_PCT,
    schema.PRECIPITATION_MM,
    INTERPOLATED,
)


def load_runs(input_dir: Path) -> pd.DataFrame:
    """Read every ingested run file into one frame."""
    paths = sorted(input_dir.glob("gfs_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No GFS runs beneath {input_dir}; run rdu_temperature.pipeline.ingest_gfs."
        )
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    for column in (schema.INIT_TIME_UTC, schema.VALID_TIME_UTC):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
    return frame


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert published names and units into the project's vocabulary."""
    columns: dict[str, pd.Series] = {}
    for raw, (canonical, convert) in DIRECT_CONVERSIONS.items():
        if raw not in frame.columns:
            continue
        values = pd.to_numeric(frame[raw], errors="coerce")
        columns[canonical] = convert(values) if convert else values

    normalized = pd.DataFrame(
        {
            schema.SOURCE: frame[schema.SOURCE],
            schema.STATION_ID: frame[schema.STATION_ID],
            schema.INIT_TIME_UTC: frame[schema.INIT_TIME_UTC],
            schema.VALID_TIME_UTC: frame[schema.VALID_TIME_UTC],
            schema.LEAD_HOURS: pd.to_numeric(frame[schema.LEAD_HOURS]),
            **columns,
        }
    )
    return normalized.sort_values(
        [schema.INIT_TIME_UTC, schema.VALID_TIME_UTC], kind="stable"
    ).reset_index(drop=True)


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Place each run on an hourly grid, filling the three-hourly tail.

    Every run is reindexed across its own span only. Extrapolating past the
    last published lead would invent a forecast the model never made, so the
    grid stops where the run does.
    """
    filled: list[pd.DataFrame] = []
    for init_time, run in frame.groupby(schema.INIT_TIME_UTC, sort=True):
        run = run.sort_values(schema.VALID_TIME_UTC, kind="stable")
        hourly = pd.date_range(
            run[schema.VALID_TIME_UTC].min(),
            run[schema.VALID_TIME_UTC].max(),
            freq="h",
            name=schema.VALID_TIME_UTC,
        )
        indexed = run.set_index(schema.VALID_TIME_UTC).reindex(hourly)
        published = indexed[schema.LEAD_HOURS].notna()

        present = [c for c in INTERPOLATED_COLUMNS if c in indexed.columns]
        indexed[present] = indexed[present].interpolate(
            method="time", limit_area="inside"
        )

        indexed[schema.SOURCE] = run[schema.SOURCE].iloc[0]
        indexed[schema.STATION_ID] = run[schema.STATION_ID].iloc[0]
        indexed[schema.INIT_TIME_UTC] = init_time
        indexed[schema.LEAD_HOURS] = (
            (hourly - init_time).total_seconds() // 3600
        ).astype(int)
        indexed[INTERPOLATED] = ~published.to_numpy()
        filled.append(indexed.reset_index())
    return pd.concat(filled, ignore_index=True)


def derive_wind(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn interpolated wind components into a speed and a bearing.

    Done after the fill, never before: a bearing cannot be interpolated across
    north, where the arithmetic midpoint of 350 and 10 degrees is due south.
    """
    if not {EASTWARD_MS, NORTHWARD_MS} <= set(frame.columns):
        return frame
    eastward, northward = frame[EASTWARD_MS], frame[NORTHWARD_MS]
    return frame.assign(
        **{
            schema.WIND_SPEED_MS: schema.wind_components_to_speed(eastward, northward),
            schema.WIND_DIRECTION_DEG: schema.wind_components_to_direction(
                eastward, northward
            ),
        }
    ).drop(columns=[EASTWARD_MS, NORTHWARD_MS])


@dataclass(frozen=True)
class GfsCleaner:
    """Normalize, grid, derive, and screen the ingested runs.

    Screening keeps the plausible-range and dew-point rules and drops the
    flatline rule. A forecast holding one value for a day is a model
    predicting settled weather, not an instrument that has stopped reporting.
    """

    screener: Screener = field(
        default_factory=lambda: Screener(ranges=PLAUSIBLE_RANGES, flatline_variables=())
    )

    def clean(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        panel = derive_wind(to_hourly(normalize(raw)))
        screened = self.screener.screen(panel)
        ordered = [c for c in PANEL_COLUMNS if c in screened.frame.columns]
        return screened.frame.loc[:, ordered], screened.report


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


class CleanGfsApp:
    """Read the ingested runs and write the hourly forecast panel."""

    panel_filename = "gfs_forecast_panel.parquet"
    report_filename = "gfs_screening_report.csv"

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self, *, overwrite: bool = False) -> dict[str, Path]:
        outputs = {
            "panel": self.output_dir / self.panel_filename,
            "report": self.output_dir / self.report_filename,
        }
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{existing[0]} already exists; pass --overwrite to replace it."
            )

        raw = load_runs(self.input_dir)
        runs = raw[schema.INIT_TIME_UTC].nunique()
        print(f"Read {runs} run(s), {len(raw):,} published lead(s).", flush=True)

        panel, report = GfsCleaner().clean(raw)
        interpolated = int(panel[INTERPOLATED].sum())
        print(
            f"Panel holds {len(panel):,} hour(s); {interpolated:,} "
            f"({interpolated / len(panel):.1%}) interpolated between leads.",
            flush=True,
        )
        if len(report):
            print(f"Screening masked {int(report['masked'].sum())} reading(s).")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write(panel.to_parquet, outputs["panel"])
        _write(lambda path: report.to_csv(path, index=False), outputs["report"])
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}", flush=True)
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    CleanGfsApp(args.input_dir, args.output_dir).run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
