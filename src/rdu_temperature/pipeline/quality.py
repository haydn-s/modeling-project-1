"""Screen implausible readings out of the aligned weather panel.

Screening masks individual readings rather than dropping rows, so one bad
variable never discards the rest of an observation. Bounds are deliberately
wider than the local climate: the aim is to catch encoding faults and stuck
sensors, not to trim genuine extremes. Every mask is counted and reported so
the effect of screening can be stated rather than assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rdu_temperature.pipeline import schema

REPORT_COLUMNS: tuple[str, ...] = (
    "rule",
    schema.SOURCE,
    schema.STATION_ID,
    "column",
    "masked",
)


@dataclass(frozen=True)
class Range:
    """Inclusive bounds outside which a reading is treated as an error."""

    minimum: float
    maximum: float

    def outside(self, values: pd.Series) -> pd.Series:
        """Flag present readings that fall outside the bounds."""
        return values.notna() & ~values.between(self.minimum, self.maximum)


# Bounds chosen well outside anything the Piedmont produces. RDU's records sit
# near -16 and 41 degrees Celsius, so the temperature bound clears them easily.
PLAUSIBLE_RANGES: Mapping[str, Range] = {
    schema.TEMPERATURE_C: Range(-30.0, 50.0),
    schema.DEWPOINT_C: Range(-40.0, 35.0),
    schema.RELATIVE_HUMIDITY_PCT: Range(0.0, 100.0),
    schema.WIND_SPEED_MS: Range(0.0, 60.0),
    schema.WIND_DIRECTION_DEG: Range(0.0, 360.0),
    schema.WIND_GUST_MS: Range(0.0, 90.0),
    schema.PRESSURE_HPA: Range(900.0, 1100.0),
    schema.SEA_LEVEL_PRESSURE_HPA: Range(900.0, 1100.0),
    schema.PRECIPITATION_MM: Range(0.0, 250.0),
    schema.CLOUD_COVER_PCT: Range(0.0, 100.0),
    schema.VISIBILITY_M: Range(0.0, 40000.0),
    schema.SHORTWAVE_RADIATION_WM2: Range(0.0, 1400.0),
}

# Variables that vary continuously, so a long run of one value means a stuck
# sensor. Precipitation, cloud cover, and radiation hold legitimately still.
FLATLINE_VARIABLES: tuple[str, ...] = (
    schema.TEMPERATURE_C,
    schema.DEWPOINT_C,
    schema.PRESSURE_HPA,
    schema.SEA_LEVEL_PRESSURE_HPA,
)


@dataclass(frozen=True)
class ScreeningResult:
    """A screened panel and a count of what each rule masked."""

    frame: pd.DataFrame
    report: pd.DataFrame

    @property
    def masked(self) -> int:
        return int(self.report["masked"].sum()) if len(self.report) else 0


def flatline_mask(values: pd.Series, minimum_run: int) -> pd.Series:
    """Flag readings inside a run of identical values at least as long as given.

    A missing reading breaks a run, so an outage never merges two stretches
    that happen to share a value.
    """
    started = values.ne(values.shift())
    lengths = values.groupby(started.cumsum()).transform("size")
    return values.notna() & (lengths >= minimum_run)


@dataclass(frozen=True)
class Screener:
    """Apply every screening rule to an aligned panel."""

    ranges: Mapping[str, Range] = field(default_factory=lambda: PLAUSIBLE_RANGES)
    flatline_variables: tuple[str, ...] = FLATLINE_VARIABLES
    # Rounding can put a dew point a touch above the temperature; a real
    # inversion of the two cannot happen.
    dewpoint_tolerance_c: float = 0.5
    flatline_hours: int = 24

    def screen(self, frame: pd.DataFrame) -> ScreeningResult:
        screened = frame.copy()
        records: list[dict[str, Any]] = []

        for column, limit in self.ranges.items():
            if column in screened.columns:
                self._mask(
                    screened, records, "range", column, limit.outside(screened[column])
                )

        if {schema.DEWPOINT_C, schema.TEMPERATURE_C} <= set(screened.columns):
            excess = screened[schema.DEWPOINT_C] - screened[schema.TEMPERATURE_C]
            self._mask(
                screened,
                records,
                "dewpoint_above_temperature",
                schema.DEWPOINT_C,
                excess > self.dewpoint_tolerance_c,
            )

        keys = [schema.SOURCE, schema.STATION_ID]
        for column in self.flatline_variables:
            if column not in screened.columns:
                continue
            stuck = screened.groupby(keys, sort=False)[column].transform(
                flatline_mask, self.flatline_hours
            )
            self._mask(screened, records, "flatline", column, stuck.astype(bool))

        report = pd.DataFrame(records, columns=list(REPORT_COLUMNS))
        return ScreeningResult(frame=screened, report=report)

    @staticmethod
    def _mask(
        frame: pd.DataFrame,
        records: list[dict[str, Any]],
        rule: str,
        column: str,
        mask: pd.Series,
    ) -> None:
        mask = mask.fillna(False).astype(bool)
        if not mask.any():
            return
        counts = (
            frame.loc[mask]
            .groupby([schema.SOURCE, schema.STATION_ID], sort=False)
            .size()
        )
        records.extend(
            {
                "rule": rule,
                schema.SOURCE: source,
                schema.STATION_ID: station,
                "column": column,
                "masked": int(count),
            }
            for (source, station), count in counts.items()
        )
        frame[column] = frame[column].mask(mask)
