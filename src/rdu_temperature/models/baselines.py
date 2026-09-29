"""Naive baselines that give the Prophet numbers something to mean.

A mean absolute error of three degrees is neither good nor bad on its own. It
only becomes a result beside the two questions any reader will ask first: how
much better is this than a lookup table of what September usually does, and how
much better is it than assuming tomorrow repeats today.

Both baselines here answer one of those, and both are deliberately simple
enough that beating them is the minimum bar rather than an achievement.

Climatology is the one that matters. A univariate Prophet is, in substance, a
smooth parametric climatology: trend plus an annual cycle plus a daily cycle.
Scoring it against a direct empirical climatology isolates what Prophet's
machinery adds over counting. If the gap is small, that is the finding, and it
is worth stating plainly rather than discovering in the write-up.

Seasonal persistence repeats the last observed day. Flat carry-forward is the
textbook persistence baseline, but against an hourly series with a ten-degree
diurnal swing it is a strawman: it would lose to anything. Repeating the last
day keeps the diurnal shape and tests only the level, which is the honest
version of the same question, and the level is exactly where Prophet is weak.

Neither baseline produces an uncertainty interval, so both report missing
bounds rather than inventing a spread they did not estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rdu_temperature.features import prophet_frame

# Day of year runs to 366 so that a leap year indexes without a special case.
DAYS_IN_YEAR = 366
HOURS_PER_DAY = 24


def _forecast_frame(timestamps: pd.DatetimeIndex, yhat: np.ndarray) -> pd.DataFrame:
    """Shape a point forecast like the model frame the backtest expects.

    The interval columns are present but missing. A baseline that estimated no
    spread must not be scored as though it had predicted one.
    """
    series = pd.Series(timestamps, name=prophet_frame.DS)
    return pd.DataFrame(
        {
            prophet_frame.DS: series,
            "timestamp_local": prophet_frame.to_local(series),
            "yhat": yhat,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )


@dataclass
class Climatology:
    """The mean temperature for this hour of this time of year.

    Cells are averaged over a window of days either side of the target day of
    year, not just the matching date. Four years of history gives a single
    calendar hour only four observations, and the mean of four is mostly noise;
    widening to a fortnight puts sixty behind each cell while staying narrow
    against the annual cycle, which moves by well under a degree across the
    window at this latitude.
    """

    window_days: int = 7
    _means: np.ndarray | None = field(default=None, repr=False)
    _fallback: float = field(default=float("nan"), repr=False)

    def fit(self, history: pd.DataFrame) -> Climatology:
        observed = history.loc[history[prophet_frame.Y].notna()]
        if observed.empty:
            raise ValueError("Climatology needs at least one observation to fit.")

        day = observed[prophet_frame.DS].dt.dayofyear.to_numpy() - 1
        hour = observed[prophet_frame.DS].dt.hour.to_numpy()
        values = observed[prophet_frame.Y].to_numpy()

        totals = np.zeros((DAYS_IN_YEAR, HOURS_PER_DAY))
        counts = np.zeros((DAYS_IN_YEAR, HOURS_PER_DAY))
        np.add.at(totals, (day, hour), values)
        np.add.at(counts, (day, hour), 1.0)

        smoothed_totals = _wrap_sum(totals, self.window_days)
        smoothed_counts = _wrap_sum(counts, self.window_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._means = np.where(
                smoothed_counts > 0, smoothed_totals / smoothed_counts, np.nan
            )
        self._fallback = float(values.mean())
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        if self._means is None:
            raise RuntimeError("Call fit() before forecast().")
        timestamps = pd.date_range(cutoff, periods=hours, freq="h")
        predicted = self._means[timestamps.dayofyear - 1, timestamps.hour]
        # A cell the window never reached falls back to the overall mean rather
        # than to nothing, so the baseline always produces a full horizon.
        return _forecast_frame(
            timestamps, np.where(np.isnan(predicted), self._fallback, predicted)
        )


def _wrap_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Moving sum over the day axis, wrapping December round to January.

    Late December and early January are a continuous stretch of winter, so the
    window has to close the circle rather than run off the end of the array.
    """
    padded = np.concatenate([values[-window:], values, values[:window]], axis=0)
    kernel = np.ones(2 * window + 1)
    stacked = [
        np.convolve(padded[:, hour], kernel, mode="valid")
        for hour in range(values.shape[1])
    ]
    return np.stack(stacked, axis=1)


@dataclass
class SeasonalPersistence:
    """Repeat the most recently observed value for each hour of the day.

    Hours are filled by scanning back over a window rather than by taking the
    final twenty-four rows, so a gap in the last day before the cutoff is
    covered by the previous day's value at that hour instead of shifting the
    whole profile.
    """

    lookback_hours: int = 72
    _by_hour: pd.Series | None = field(default=None, repr=False)
    _fallback: float = field(default=float("nan"), repr=False)

    def fit(self, history: pd.DataFrame) -> SeasonalPersistence:
        recent = history.tail(self.lookback_hours)
        observed = recent.loc[recent[prophet_frame.Y].notna()]
        if observed.empty:
            raise ValueError(
                "Seasonal persistence needs an observation in the lookback window."
            )
        self._by_hour = observed.groupby(observed[prophet_frame.DS].dt.hour)[
            prophet_frame.Y
        ].last()
        self._fallback = float(observed[prophet_frame.Y].iloc[-1])
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        if self._by_hour is None:
            raise RuntimeError("Call fit() before forecast().")
        timestamps = pd.date_range(cutoff, periods=hours, freq="h")
        predicted = pd.Series(timestamps.hour).map(self._by_hour).fillna(self._fallback)
        return _forecast_frame(timestamps, predicted.to_numpy())
