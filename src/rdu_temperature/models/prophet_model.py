"""A univariate Prophet model for hourly RDU temperature.

Prophet decomposes a series into trend plus seasonalities. For temperature the
seasonalities carry essentially all of the signal and the trend carries almost
none, so the configuration here is mostly a matter of stopping Prophet from
spending flexibility where there is nothing to find.

The model is univariate by choice. Prophet regressors have to be supplied over
the forecast window as well as the history, and the project holds no covariate
values for a period that has not happened; the repository's deferred note on
numerical weather prediction archives is the route to changing that. Until
then, adding a regressor would mean forecasting the regressor first, which
moves the problem rather than solving it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from rdu_temperature.features import prophet_frame

# Prophet's uncertainty interval is sampled rather than solved, so yhat is
# reproducible but yhat_lower and yhat_upper drift between runs unless numpy is
# seeded first. Interval coverage is a reported metric, so it has to be stable.
DEFAULT_SEED = 20260917


def _below_warning(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING


# Prophet emits a block of cmdstanpy chatter per fit, which buries the progress
# a backtest prints. Setting the level does not hold: Prophet reopens the
# cmdstanpy logger at DEBUG inside every fit and attaches its own handler. A
# filter is consulted whatever the level has been changed to, so it survives.
# Warnings and errors still come through.
logging.getLogger("cmdstanpy").addFilter(_below_warning)


@dataclass(frozen=True)
class ProphetConfig:
    """Prophet settings chosen for hourly Piedmont temperature.

    Seasonality is additive. Multiplicative seasonality scales the trend, which
    is meaningless for a Celsius series that crosses zero every winter and
    would invert the diurnal swing on any hour the trend passed through it.

    The trend is held stiff. Prophet's changepoint machinery exists to track
    growth in business series; five years of temperature holds no trend worth
    fitting beyond a fraction of a degree, and a flexible one extrapolates that
    noise straight through a 336-hour horizon. A low changepoint prior keeps
    the trend near flat and leaves the seasonalities to do the work. This is
    the first parameter to tune, and the one most likely to matter.

    Weekly seasonality is off: air temperature has no weekly cycle to find, so
    enabling it fits noise and nothing else.

    The yearly order is below Prophet's default of 10. The annual temperature
    cycle is close to a single sinusoid, and the spare harmonics mostly chase
    individual warm and cold spells that will not recur on the same dates.

    The daily seasonality is one global curve. Splitting it per meteorological
    season is the more physical model, since the Piedmont's diurnal range is
    genuinely wider in spring and autumn than under summer humidity, but the
    backtest does not pay for it: the split moves weighted MAE by 0.017
    degrees in the wrong direction and loses on three folds of four. The
    diurnal shape is not where the error lives at this horizon, so the global
    curve is kept on parsimony. Set ``conditional_daily`` to reproduce the
    comparison.
    """

    yearly_fourier_order: int = 6
    daily_fourier_order: int = 6
    changepoint_prior_scale: float = 0.01
    seasonality_prior_scale: float = 10.0
    interval_width: float = 0.8
    conditional_daily: bool = False
    seed: int = DEFAULT_SEED

    def build(self) -> Prophet:
        """Return an unfitted Prophet configured for this series."""
        model = Prophet(
            growth="linear",
            yearly_seasonality=self.yearly_fourier_order,
            weekly_seasonality=False,
            daily_seasonality=(
                False if self.conditional_daily else self.daily_fourier_order
            ),
            seasonality_mode="additive",
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
        )
        if self.conditional_daily:
            for column in prophet_frame.SEASON_COLUMNS:
                model.add_seasonality(
                    name=f"daily_{column.removeprefix('is_')}",
                    period=1.0,
                    fourier_order=self.daily_fourier_order,
                    condition_name=column,
                )
        return model


@dataclass
class TemperatureProphet:
    """Fit Prophet on the target history and forecast forward from a cutoff."""

    config: ProphetConfig = ProphetConfig()
    model: Prophet | None = None

    def fit(self, history: pd.DataFrame) -> TemperatureProphet:
        """Fit on every row before the cutoff the caller has already applied.

        Prophet drops null ``y`` rows itself, so the uncovered hours pass
        through untouched rather than being filled.
        """
        observed = history[prophet_frame.Y].notna().sum()
        if observed < 2:
            raise ValueError(
                f"Prophet needs at least two observations to fit; got {observed}."
            )
        self.model = self.config.build()
        self.model.fit(history)
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        """Predict ``hours`` from ``cutoff``, in UTC and local time alike."""
        if self.model is None:
            raise RuntimeError("Call fit() before forecast().")
        future = prophet_frame.future_frame(cutoff, hours)
        # Seeded immediately before predict, which is where Prophet draws the
        # samples its interval is built from.
        np.random.seed(self.config.seed)
        predicted = self.model.predict(future)
        return pd.DataFrame(
            {
                prophet_frame.DS: predicted[prophet_frame.DS],
                "timestamp_local": prophet_frame.to_local(predicted[prophet_frame.DS]),
                "yhat": predicted["yhat"],
                "yhat_lower": predicted["yhat_lower"],
                "yhat_upper": predicted["yhat_upper"],
            }
        )

    def components(self) -> dict[str, Any]:
        """Return the fitted seasonality names and their periods, for writeup."""
        if self.model is None:
            raise RuntimeError("Call fit() before components().")
        return {
            name: {
                "period": term["period"],
                "fourier_order": term["fourier_order"],
                "condition_name": term["condition_name"],
            }
            for name, term in self.model.seasonalities.items()
        }
