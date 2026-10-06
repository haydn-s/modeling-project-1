"""Prophet with GFS forecasts as regressors.

The univariate model ties an empirical climatology: over 104 rolling folds the
two differ by 0.05 degrees with an interval spanning zero. That is not a tuning
problem. A model built from trend and seasonality knows what a September
fortnight usually looks like and has no way to know which September it is being
asked about, so it sits two to three degrees off and the error is almost all
bias. Nothing inside Prophet reaches that.

GFS does know. A run initialised before the cutoff carries the atmosphere's
actual state into the forecast period, which is the one piece of information
the univariate model is missing. Handing it to Prophet as a regressor is the
cheapest honest way to find out whether it closes the gap.

Two things make this different from the univariate model rather than a variant
of it.

**Training is restricted to covered hours.** A regressor has to exist on every
row Prophet fits, and the archive is fetched for a sample of fortnights rather
than for five continuous years, so the covered history is a few thousand hours
scattered across the window instead of 43,824 contiguous ones. The model
selects those rows and says how many it got. That is a real cost: the yearly
seasonality is estimated from a sample of the year, not from all of it, which
is why this is a different model and not a better one by construction.

**The seasonalities fight the regressor, so they are off by default.** GFS
already contains a diurnal cycle, so Prophet's own daily seasonality fits
something its regressor has largely supplied. This was left as a switch for
the backtest to settle, and it has: over the nineteen rolling folds that have
a year of covariate history, leaving them on costs 0.94 C of MAE (3.94
against 3.00) and lands *worse than reading the GFS number straight off the
file*, with the same −3 C bias the univariate model carries. Restricted
training is why. The seasonalities are estimated from the fifth of the hours
that have a covariate, and a yearly term fit on a sample of fortnights is
confident and wrong; on the earliest folds, where that sample is thinnest, it
misses by six degrees. With them off the model is a pure statistical
correction of GFS, and it is the first thing here to beat climatology on a
paired test: −0.75 C, 95% CI [−1.44, −0.12], winning 14 of 19 folds.

The switch stays, because the finding is contingent on sparse covariate
history. A fetch covering continuous years rather than a sample of fortnights
would be grounds to ask the question again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from prophet import Prophet

from rdu_temperature.features import forecast_covariates, prophet_frame
from rdu_temperature.models.prophet_model import DEFAULT_SEED, ProphetConfig
from rdu_temperature.pipeline import schema

logging.getLogger("cmdstanpy").addFilter(
    lambda record: record.levelno >= logging.WARNING
)

# The covariate most likely to carry the signal, kept as the default so a run
# that adds the rest can be compared against it.
DEFAULT_REGRESSORS: tuple[str, ...] = (
    f"{forecast_covariates.PREFIX}{schema.TEMPERATURE_C}",
)

ALL_REGRESSORS: tuple[str, ...] = tuple(
    f"{forecast_covariates.PREFIX}{column}"
    for column in forecast_covariates.COVARIATE_COLUMNS
)


@dataclass
class CovariateProphet:
    """Fit Prophet on the hours a GFS covariate exists for.

    The panel is held rather than passed, because the backtest asks a model to
    forecast from a cutoff and nothing more; where the covariates for that
    window come from is the model's business.
    """

    panel: pd.DataFrame
    config: ProphetConfig = field(default_factory=ProphetConfig)
    regressors: tuple[str, ...] = DEFAULT_REGRESSORS
    # Prophet's own trend and seasonality, on top of the regressor. Off makes
    # the model a pure statistical correction of GFS; on lets it also carry
    # whatever seasonal structure GFS gets wrong. Off by default because on
    # measures worse than the uncorrected forecast -- see the module docstring.
    seasonality: bool = False
    cutoff: pd.Timestamp | None = None
    model: Prophet | None = None
    training_hours: int = 0

    def _build(self) -> Prophet:
        model = Prophet(
            growth="linear" if self.seasonality else "flat",
            yearly_seasonality=(
                self.config.yearly_fourier_order if self.seasonality else False
            ),
            weekly_seasonality=False,
            daily_seasonality=(
                self.config.daily_fourier_order if self.seasonality else False
            ),
            seasonality_mode="additive",
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=self.config.interval_width,
        )
        for regressor in self.regressors:
            model.add_regressor(regressor, standardize=True)
        return model

    def fit(self, history: pd.DataFrame) -> CovariateProphet:
        """Fit on the rows of ``history`` that have every regressor.

        The cutoff is taken as one hour past the last row, matching the
        half-open window the backtest splits on, so only runs that existed by
        then are joined.
        """
        self.cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
        joined = forecast_covariates.attach_available(history, self.panel, self.cutoff)
        usable = joined.dropna(subset=[*self.regressors, prophet_frame.Y])
        if len(usable) < 2:
            raise ValueError(
                f"Only {len(usable)} hour(s) have both an observation and a "
                "GFS covariate; fetch the runs covering the training window."
            )
        self.training_hours = len(usable)
        self.model = self._build()
        self.model.fit(usable)
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        """Predict ``hours`` from ``cutoff``, using that window's covariates."""
        if self.model is None:
            raise RuntimeError("Call fit() before forecast().")
        future = prophet_frame.future_frame(cutoff, hours)
        covariates = forecast_covariates.forecast_covariates(self.panel, cutoff, hours)
        joined = forecast_covariates.attach_to_prophet_frame(future, covariates)

        np.random.seed(self.config.seed)
        predicted = self.model.predict(joined)
        return pd.DataFrame(
            {
                prophet_frame.DS: predicted[prophet_frame.DS],
                "timestamp_local": prophet_frame.to_local(predicted[prophet_frame.DS]),
                "yhat": predicted["yhat"],
                "yhat_lower": predicted["yhat_lower"],
                "yhat_upper": predicted["yhat_upper"],
            }
        )


@dataclass
class RawGfs:
    """The GFS temperature forecast, uncorrected.

    The baseline every corrected model has to beat. If a statistical
    correction cannot improve on reading the number straight off the model,
    the correction is not earning its place, and that is worth being able to
    show rather than assume.
    """

    panel: pd.DataFrame
    seed: int = DEFAULT_SEED

    def fit(self, history: pd.DataFrame) -> RawGfs:
        # Nothing is learned: the forecast is taken as issued. Accepting the
        # history keeps the interface the backtest expects.
        del history
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        covariates = forecast_covariates.forecast_covariates(self.panel, cutoff, hours)
        valid_times = covariates[schema.VALID_TIME_UTC]
        return pd.DataFrame(
            {
                prophet_frame.DS: valid_times,
                "timestamp_local": prophet_frame.to_local(valid_times),
                "yhat": covariates[
                    f"{forecast_covariates.PREFIX}{schema.TEMPERATURE_C}"
                ],
                "yhat_lower": np.nan,
                "yhat_upper": np.nan,
            }
        )
