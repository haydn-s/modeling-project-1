"""Linear regression forecast of hourly RDU temperature.

The forecast covers 336 hours that begin at the cutoff, and nothing observed
after the cutoff may be used. Every feature must therefore be known in advance
for any future hour. Calendar time is the only thing that qualifies, so the
model is a regression on smooth functions of time:

- a linear trend, in years since the start of the history,
- Fourier terms for the daily cycle (sine and cosine of hour of day),
- Fourier terms for the annual cycle (sine and cosine of day of year),
- daily-by-annual interactions, so the day-night swing can change size and
  timing through the year.

The regression alone is a climatology: it predicts what this hour of this time
of year usually looks like. A two-week forecast also depends on whether the
weather right before the cutoff was warmer or cooler than usual, and that
anomaly fades over a few days. The second stage measures the mean residual over
the last few days of history and adds it to the forecast with an exponential
decay. That correction uses only hours before the cutoff, so it leaks nothing.

The interval is the point forecast plus or minus a multiple of the residual
standard deviation, which assumes roughly normal, constant-spread errors.

:class:`GfsLinearModel` is the second linear model. Instead of calendar time
alone it regresses the observation on the GFS temperature forecast that was
issued before the cutoff (model output statistics). GFS skill falls with lead
time, so lead time and a GFS-by-lead interaction let the fitted slope shrink
toward the mean at long leads instead of trusting a twelve-day forecast as
much as a twelve-hour one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from rdu_temperature.features import forecast_covariates, prophet_frame
from rdu_temperature.pipeline import schema

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.25
# Two-sided 95 per cent normal interval.
INTERVAL_Z = 1.96


def calendar_features(
    timestamps: pd.Series | pd.DatetimeIndex,
    origin: pd.Timestamp,
    daily_harmonics: int = 3,
    annual_harmonics: int = 3,
    interaction_harmonics: int = 1,
) -> pd.DataFrame:
    """Build the design matrix from timestamps alone.

    Hour and day of year are taken from the naive UTC ``ds`` clock, which runs
    at a fixed offset from solar time all year and so has no daylight saving
    jump in the daily cycle.
    """
    stamps = pd.DatetimeIndex(timestamps)
    hour = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60.0
    day = stamps.dayofyear.to_numpy() - 1 + hour / HOURS_PER_DAY

    columns: dict[str, np.ndarray] = {
        "trend_years": (stamps - origin).total_seconds().to_numpy()
        / (DAYS_PER_YEAR * HOURS_PER_DAY * 3600)
    }
    for k in range(1, daily_harmonics + 1):
        angle = 2 * np.pi * k * hour / HOURS_PER_DAY
        columns[f"daily_sin_{k}"] = np.sin(angle)
        columns[f"daily_cos_{k}"] = np.cos(angle)
    for k in range(1, annual_harmonics + 1):
        angle = 2 * np.pi * k * day / DAYS_PER_YEAR
        columns[f"annual_sin_{k}"] = np.sin(angle)
        columns[f"annual_cos_{k}"] = np.cos(angle)
    for i in range(1, interaction_harmonics + 1):
        for j in range(1, interaction_harmonics + 1):
            for d_name in ("sin", "cos"):
                for a_name in ("sin", "cos"):
                    columns[f"daily_{d_name}_{i}_x_annual_{a_name}_{j}"] = (
                        columns[f"daily_{d_name}_{i}"] * columns[f"annual_{a_name}_{j}"]
                    )
    return pd.DataFrame(columns, index=stamps)


@dataclass
class LinearTemperatureModel:
    """Linear regression on calendar features, plus a decaying recent anomaly.

    Parameters
    ----------
    daily_harmonics, annual_harmonics, interaction_harmonics:
        How many Fourier pairs to use for each cycle and for their interaction.
    anomaly_days:
        How many days before the cutoff to average the residual over.
    anomaly_decay_days:
        e-folding time of the anomaly correction. Set ``use_anomaly=False`` to
        score the pure regression.
    """

    daily_harmonics: int = 3
    annual_harmonics: int = 3
    interaction_harmonics: int = 1
    use_anomaly: bool = True
    anomaly_days: float = 3.0
    anomaly_decay_days: float = 2.0

    _regression: LinearRegression | None = field(default=None, repr=False)
    _origin: pd.Timestamp | None = field(default=None, repr=False)
    _residual_std: float = field(default=float("nan"), repr=False)
    _anomaly: float = field(default=0.0, repr=False)
    _last_observed: pd.Timestamp | None = field(default=None, repr=False)

    def _features(self, timestamps: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        assert self._origin is not None
        return calendar_features(
            timestamps,
            self._origin,
            self.daily_harmonics,
            self.annual_harmonics,
            self.interaction_harmonics,
        )

    def fit(self, history: pd.DataFrame) -> LinearTemperatureModel:
        observed = history.loc[history[prophet_frame.Y].notna()]
        if observed.empty:
            raise ValueError("LinearTemperatureModel needs observations to fit.")

        timestamps = observed[prophet_frame.DS]
        self._origin = pd.Timestamp(timestamps.min())
        x = self._features(timestamps)
        y = observed[prophet_frame.Y].to_numpy(dtype="float64")

        self._regression = LinearRegression().fit(x, y)
        residuals = y - self._regression.predict(x)
        self._residual_std = float(np.std(residuals, ddof=x.shape[1] + 1))

        self._last_observed = pd.Timestamp(timestamps.max())
        window_start = self._last_observed - pd.Timedelta(days=self.anomaly_days)
        recent = (timestamps > window_start).to_numpy()
        self._anomaly = float(residuals[recent].mean()) if recent.any() else 0.0
        return self

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients by feature name, for the write-up."""
        if self._regression is None:
            raise RuntimeError("Call fit() before reading coefficients.")
        names = self._regression.feature_names_in_
        return pd.Series(self._regression.coef_, index=names).rename("coefficient")

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        if self._regression is None or self._last_observed is None:
            raise RuntimeError("Call fit() before forecast().")

        timestamps = pd.date_range(cutoff, periods=hours, freq="h")
        yhat = self._regression.predict(self._features(timestamps))

        if self.use_anomaly:
            lead_days = (
                timestamps - self._last_observed
            ).total_seconds().to_numpy() / (HOURS_PER_DAY * 3600)
            yhat = yhat + self._anomaly * np.exp(-lead_days / self.anomaly_decay_days)

        series = pd.Series(timestamps, name=prophet_frame.DS)
        spread = INTERVAL_Z * self._residual_std
        return pd.DataFrame(
            {
                prophet_frame.DS: series,
                "timestamp_local": prophet_frame.to_local(series),
                "yhat": yhat,
                "yhat_lower": yhat - spread,
                "yhat_upper": yhat + spread,
            }
        )


GFS_TEMPERATURE = f"{forecast_covariates.PREFIX}{schema.TEMPERATURE_C}"


def gfs_features(rows: pd.DataFrame, daily_harmonics: int = 2) -> pd.DataFrame:
    """Design matrix for :class:`GfsLinearModel` from joined GFS rows.

    ``rows`` carries the prefixed GFS temperature, the valid time and the lead
    in hours, as both :func:`forecast_covariates.training_pairs` and
    :func:`forecast_covariates.forecast_covariates` return them.
    """
    valid = pd.DatetimeIndex(rows[schema.VALID_TIME_UTC])
    gfs = rows[GFS_TEMPERATURE].to_numpy(dtype="float64")
    lead_days = rows[schema.LEAD_HOURS].to_numpy(dtype="float64") / HOURS_PER_DAY
    hour = valid.hour.to_numpy()
    day = valid.dayofyear.to_numpy() - 1

    columns: dict[str, np.ndarray] = {
        "gfs_temperature_c": gfs,
        "lead_days": lead_days,
        "gfs_x_lead_days": gfs * lead_days,
    }
    # GFS has a systematic diurnal and seasonal bias at a single grid point,
    # which a few smooth calendar terms can absorb.
    for k in range(1, daily_harmonics + 1):
        angle = 2 * np.pi * k * hour / HOURS_PER_DAY
        columns[f"daily_sin_{k}"] = np.sin(angle)
        columns[f"daily_cos_{k}"] = np.cos(angle)
    angle = 2 * np.pi * day / DAYS_PER_YEAR
    columns["annual_sin_1"] = np.sin(angle)
    columns["annual_cos_1"] = np.cos(angle)
    return pd.DataFrame(columns, index=rows.index)


@dataclass
class GfsLinearModel:
    """Linear regression of observed temperature on the GFS forecast.

    Trained on every (forecast, observation) pair whose run was initialised
    and whose valid hour falls before the cutoff, at every lead, so the
    lead-time terms see the same range of leads the forecast will use.
    """

    panel: pd.DataFrame
    daily_harmonics: int = 2

    _regression: LinearRegression | None = field(default=None, repr=False)
    _residual_std: float = field(default=float("nan"), repr=False)
    training_pairs: int = 0

    def fit(self, history: pd.DataFrame) -> GfsLinearModel:
        cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
        target = pd.DataFrame(
            {
                schema.TIMESTAMP_UTC: history[prophet_frame.DS],
                schema.TEMPERATURE_C: history[prophet_frame.Y],
            }
        )
        pairs = forecast_covariates.training_pairs(self.panel, target, cutoff)
        pairs = pairs.dropna(subset=[GFS_TEMPERATURE])
        if len(pairs) < 50:
            raise ValueError(
                f"Only {len(pairs)} GFS/observation pair(s) before {cutoff}; "
                "fetch runs covering the training window."
            )
        x = gfs_features(pairs, self.daily_harmonics)
        y = pairs[forecast_covariates.OBSERVED_TEMPERATURE_C].to_numpy("float64")
        self._regression = LinearRegression().fit(x, y)
        residuals = y - self._regression.predict(x)
        self._residual_std = float(np.std(residuals, ddof=x.shape[1] + 1))
        self.training_pairs = len(pairs)
        return self

    @property
    def coefficients(self) -> pd.Series:
        if self._regression is None:
            raise RuntimeError("Call fit() before reading coefficients.")
        names = self._regression.feature_names_in_
        return pd.Series(self._regression.coef_, index=names).rename("coefficient")

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        if self._regression is None:
            raise RuntimeError("Call fit() before forecast().")
        rows = forecast_covariates.forecast_covariates(self.panel, cutoff, hours)
        rows = rows.dropna(subset=[GFS_TEMPERATURE]).reset_index(drop=True)
        yhat = self._regression.predict(gfs_features(rows, self.daily_harmonics))
        series = pd.Series(
            rows[schema.VALID_TIME_UTC].to_numpy(), name=prophet_frame.DS
        )
        spread = INTERVAL_Z * self._residual_std
        return pd.DataFrame(
            {
                prophet_frame.DS: series,
                "timestamp_local": prophet_frame.to_local(series),
                "yhat": yhat,
                "yhat_lower": yhat - spread,
                "yhat_upper": yhat + spread,
            }
        )


@dataclass
class RawGfsForecast:
    """The GFS temperature forecast read straight off the file.

    The reference :class:`GfsLinearModel` has to beat. Kept here rather than
    imported from ``covariate_prophet`` so this module does not need Prophet.
    """

    panel: pd.DataFrame

    def fit(self, history: pd.DataFrame) -> RawGfsForecast:
        del history
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        rows = forecast_covariates.forecast_covariates(self.panel, cutoff, hours)
        series = pd.Series(
            rows[schema.VALID_TIME_UTC].to_numpy(), name=prophet_frame.DS
        )
        return pd.DataFrame(
            {
                prophet_frame.DS: series,
                "timestamp_local": prophet_frame.to_local(series),
                "yhat": rows[GFS_TEMPERATURE].to_numpy(),
                "yhat_lower": np.nan,
                "yhat_upper": np.nan,
            }
        )
