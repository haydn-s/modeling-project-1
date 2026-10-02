"""Join GFS forecasts to observations, for whichever model wants them.

This is the shared surface between the forecast panel and the models. Prophet
wants named columns beside its ``ds``; XGBoost wants a wide numeric matrix.
Neither preference belongs in the panel, so it stays keyed by initialisation
and valid time and this module does the joining.

The useful shape is a **pair**: what the model predicted for an hour, and what
was actually measured in it. Pairs are what let a correction be learned rather
than assumed, and they are the same rows whichever estimator consumes them --
which is the point of putting them here instead of beside one model.

Pairs also explain why GFS is worth the download. A univariate Prophet carries
no information about which fortnight it is forecasting, which is why it sits
two to three degrees off and why a lookup table matches it. GFS does carry
that information. Learning the mapping from its prediction to the observation
is the operational practice of model output statistics, and it needs matched
history: the forecast the model issued at a given lead, beside the truth.

Two guards matter here, and both are enforced rather than documented.

Nothing is paired across the cutoff. An observation inside the forecast period
would be the answer, so the join refuses any valid time at or after the cutoff
when it is building training data.

Lead time travels with every row. A forecast twelve hours out and one twelve
days out are not the same covariate, and a model given them unlabelled will
average two very different error distributions. The ``interpolated`` flag
travels for the same reason: past 120 hours the product is three-hourly, and
two thirds of a fourteen-day horizon is filled between published leads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rdu_temperature.features import prophet_frame
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "gfs_forecast_panel.parquet"

# The covariate columns, in a fixed order so a feature matrix built today has
# the same column order as one built next week.
COVARIATE_COLUMNS: tuple[str, ...] = (
    schema.TEMPERATURE_C,
    schema.DEWPOINT_C,
    schema.WIND_SPEED_MS,
    schema.WIND_DIRECTION_DEG,
    schema.CLOUD_COVER_PCT,
    schema.PRECIPITATION_MM,
)

# Prefixed on the way out, so that a joined frame never has two columns called
# temperature_c meaning different things.
PREFIX = "gfs_"

OBSERVED_TEMPERATURE_C = "observed_temperature_c"


def load_panel(path: Path = DEFAULT_PANEL_PATH) -> pd.DataFrame:
    """Read the cleaned GFS forecast panel."""
    if not path.exists():
        raise FileNotFoundError(
            f"No GFS panel at {path}; run rdu_temperature.pipeline.clean_gfs."
        )
    return pd.read_parquet(path)


def _prefixed(panel: pd.DataFrame) -> pd.DataFrame:
    """Rename the covariates so they cannot collide with observed columns."""
    present = [column for column in COVARIATE_COLUMNS if column in panel.columns]
    renamed = {column: f"{PREFIX}{column}" for column in present}
    keep = [
        schema.INIT_TIME_UTC,
        schema.VALID_TIME_UTC,
        schema.LEAD_HOURS,
        INTERPOLATED,
        *present,
    ]
    return panel.loc[:, keep].rename(columns=renamed)


def forecast_covariates(
    panel: pd.DataFrame, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
) -> pd.DataFrame:
    """The covariates for the window a forecast from ``cutoff`` has to cover.

    The run is the latest one initialised before the cutoff: the most skilful
    forecast legitimately available at that moment. Where several runs cover an
    hour the later one wins, which is the operational choice too.
    """
    legitimate = panel.loc[panel[schema.INIT_TIME_UTC] < cutoff]
    if legitimate.empty:
        raise ValueError(f"No GFS run in the panel was initialised before {cutoff}.")

    end = cutoff + pd.Timedelta(hours=hours)
    window = legitimate.loc[
        legitimate[schema.VALID_TIME_UTC].between(cutoff, end, inclusive="left")
    ]
    freshest = (
        window.sort_values([schema.VALID_TIME_UTC, schema.INIT_TIME_UTC])
        .groupby(schema.VALID_TIME_UTC, as_index=False)
        .last()
    )
    return _prefixed(freshest).reset_index(drop=True)


def training_pairs(
    panel: pd.DataFrame, target: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Every forecast hour that has a measured temperature to learn against.

    Only hours strictly before the cutoff are returned. An observation at or
    after it is the answer the project is scored on, and pairing one here
    would train a correction on the thing it is meant to predict.
    """
    observations = (
        target.loc[:, [schema.TIMESTAMP_UTC, schema.TEMPERATURE_C]]
        .rename(
            columns={
                schema.TIMESTAMP_UTC: schema.VALID_TIME_UTC,
                schema.TEMPERATURE_C: OBSERVED_TEMPERATURE_C,
            }
        )
        .assign(
            **{
                schema.VALID_TIME_UTC: lambda frame: pd.to_datetime(
                    frame[schema.VALID_TIME_UTC]
                )
            }
        )
    )

    usable = panel.loc[
        (panel[schema.INIT_TIME_UTC] < cutoff) & (panel[schema.VALID_TIME_UTC] < cutoff)
    ]
    paired = _prefixed(usable).merge(
        observations, on=schema.VALID_TIME_UTC, how="inner"
    )
    paired = paired.loc[paired[OBSERVED_TEMPERATURE_C].notna()]
    return paired.sort_values(
        [schema.INIT_TIME_UTC, schema.VALID_TIME_UTC], kind="stable"
    ).reset_index(drop=True)


def forecast_error(pairs: pd.DataFrame) -> pd.Series:
    """What the model got wrong, as forecast minus observed.

    The same sign convention the backtest uses, so a positive number means too
    warm in both places.
    """
    return pairs[f"{PREFIX}{schema.TEMPERATURE_C}"] - pairs[OBSERVED_TEMPERATURE_C]


def regressor_columns(frame: pd.DataFrame) -> list[str]:
    """The covariate columns on a joined frame, in their fixed order."""
    return [column for column in frame.columns if column.startswith(PREFIX)]


def covered_cutoffs(
    panel: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    hours: int = prophet_frame.FORECAST_HOURS,
) -> list[pd.Timestamp]:
    """The cutoffs whose whole horizon a legitimate run covers.

    A covariate model cannot score a fold it has no forecast for, and a paired
    comparison needs every model on the same folds. Filtering the fold set to
    the covered ones keeps the pairing exact, which matters more than scoring
    the univariate models on folds their rivals cannot reach: an average taken
    over a different set of fortnights is not comparable.
    """
    covered: list[pd.Timestamp] = []
    for cutoff in cutoffs:
        try:
            window = forecast_covariates(panel, cutoff, hours)
        except ValueError:
            continue
        if len(window) == hours and not window.isna().any(axis=None):
            covered.append(cutoff)
    return covered


def attach_available(
    frame: pd.DataFrame, panel: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Join whatever covariates the panel holds, leaving the rest missing.

    The counterpart to :func:`attach_to_prophet_frame`, for training rather
    than forecasting. A fetch covers a handful of fortnights out of five
    years, so most of the history has no covariate and never will; a model
    that wants one selects the rows that have it. Demanding full coverage
    here would mean refusing to train at all.

    Only runs initialised before the cutoff are joined, so the covariate on a
    training row is one that would have existed at the time.
    """
    legitimate = panel.loc[panel[schema.INIT_TIME_UTC] < cutoff]
    freshest = (
        _prefixed(legitimate)
        .sort_values([schema.VALID_TIME_UTC, schema.INIT_TIME_UTC])
        .groupby(schema.VALID_TIME_UTC, as_index=False)
        .last()
        .rename(columns={schema.VALID_TIME_UTC: prophet_frame.DS})
    )
    return frame.merge(freshest, on=prophet_frame.DS, how="left")


def attach_to_prophet_frame(
    frame: pd.DataFrame, covariates: pd.DataFrame
) -> pd.DataFrame:
    """Join covariates onto a Prophet frame by valid hour.

    Prophet matches a regressor to a row by ``ds`` and will not tolerate a gap
    in one, so this reports what it could not cover rather than quietly
    leaving nulls for Prophet to reject later.
    """
    joined = frame.merge(
        covariates.rename(columns={schema.VALID_TIME_UTC: prophet_frame.DS}),
        on=prophet_frame.DS,
        how="left",
    )
    regressors = [column for column in joined.columns if column.startswith(PREFIX)]
    missing = int(joined.loc[:, regressors].isna().any(axis=1).sum())
    if missing:
        raise ValueError(
            f"{missing} of {len(joined)} hour(s) have no GFS covariate. Prophet "
            "rejects a regressor with gaps; fetch the runs covering them, or "
            "restrict the frame to the hours the panel covers."
        )
    return joined
