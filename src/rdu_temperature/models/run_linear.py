"""Backtest the two linear regression models and write the 2026 forecast.

Run from the repository root with:

    python -m rdu_temperature.models.run_linear

Two comparisons, matching the two fold sets the other models are scored on.

1. **Calendar linear model** (``linear``) on the same seasonal and rolling
   folds as ``evaluation.backtest``, against climatology and persistence. The
   ``climatology`` and ``persistence`` rows here should reproduce
   ``backtest_summary.csv`` exactly, which is the check that the folds match.
2. **GFS linear model** (``linear_gfs``) on the folds a covariate model can be
   trained for: a GFS run covers the whole horizon and the covariate history
   behind it spans a year. This is the same rule ``run_covariate_prophet``
   applies, so the numbers line up with ``covariate_backtest_summary.csv``.
   The reference is raw GFS, because beating climatology only says the model
   knows what September looks like; beating raw GFS says the regression earns
   its place.

Finally both models are refit on all history and forecast the real period.

This runner does not import Prophet, so it runs without it installed. That is
why the fold-selection helpers below repeat ``evaluation.backtest`` rather
than import it: that module imports Prophet at load time.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdu_temperature.evaluation import compare
from rdu_temperature.features import forecast_covariates, prophet_frame
from rdu_temperature.models.baselines import Climatology, SeasonalPersistence
from rdu_temperature.models.linear_model import (
    GfsLinearModel,
    LinearTemperatureModel,
    RawGfsForecast,
)
from rdu_temperature.pipeline.clean_gfs import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    CleanGfsApp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"

HORIZON = prophet_frame.FORECAST_HOURS
# The same bars as evaluation.backtest.
MINIMUM_TRAINING_DAYS = 365
ROLLING_STEP_DAYS = 14

OUTPUTS = {
    "summary": METRICS_DIR / "linear_backtest_summary.csv",
    "comparison": METRICS_DIR / "linear_comparison.csv",
    "coefficients": METRICS_DIR / "linear_coefficients.csv",
    "forecast": PREDICTIONS_DIR / "linear_forecast.csv",
}

Factory = Callable[[], Any]


class FullPanelApp(CleanGfsApp):
    """Every ingested run, under the filenames ``run_covariate_prophet`` uses."""

    panel_filename = "gfs_full_panel.parquet"
    report_filename = "gfs_full_screening_report.csv"


# --- folds -------------------------------------------------------------------


def seasonal_cutoffs(frame: pd.DataFrame, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    """Anniversaries of the real cutoff with a full horizon and a year before."""
    first = frame[prophet_frame.DS].min()
    end = frame[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    folds: list[pd.Timestamp] = []
    candidate = cutoff - pd.DateOffset(years=1)
    while candidate - first >= pd.Timedelta(days=MINIMUM_TRAINING_DAYS):
        if candidate + pd.Timedelta(hours=HORIZON) <= end:
            folds.append(pd.Timestamp(candidate))
        candidate -= pd.DateOffset(years=1)
    return sorted(folds)


def rolling_cutoffs(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """A cutoff every fortnight from one year in to one horizon before the end."""
    first = frame[prophet_frame.DS].min()
    end = frame[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    folds: list[pd.Timestamp] = []
    candidate = first + pd.Timedelta(days=MINIMUM_TRAINING_DAYS)
    while candidate + pd.Timedelta(hours=HORIZON) <= end:
        folds.append(pd.Timestamp(candidate))
        candidate += pd.Timedelta(days=ROLLING_STEP_DAYS)
    return folds


def trainable_cutoffs(
    frame: pd.DataFrame, panel: pd.DataFrame, cutoffs: Sequence[pd.Timestamp]
) -> list[pd.Timestamp]:
    """Covered folds whose covariate history spans a year (as in PR #7)."""
    trainable: list[pd.Timestamp] = []
    for cutoff in forecast_covariates.covered_cutoffs(panel, cutoffs, HORIZON):
        history = frame.loc[frame[prophet_frame.DS] < cutoff]
        joined = forecast_covariates.attach_available(history, panel, cutoff)
        usable = joined.dropna(
            subset=[forecast_covariates.PREFIX + "temperature_c", prophet_frame.Y]
        )
        if usable.empty:
            continue
        span = usable[prophet_frame.DS].max() - usable[prophet_frame.DS].min()
        if span >= pd.Timedelta(days=MINIMUM_TRAINING_DAYS):
            trainable.append(cutoff)
    return trainable


# --- scoring -----------------------------------------------------------------


def score_fold(
    frame: pd.DataFrame, factory: Factory, cutoff: pd.Timestamp
) -> dict[str, Any]:
    history = frame.loc[frame[prophet_frame.DS] < cutoff]
    truth = frame.loc[
        frame[prophet_frame.DS] >= cutoff, [prophet_frame.DS, prophet_frame.Y]
    ]
    forecast = factory().fit(history).forecast(cutoff, HORIZON)
    scored = forecast.merge(truth, on=prophet_frame.DS).dropna(subset=[prophet_frame.Y])
    error = scored["yhat"] - scored[prophet_frame.Y]
    bounds = scored[["yhat_lower", "yhat_upper"]]
    coverage = (
        float("nan")
        if bounds.isna().all(axis=None)
        else float(scored[prophet_frame.Y].between(*bounds.T.to_numpy()).mean())
    )
    return {
        "cutoff": cutoff,
        "scored_hours": len(scored),
        "mae_c": float(error.abs().mean()),
        "rmse_c": float(np.sqrt((error**2).mean())),
        "bias_c": float(error.mean()),
        "max_abs_error_c": float(error.abs().max()),
        "interval_coverage": coverage,
    }


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-fold rows plus an hour-weighted ``all`` row, as backtest writes."""
    per_fold = pd.DataFrame(rows)
    w = per_fold["scored_hours"]
    total: dict[str, Any] = {
        "cutoff": compare.TOTAL_ROW,
        "scored_hours": int(w.sum()),
        "rmse_c": float(np.sqrt(np.average(per_fold["rmse_c"] ** 2, weights=w))),
        "max_abs_error_c": float(per_fold["max_abs_error_c"].max()),
    }
    for column in ("mae_c", "bias_c", "interval_coverage"):
        present = per_fold[column].notna()
        total[column] = (
            float(np.average(per_fold.loc[present, column], weights=w[present]))
            if present.any()
            else float("nan")
        )
    return pd.concat([per_fold, pd.DataFrame([total])], ignore_index=True)


def run_fold_set(
    frame: pd.DataFrame,
    name: str,
    cutoffs: Sequence[pd.Timestamp],
    models: Mapping[str, Factory],
) -> pd.DataFrame:
    print(f"\n{name}: {len(cutoffs)} fold(s)", flush=True)
    summaries = []
    for model, factory in models.items():
        rows = [score_fold(frame, factory, cutoff) for cutoff in cutoffs]
        summary = summarize(rows).assign(model=model, folds=name)
        total = summary.iloc[-1]
        print(
            f"  {model:<18s} MAE {total['mae_c']:5.2f} C  RMSE {total['rmse_c']:5.2f} C"
            f"  bias {total['bias_c']:+5.2f} C",
            flush=True,
        )
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def paired(
    summary: pd.DataFrame, pairs: Sequence[tuple[str, str, str]]
) -> pd.DataFrame:
    """Bootstrap each (challenger, reference, fold set) difference in MAE."""
    rows = [
        compare.compare(summary, challenger, reference, folds).as_row()
        for challenger, reference, folds in pairs
    ]
    table = pd.DataFrame(rows)
    print("\nPaired MAE differences (negative = challenger better):")
    for _, row in table.iterrows():
        print(
            f"  {row['challenger']:<18s} vs {row['reference']:<12s} "
            f"[{row['folds']:<17s}] {row['mean_difference']:+.3f} C  "
            f"95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  "
            f"wins {row['challenger_wins']}/{row['n_folds']}"
        )
    return table


# --- entry point -------------------------------------------------------------


def load_panel(rebuild: bool) -> pd.DataFrame:
    try:
        FullPanelApp(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR).run(overwrite=rebuild)
    except FileExistsError:
        print("Reusing the existing GFS panel; pass --rebuild-panel to rebuild.")
    return forecast_covariates.load_panel(
        DEFAULT_OUTPUT_DIR / FullPanelApp.panel_filename
    )


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(temporary)
    temporary.replace(path)


def run(*, rebuild_panel: bool = False, overwrite: bool = False) -> dict[str, Path]:
    existing = [path for path in OUTPUTS.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it.")

    frame = prophet_frame.load()
    panel = load_panel(rebuild_panel)
    cutoff = prophet_frame.forecast_cutoff()

    univariate: dict[str, Factory] = {
        "linear": LinearTemperatureModel,
        "linear_no_anomaly": lambda: LinearTemperatureModel(use_anomaly=False),
        "climatology": Climatology,
        "persistence": SeasonalPersistence,
    }
    covariate: dict[str, Factory] = {
        "linear_gfs": lambda: GfsLinearModel(panel=panel),
        "raw_gfs": lambda: RawGfsForecast(panel=panel),
        "linear": LinearTemperatureModel,
        "climatology": Climatology,
    }

    seasonal = seasonal_cutoffs(frame, cutoff)
    rolling = rolling_cutoffs(frame)
    summary = pd.concat(
        [
            run_fold_set(frame, "seasonal", seasonal, univariate),
            run_fold_set(frame, "rolling", rolling, univariate),
            run_fold_set(
                frame,
                "gfs_seasonal",
                trainable_cutoffs(frame, panel, seasonal),
                covariate,
            ),
            run_fold_set(
                frame,
                "gfs_rolling",
                trainable_cutoffs(frame, panel, rolling),
                covariate,
            ),
        ],
        ignore_index=True,
    )
    comparison = paired(
        summary,
        [
            ("linear", "climatology", "seasonal"),
            ("linear", "climatology", "rolling"),
            ("linear", "linear_no_anomaly", "rolling"),
            ("linear_gfs", "raw_gfs", "gfs_seasonal"),
            ("linear_gfs", "raw_gfs", "gfs_rolling"),
            ("linear_gfs", "climatology", "gfs_rolling"),
        ],
    )

    # The submission forecast: both models refit on every hour before the cutoff.
    calendar = LinearTemperatureModel().fit(frame)
    gfs = GfsLinearModel(panel=panel).fit(frame)
    raw = RawGfsForecast(panel=panel).forecast(cutoff, HORIZON)
    forecast = (
        calendar.forecast(cutoff, HORIZON)
        .loc[:, [prophet_frame.DS, "timestamp_local", "yhat"]]
        .rename(columns={"yhat": "linear_c"})
        .merge(
            gfs.forecast(cutoff, HORIZON).loc[
                :, [prophet_frame.DS, "yhat", "yhat_lower", "yhat_upper"]
            ],
            on=prophet_frame.DS,
            how="left",
        )
        .rename(
            columns={
                "yhat": "linear_gfs_c",
                "yhat_lower": "linear_gfs_lower_c",
                "yhat_upper": "linear_gfs_upper_c",
            }
        )
        .merge(
            raw[[prophet_frame.DS, "yhat"]].rename(columns={"yhat": "raw_gfs_c"}),
            on=prophet_frame.DS,
            how="left",
        )
    )
    coefficients = pd.concat(
        [
            calendar.coefficients.to_frame().assign(model="linear"),
            gfs.coefficients.to_frame().assign(model="linear_gfs"),
        ]
    ).rename_axis("feature")
    print(f"\nFinal forecast: {len(forecast)} hours from {cutoff} UTC.")
    print(gfs.coefficients.round(3).to_string())

    _write(lambda p: summary.to_csv(p, index=False), OUTPUTS["summary"])
    _write(lambda p: comparison.to_csv(p, index=False), OUTPUTS["comparison"])
    _write(lambda p: coefficients.to_csv(p), OUTPUTS["coefficients"])
    _write(lambda p: forecast.to_csv(p, index=False), OUTPUTS["forecast"])
    for name, path in OUTPUTS.items():
        print(f"Wrote {name}: {path}")
    return OUTPUTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(rebuild_panel=args.rebuild_panel, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
