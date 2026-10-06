"""XGBoost on the shared covariate backtest folds.

Puts the GFS-correcting XGBoost on the exact fold protocol the covariate
Prophet comparison uses, so the two models can be compared fold for fold.

Run from the repository root with:

    python -m rdu_temperature.models.xgboost_backtest

The training recipe mirrors ``run_xgboost.py``: the target is the residual
(observed temperature minus raw GFS temperature), eight features are selected
per fold, early stopping watches the latest historical run, and the final fit
refits on all pre-cutoff data with ``n_estimators = best_iteration + 1``.

The one deliberate difference from ``run_xgboost_rolling.py`` is the dedupe.
That script collapses the panel globally to the latest run per valid hour,
which is fine for a single final holdout but wrong for backtest folds: seen
from an early cutoff, the global choice can keep a run initialized *after*
the cutoff and drop the legitimate older one. Here the collapse is
cutoff-aware -- among the runs initialized before the fold's cutoff, the
latest per valid hour wins -- the same operational choice
``forecast_covariates`` makes for the forecast window.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdu_temperature.evaluation import compare
from rdu_temperature.evaluation.backtest import (
    MINIMUM_TRAINING_DAYS,
    Backtest,
    FoldResult,
    FoldSet,
    rolling_cutoffs,
    seasonal_cutoffs,
    summarize,
)
from rdu_temperature.features import forecast_covariates, prophet_frame, xgboost_frame
from rdu_temperature.models.run_xgboost import DEFAULT_GFS_MODEL_PARAMS
from rdu_temperature.models.xgboost_model import (
    XGBoostFeatureSelector,
    XGBoostForecaster,
    XGBoostForecastPipeline,
)
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    CleanGfsApp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

FULL_PANEL_NAME = "gfs_full_panel.parquet"
FULL_REPORT_NAME = "gfs_full_screening_report.csv"

SUMMARY_FILENAME = "xgboost_backtest_summary.csv"
COMPARISON_FILENAME = "xgboost_backtest_comparison.csv"
# Hayden's paired artifacts on this branch; the comparison below is appended
# to them rather than recomputed, so every model stays on identical folds.
COVARIATE_SUMMARY_FILENAME = "covariate_backtest_summary.csv"

MAX_FEATURES = 8


class FullPanelApp(CleanGfsApp):
    """Build the panel from every ingested run, under its own filenames."""

    panel_filename = FULL_PANEL_NAME
    report_filename = FULL_REPORT_NAME


def dedupe_cutoff_aware(
    data: xgboost_frame.XGBoostTrainingData,
) -> xgboost_frame.XGBoostTrainingData:
    """Keep the latest legitimate run per valid hour, in time order.

    Every row handed in already satisfies ``init_time_utc < cutoff`` (see
    ``forecast_covariates.training_pairs``), so "latest per hour" is the
    freshest forecast that existed at the fold's cutoff. Sorting the survivors
    by valid time keeps the temporal-holdout validation honest. Selection is
    positional throughout: the input index is not unique, so label-based
    ``.loc`` would silently reintroduce the dropped duplicates.
    """
    order = np.argsort(data.run_times.to_numpy(), kind="stable")
    positions = np.arange(len(data.features))[order]
    index_labels = data.features.index.to_numpy()[order]
    keep = ~pd.Series(index_labels).duplicated(keep="last").to_numpy()
    selected = np.sort(positions[keep])
    time_order = np.argsort(data.features.index.to_numpy()[selected], kind="stable")
    final = selected[time_order]
    return xgboost_frame.XGBoostTrainingData(
        features=data.features.iloc[final],
        target=data.target.iloc[final],
        run_times=data.run_times.iloc[final],
    )


@dataclass
class XGBoostBacktestModel:
    """The GFS-correcting XGBoost behind the shared backtest protocol."""

    panel: pd.DataFrame
    max_features: int = MAX_FEATURES
    model_params: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_GFS_MODEL_PARAMS)
    )
    cutoff: pd.Timestamp | None = None
    pipeline: XGBoostForecastPipeline | None = None
    best_iteration: int | None = None
    training_rows: int = 0

    def fit(self, history: pd.DataFrame) -> XGBoostBacktestModel:
        """Fit on pre-cutoff pairs; the cutoff is one hour past ``history``."""
        self.cutoff = history[prophet_frame.DS].max() + pd.Timedelta(hours=1)
        target = history.rename(
            columns={
                prophet_frame.DS: schema.TIMESTAMP_UTC,
                prophet_frame.Y: schema.TEMPERATURE_C,
            }
        )
        data = dedupe_cutoff_aware(
            xgboost_frame.build_training_data(self.panel, target, self.cutoff)
        )
        self.training_rows = len(data.features)

        validation_run = data.run_times.max()
        validation_mask = data.run_times.eq(validation_run).to_numpy()
        train_features = data.features.iloc[~validation_mask]
        validation_features = data.features.iloc[validation_mask]
        if train_features.empty or validation_features.empty:
            raise ValueError(
                f"Fold at {self.cutoff}: at least two historical GFS runs are required."
            )
        raw_train = train_features[xgboost_frame.GFS_TEMPERATURE]
        raw_validation = validation_features[xgboost_frame.GFS_TEMPERATURE]
        train_target = data.target.loc[train_features.index] - raw_train
        validation_target = data.target.loc[validation_features.index] - raw_validation

        validation_pipeline = XGBoostForecastPipeline(
            selector=XGBoostFeatureSelector(
                max_features=self.max_features,
                model_params=self.model_params,
            ),
            forecaster=XGBoostForecaster(model_params=self.model_params),
        ).fit(
            train_features,
            train_target,
            validation_features=validation_features,
            validation_target=validation_target,
        )

        fitted = validation_pipeline.forecaster.model_
        self.best_iteration = getattr(fitted, "best_iteration", None)
        final_params = dict(self.model_params)
        if self.best_iteration is not None:
            final_params["n_estimators"] = int(self.best_iteration) + 1

        self.pipeline = XGBoostForecastPipeline(
            selector=XGBoostFeatureSelector(
                max_features=self.max_features,
                model_params=self.model_params,
            ),
            forecaster=XGBoostForecaster(
                model_params=final_params,
                early_stopping_rounds=None,
            ),
        ).fit(
            data.features,
            data.target - data.features[xgboost_frame.GFS_TEMPERATURE],
        )
        return self

    def forecast(
        self, cutoff: pd.Timestamp, hours: int = prophet_frame.FORECAST_HOURS
    ) -> pd.DataFrame:
        """Correct the latest legitimate GFS run over the forecast window."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() before forecast().")
        future = xgboost_frame.build_forecast_data(self.panel, cutoff, hours)
        if len(future.features) != hours:
            raise ValueError(
                f"Forecast window from {cutoff} has {len(future.features)} of "
                f"{hours} hours; the fold is not covered."
            )
        raw = future.features[xgboost_frame.GFS_TEMPERATURE]
        corrected = raw + self.pipeline.predict(future.features)
        frame = prophet_frame.future_frame(cutoff, hours).loc[:, [prophet_frame.DS]]
        frame["yhat"] = corrected.to_numpy()
        frame["yhat_lower"] = np.nan
        frame["yhat_upper"] = np.nan
        return frame


def fold_sets(
    frame: pd.DataFrame, panel: pd.DataFrame, step_days: int
) -> list[FoldSet]:
    """The seasonal and rolling folds a covariate model can be trained for.

    Identical to ``run_covariate_prophet.fold_sets``: only folds whose horizon
    a legitimate run covers *and* whose covariate history spans an annual
    cycle, so XGBoost is scored on exactly the same fortnights.
    """
    cutoff = prophet_frame.forecast_cutoff()
    candidates = {
        "seasonal": seasonal_cutoffs(frame, cutoff),
        "rolling": rolling_cutoffs(frame, step_days=step_days),
    }
    return [
        FoldSet(
            name,
            forecast_covariates.trainable_cutoffs(
                frame,
                panel,
                cutoffs,
                minimum_training_days=MINIMUM_TRAINING_DAYS,
            ),
        )
        for name, cutoffs in candidates.items()
    ]


def _log_fold(result: FoldResult) -> None:
    metrics = result.metrics()
    print(
        f"  fold {result.cutoff.date()}  MAE {metrics['mae_c']:.2f} C  "
        f"RMSE {metrics['rmse_c']:.2f} C  bias {metrics['bias_c']:+.2f} C  "
        f"({metrics['scored_hours']} hours)",
        flush=True,
    )


def run(
    *,
    raw_dir: Path = DEFAULT_INPUT_DIR,
    panel_dir: Path = DEFAULT_OUTPUT_DIR,
    target_path: Path = prophet_frame.DEFAULT_TARGET_PATH,
    output_dir: Path = DEFAULT_METRICS_DIR,
    step_days: int = 14,
    rebuild_panel: bool = False,
    overwrite: bool = False,
    fold_names: Sequence[str] = ("seasonal", "rolling"),
    on_fold: Callable[[FoldResult], None] | None = _log_fold,
) -> dict[str, Path]:
    """Score XGBoost on the shared trainable folds and pair it with Prophet."""
    outputs = {
        "summary": output_dir / SUMMARY_FILENAME,
        "comparison": output_dir / COMPARISON_FILENAME,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --overwrite to replace it."
        )

    panel_path = panel_dir / FULL_PANEL_NAME
    try:
        FullPanelApp(raw_dir, panel_dir).run(overwrite=rebuild_panel)
    except FileExistsError:
        print(
            f"Reusing existing panel at {panel_path}; pass --rebuild-panel to "
            "rebuild it after a fetch.",
            flush=True,
        )

    panel = forecast_covariates.load_panel(panel_path)
    frame = prophet_frame.load(target_path)

    summaries: list[pd.DataFrame] = []
    for folds in fold_sets(frame, panel, step_days):
        if folds.name not in fold_names:
            continue
        if not folds.cutoffs:
            print(
                f"\n{folds.name}: no fold has a year of covariate history; skipped.",
                flush=True,
            )
            continue
        print(
            f"\n{folds.name}: {len(folds.cutoffs)} trainable fold(s), "
            f"{folds.cutoffs[0].date()} to {folds.cutoffs[-1].date()}.",
            flush=True,
        )
        results = list(
            Backtest(lambda: XGBoostBacktestModel(panel=panel)).run(
                frame, folds.cutoffs, on_fold=on_fold
            )
        )
        summaries.append(summarize(results).assign(model="xgboost", folds=folds.name))

    if not summaries:
        raise ValueError("No trainable fold found; check the panel and target.")
    summary = pd.concat(summaries, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs["summary"], index=False)

    comparison: pd.DataFrame | None = None
    covariate_summary_path = output_dir / COVARIATE_SUMMARY_FILENAME
    if covariate_summary_path.exists():
        covariate_summary = pd.read_csv(covariate_summary_path)
        combined = pd.concat([covariate_summary, summary], ignore_index=True)
        combined["cutoff"] = combined["cutoff"].astype(str)
        comparison = compare.compare_all(combined, "raw_gfs")
        comparison.to_csv(outputs["comparison"], index=False)
        for folds, group in comparison.groupby("folds", sort=True):
            print(f"\n{folds}: paired MAE against raw_gfs", flush=True)
            for _, row in group.iterrows():
                _report_difference(row)
        for reference in ("prophet_gfs", "raw_gfs"):
            for folds in sorted(summary["folds"].unique()):
                try:
                    pairing = compare.compare(combined, "xgboost", reference, folds)
                except ValueError as exc:
                    print(f"\nxgboost vs {reference} on {folds}: {exc}", flush=True)
                    continue
                print(f"\nxgboost vs {reference} on {folds} folds:", flush=True)
                _report_difference(pairing.as_row())

    for name, path in outputs.items():
        if name == "comparison" and comparison is None:
            continue
        print(f"Wrote {name}: {path}", flush=True)
    return outputs


def _report_difference(row: pd.Series) -> None:
    direction = "better" if row["mean_difference"] < 0 else "worse"
    verdict = (
        "real difference" if row["significant"] else "indistinguishable from reference"
    )
    print(
        f"  {row['challenger']:<12s} {row['mean_difference']:+6.3f} C "
        f"({direction})  95% CI [{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]  "
        f"wins {row['challenger_wins']:>2d}/{row['n_folds']:<3d}  {verdict}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target", type=Path, default=prophet_frame.DEFAULT_TARGET_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument(
        "--fold",
        action="append",
        choices=("seasonal", "rolling"),
        dest="folds",
        help="Restrict the run to one fold set; repeat for several.",
    )
    parser.add_argument(
        "--rebuild-panel",
        action="store_true",
        help="Rebuild the panel even if it already exists.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(
        raw_dir=args.raw_dir,
        panel_dir=args.panel_dir,
        target_path=args.target,
        output_dir=args.output_dir,
        step_days=args.step_days,
        rebuild_panel=args.rebuild_panel,
        overwrite=args.overwrite,
        fold_names=tuple(args.folds or ("seasonal", "rolling")),
    )


if __name__ == "__main__":
    main()
