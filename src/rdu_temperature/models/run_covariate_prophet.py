"""Backtest Prophet-with-GFS against the baselines it has to beat.

Run from the repository root with:

    python -m rdu_temperature.models.run_covariate_prophet

The univariate backtest cannot host this model. Its fold sets are built from
the observed grid, which is continuous, while a covariate model can only be
scored where a GFS run reaches -- four of the four seasonal folds and 26 of
the 104 rolling ones -- and can only be *trained* on the subset of those with
a year of covariate history behind them. Running the two in one pass would
either crash on the folds the covariate model cannot reach or quietly average
each model over a different set of fortnights, which is not a comparison.

So this is a separate entry point writing separate artifacts, the same way
``run_xgboost`` keeps clear of the Prophet experiment. It never touches the
univariate results:

- the panel is built from every ingested run into
  ``data/processed/gfs_full_panel.parquet``, leaving the September-only
  ``gfs_forecast_panel.parquet`` and the XGBoost ``gfs_rolling_panel.parquet``
  alone;
- metrics go to ``covariate_backtest_summary.csv`` and
  ``covariate_comparison.csv``, leaving ``backtest_summary.csv`` alone.

Every model runs on the identical fold set, baselines included, so the
baselines here will not match their numbers in ``backtest_summary.csv`` -- the
folds are a subset and a differently-weighted one. That is the price of an
exact pairing, and the pairing is what makes the differences readable.

The panel is deliberately **not** deduplicated to one row per valid hour, as
``run_xgboost_rolling`` does for its own. That dedupe keeps the latest
initialisation per hour regardless of any cutoff, which is right for a single
training matrix and wrong here: a fold has to see the freshest run that
existed *before its own cutoff*, and dropping the earlier overlapping runs
would leave some folds with no legitimate covariate at all.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from rdu_temperature.evaluation import compare
from rdu_temperature.evaluation.backtest import (
    MINIMUM_TRAINING_DAYS,
    Backtest,
    FoldSet,
    ModelFactory,
    rolling_cutoffs,
    seasonal_cutoffs,
    summarize,
)
from rdu_temperature.features import forecast_covariates, prophet_frame
from rdu_temperature.models.baselines import Climatology, SeasonalPersistence
from rdu_temperature.models.covariate_prophet import (
    ALL_REGRESSORS,
    CovariateProphet,
    RawGfs,
)
from rdu_temperature.models.prophet_model import ProphetConfig, TemperatureProphet
from rdu_temperature.pipeline.clean_gfs import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    CleanGfsApp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

FULL_PANEL_NAME = "gfs_full_panel.parquet"
FULL_REPORT_NAME = "gfs_full_screening_report.csv"

SUMMARY_FILENAME = "covariate_backtest_summary.csv"
COMPARISON_FILENAME = "covariate_comparison.csv"

# The uncorrected forecast, because that is the honest reference. Beating
# climatology only says the model knows what September looks like; beating raw
# GFS says the correction earns its place.
REFERENCE_MODEL = "raw_gfs"

# Below this many folds, the bootstrap is not reporting an interval so much as
# reporting the folds back. Resampling two numbers can only ever produce three
# distinct means, so the interval comes out tight on a sample of two and reads
# as a confident finding; the covariate fold sets are small enough that this is
# a live trap rather than a theoretical one. The numbers are still written out,
# because they are the per-fold truth, but the verdict is withheld.
MINIMUM_RESOLVING_FOLDS = 8


class FullPanelApp(CleanGfsApp):
    """Build the panel from every ingested run, under its own filenames.

    ``load_runs`` already globs every ``gfs_*.csv``, so no ingestion change is
    needed: a fetch that adds runs is picked up by rebuilding.
    """

    panel_filename = FULL_PANEL_NAME
    report_filename = FULL_REPORT_NAME


def build_models(panel: pd.DataFrame) -> Mapping[str, ModelFactory]:
    """The models to score, ordered worst-expected first for a readable log."""
    return {
        "persistence": SeasonalPersistence,
        "climatology": Climatology,
        "prophet": lambda: TemperatureProphet(ProphetConfig()),
        "raw_gfs": lambda: RawGfs(panel=panel),
        # The three covariate variants: the default pure correction, the same
        # with every covariate rather than temperature alone, and the one that
        # keeps Prophet's own trend and seasonality.
        "prophet_gfs": lambda: CovariateProphet(panel=panel),
        "prophet_gfs_all": lambda: CovariateProphet(
            panel=panel, regressors=ALL_REGRESSORS
        ),
        "prophet_gfs_seasonal": lambda: CovariateProphet(panel=panel, seasonality=True),
    }


def fold_sets(
    frame: pd.DataFrame, panel: pd.DataFrame, step_days: int
) -> list[FoldSet]:
    """The seasonal and rolling folds a covariate model can be trained for."""
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
                # The same annual cycle the univariate models must clear, so a
                # fold is not held to a different standard here than there.
                minimum_training_days=MINIMUM_TRAINING_DAYS,
            ),
        )
        for name, cutoffs in candidates.items()
    ]


def run(
    *,
    raw_dir: Path = DEFAULT_INPUT_DIR,
    panel_dir: Path = DEFAULT_OUTPUT_DIR,
    target_path: Path = prophet_frame.DEFAULT_TARGET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_METRICS_DIR,
    step_days: int = 14,
    rebuild_panel: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Score every model on the trainable folds and test the differences."""
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
    models = build_models(panel)

    summaries: list[pd.DataFrame] = []
    for folds in fold_sets(frame, panel, step_days):
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
        for name, factory in models.items():
            results = list(Backtest(factory).run(frame, folds.cutoffs))
            summary = summarize(results).assign(model=name, folds=folds.name)
            summaries.append(summary)
            _report(name, summary.iloc[-1])

    if not summaries:
        raise ValueError(
            "No fold has both a covered horizon and a year of covariate "
            "history; fetch runs spanning more of the archive."
        )

    summary = pd.concat(summaries, ignore_index=True)
    # ``resolving`` travels with the rows so the file carries the caveat too. A
    # reader filtering on ``significant`` alone would take a two-fold interval
    # at face value, and on these fold sets that is the likelier mistake than
    # missing a real effect.
    comparison = compare.compare_all(summary, REFERENCE_MODEL).assign(
        resolving=lambda frame: frame["n_folds"] >= MINIMUM_RESOLVING_FOLDS
    )
    for folds, group in comparison.groupby("folds", sort=True):
        print(f"\n{folds}: paired MAE against {REFERENCE_MODEL}", flush=True)
        for _, row in group.iterrows():
            _report_difference(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(lambda path: summary.to_csv(path, index=False), outputs["summary"])
    _write(lambda path: comparison.to_csv(path, index=False), outputs["comparison"])
    for name, path in outputs.items():
        print(f"\nWrote {name}: {path}", flush=True)
    return outputs


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


def _report(model: str, total: pd.Series) -> None:
    coverage = total["interval_coverage"]
    coverage_text = "     —" if pd.isna(coverage) else f"{coverage:6.0%}"
    print(
        f"  {model:<22s} MAE {total['mae_c']:5.2f} C  "
        f"RMSE {total['rmse_c']:5.2f} C  "
        f"bias {total['bias_c']:+5.2f} C  "
        f"coverage {coverage_text}  "
        f"({int(total['scored_hours']):,} hours)",
        flush=True,
    )


def _report_difference(row: pd.Series) -> None:
    """One paired comparison, with whether the folds can resolve it."""
    if row["n_folds"] < MINIMUM_RESOLVING_FOLDS:
        verdict = f"too few folds to resolve ({int(row['n_folds'])})"
    elif row["significant"]:
        verdict = "real difference"
    else:
        verdict = "indistinguishable from the reference"
    direction = "better" if row["mean_difference"] < 0 else "worse"
    print(
        f"  {row['challenger']:<22s} {row['mean_difference']:+6.3f} C "
        f"({direction})  95% CI [{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]  "
        f"wins {row['challenger_wins']:>2d}/{row['n_folds']:<3d} {verdict}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target", type=Path, default=prophet_frame.DEFAULT_TARGET_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_METRICS_DIR)
    parser.add_argument("--step-days", type=int, default=14)
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
    )


if __name__ == "__main__":
    main()
