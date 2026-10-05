"""Train the GFS-correcting XGBoost on the full rolling-run archive.

Run from the repository root with:

    python -m rdu_temperature.models.run_xgboost_rolling

This is the rolling-data counterpart to ``run_xgboost``. That entry point
trained the same correction pipeline on the September-seasonal runs only
(``gfs_forecast_panel.parquet``). This one trains on every ingested GFS run:
the 26-run rolling archive (one run every 56 days from 2022-09-16 through
2026-07-17) plus the extra September runs, to test whether full-year history
beats September-only history.

It never touches the September experiment:

- the panel is rebuilt into ``data/processed/gfs_rolling_panel.parquet``,
  leaving ``gfs_forecast_panel.parquet`` alone;
- the model, metrics, and forecast go to ``xgboost_rolling``-named outputs,
  leaving the ``xgboost_gfs`` artifacts alone.

Overlapping runs are collapsed to one forecast per valid hour (see
``_dedupe_panel``): the 2024-09-13 rolling run and the 2024-09-16 September
run cover the same valid hours, and the training pipeline requires unique
valid-time indices.

It performs no git operations: outputs are local files only, nothing is
committed or pushed. Validation still holds out the latest historical run
(forward validation), then refits on all pairs before forecasting the
336-hour window from the operational forecast run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rdu_temperature.models.run_xgboost import DEFAULT_TARGET_PATH
from rdu_temperature.models.run_xgboost import run as train_xgboost
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    CleanGfsApp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROLLING_PANEL_NAME = "gfs_rolling_panel.parquet"
ROLLING_REPORT_NAME = "gfs_rolling_screening_report.csv"

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "xgboost_rolling"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_rolling.json"
DEFAULT_FORECAST_PATH = (
    PROJECT_ROOT / "artifacts" / "predictions" / "xgboost_rolling_forecast.csv"
)
SEPTEMBER_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_gfs.json"


class RollingPanelApp(CleanGfsApp):
    """Build the panel under rolling-specific filenames.

    ``load_runs`` already globs every ``gfs_*.csv`` in the raw directory, so
    no ingestion change is needed: rebuilding the panel picks up the rolling
    runs automatically.
    """

    panel_filename = ROLLING_PANEL_NAME
    report_filename = ROLLING_REPORT_NAME


def _dedupe_panel(panel_path: Path) -> None:
    """Collapse the panel to one forecast row per valid hour.

    The 26-run rolling archive and the extra September runs overlap on
    2024-09-16 -> 2024-09-27: the 2024-09-13 rolling run and the 2024-09-16
    September run cover the same valid hours. The training pipeline indexes
    rows by valid time and requires unique indices, so keep the latest
    initialization per valid hour -- the forecast an operational run would
    actually use.
    """
    panel = pd.read_parquet(panel_path)
    before = len(panel)
    deduped = panel.sort_values(schema.INIT_TIME_UTC).drop_duplicates(
        subset=schema.VALID_TIME_UTC, keep="last"
    )
    removed = before - len(deduped)
    if removed == 0:
        return
    deduped.to_parquet(panel_path)
    print(
        f"Deduped rolling panel to one forecast per valid hour "
        f"({removed:,} overlapping row(s) removed).",
        flush=True,
    )


def _comparison(new_report: dict) -> None:
    """Print rolling metrics beside the September-only experiment, if present."""
    if not SEPTEMBER_METRICS_PATH.exists():
        return
    old = json.loads(SEPTEMBER_METRICS_PATH.read_text(encoding="utf-8"))
    print("\nComparison against the September-only experiment:")
    print(
        f"{'':<22}{'september':>14}{'rolling':>14}",
        f"\n{'training_rows':<22}{old['training_rows']:>14}{new_report['training_rows']:>14}",
        f"\n{'historical_runs':<22}{old['historical_runs']:>14}{new_report['historical_runs']:>14}",
        f"\n{'validation_run':<22}{old['validation_run']!s:>14}{new_report['validation_run']!s:>14}",
        flush=True,
    )
    for metric in ("mae_c", "rmse_c", "bias_c"):
        old_xgb = old["xgboost_validation"][metric]
        new_xgb = new_report["xgboost_validation"][metric]
        old_raw = old["raw_gfs_validation"][metric]
        new_raw = new_report["raw_gfs_validation"][metric]
        print(
            f"{'xgb_' + metric:<22}{old_xgb:>14.4f}{new_xgb:>14.4f}"
            f"\n{'raw_gfs_' + metric:<22}{old_raw:>14.4f}{new_raw:>14.4f}",
            flush=True,
        )
    print(
        "\nNote: the two experiments validate on different holdout runs "
        "(latest run each), so the metrics are indicative, not a paired test.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-panel",
        action="store_true",
        help="Rebuild the rolling panel even if it already exists.",
    )
    parser.add_argument("--max-features", type=int, default=8)
    args = parser.parse_args()

    panel_path = DEFAULT_OUTPUT_DIR / ROLLING_PANEL_NAME
    app = RollingPanelApp(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)
    try:
        app.run(overwrite=args.rebuild_panel)
    except FileExistsError:
        print(
            f"Reusing existing rolling panel at {panel_path}; "
            "pass --rebuild-panel to rebuild it.",
            flush=True,
        )
    _dedupe_panel(panel_path)

    paths = train_xgboost(
        panel_path=panel_path,
        target_path=DEFAULT_TARGET_PATH,
        model_dir=DEFAULT_MODEL_DIR,
        metrics_path=DEFAULT_METRICS_PATH,
        forecast_path=DEFAULT_FORECAST_PATH,
        max_features=args.max_features,
    )
    report = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    _comparison(report)
    print(f"\nWrote model: {paths['model']}", flush=True)


if __name__ == "__main__":
    main()
