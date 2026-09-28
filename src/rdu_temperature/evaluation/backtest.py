"""Score the Prophet model on past Septembers without touching the forecast.

Run from the repository root with:

    python -m rdu_temperature.evaluation.backtest

A generic rolling-origin split would report how the model does on an average
fortnight. That is not the task. The project forecasts 336 hours beginning on
September 17, so every fold here reproduces exactly that: a cutoff on the same
date in an earlier year, the same horizon, and training restricted to the hours
before it. What comes out is the model's record on four reruns of the problem
it will actually be asked to solve, rather than on a fortnight in March.

Folds are held strictly before the real cutoff, so no fold can see an hour from
the forecast period. The earliest fold trains on a single annual cycle and the
latest on four, which is close to the five the final fit will have; reading the
folds in order shows what the extra years are worth.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdu_temperature.features import prophet_frame
from rdu_temperature.models.prophet_model import (
    ProphetConfig,
    TemperatureProphet,
    split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "metrics"

# One full annual cycle before Prophet has any yearly seasonality to estimate.
MINIMUM_TRAINING_DAYS = 365

HOURS_PER_DAY = 24


def seasonal_cutoffs(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_hours: int = prophet_frame.FORECAST_HOURS,
    minimum_training_days: int = MINIMUM_TRAINING_DAYS,
) -> list[pd.Timestamp]:
    """Return anniversaries of ``cutoff`` the history can support, oldest first.

    A fold qualifies when its whole horizon falls inside the observed grid and
    at least one annual cycle precedes it. Stepping by calendar years rather
    than by a fixed hour count keeps every fold on September 17, which is the
    point of matching the season.
    """
    first = frame[prophet_frame.DS].min()
    frame_end = frame[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    horizon = pd.Timedelta(hours=horizon_hours)
    minimum_training = pd.Timedelta(days=minimum_training_days)

    folds: list[pd.Timestamp] = []
    candidate = cutoff - pd.DateOffset(years=1)
    while candidate - first >= minimum_training:
        if candidate + horizon <= frame_end:
            folds.append(pd.Timestamp(candidate))
        candidate -= pd.DateOffset(years=1)
    return sorted(folds)


@dataclass(frozen=True)
class FoldResult:
    """One fold's predictions joined to the truth it was scored against."""

    cutoff: pd.Timestamp
    training_hours: int
    predictions: pd.DataFrame

    @property
    def scored(self) -> pd.DataFrame:
        """Rows where an observation exists to compare against.

        The uncovered hours are never filled, so a fold whose horizon overlaps
        one simply scores the hours it can and reports how many that was.
        """
        return self.predictions.loc[self.predictions[prophet_frame.Y].notna()]

    def metrics(self) -> dict[str, Any]:
        scored = self.scored
        error = scored["yhat"] - scored[prophet_frame.Y]
        inside = scored[prophet_frame.Y].between(
            scored["yhat_lower"], scored["yhat_upper"]
        )
        return {
            "cutoff": self.cutoff,
            "training_hours": self.training_hours,
            "training_years": round(self.training_hours / (365.25 * HOURS_PER_DAY), 2),
            "scored_hours": len(scored),
            "mae_c": float(error.abs().mean()),
            "rmse_c": float(np.sqrt((error**2).mean())),
            "bias_c": float(error.mean()),
            "max_abs_error_c": float(error.abs().max()),
            "interval_coverage": float(inside.mean()),
        }

    def by_lead_day(self) -> pd.DataFrame:
        """Mean absolute error per day of lead time, to show horizon decay."""
        scored = self.scored.copy()
        elapsed = scored[prophet_frame.DS] - self.cutoff
        scored["lead_day"] = (
            elapsed.dt.total_seconds() // (HOURS_PER_DAY * 3600)
        ).astype(int) + 1
        error = (scored["yhat"] - scored[prophet_frame.Y]).abs()
        return (
            scored.assign(absolute_error_c=error)
            .groupby("lead_day", as_index=False)
            .agg(mae_c=("absolute_error_c", "mean"), scored_hours=("lead_day", "size"))
            .assign(cutoff=self.cutoff)
        )


@dataclass(frozen=True)
class Backtest:
    """Refit the model at each cutoff and score the horizon that follows."""

    config: ProphetConfig = field(default_factory=ProphetConfig)
    horizon_hours: int = prophet_frame.FORECAST_HOURS

    def run_fold(self, frame: pd.DataFrame, cutoff: pd.Timestamp) -> FoldResult:
        history, held_out = split(frame, cutoff)
        model = TemperatureProphet(self.config).fit(history)
        predictions = model.forecast(cutoff, self.horizon_hours)
        truth = held_out.loc[:, [prophet_frame.DS, prophet_frame.Y]]
        return FoldResult(
            cutoff=cutoff,
            training_hours=int(history[prophet_frame.Y].notna().sum()),
            predictions=predictions.merge(truth, on=prophet_frame.DS, how="left"),
        )

    def run(
        self,
        frame: pd.DataFrame,
        cutoffs: Sequence[pd.Timestamp],
        on_fold: Callable[[FoldResult], None] | None = None,
    ) -> Iterator[FoldResult]:
        for cutoff in cutoffs:
            result = self.run_fold(frame, cutoff)
            if on_fold is not None:
                on_fold(result)
            yield result


def summarize(results: Sequence[FoldResult]) -> pd.DataFrame:
    """Collect every fold's metrics, with a weighted total row appended.

    The total weights each fold by the hours it actually scored, so a fold
    short a few uncovered hours does not count the same as a full one.
    """
    per_fold = pd.DataFrame([result.metrics() for result in results])
    weights = per_fold["scored_hours"]
    total = {
        "cutoff": "all",
        "training_hours": int(per_fold["training_hours"].sum()),
        "training_years": None,
        "scored_hours": int(weights.sum()),
        "max_abs_error_c": float(per_fold["max_abs_error_c"].max()),
    }
    for column in ("mae_c", "rmse_c", "bias_c", "interval_coverage"):
        total[column] = float(np.average(per_fold[column], weights=weights))
    return pd.concat([per_fold, pd.DataFrame([total])], ignore_index=True)


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


class BacktestApp:
    """Load the frame, run every seasonal fold, and write the metrics."""

    summary_filename = "prophet_backtest_summary.csv"
    lead_filename = "prophet_backtest_by_lead_day.csv"
    predictions_filename = "prophet_backtest_predictions.parquet"

    def __init__(
        self,
        target_path: Path,
        output_dir: Path,
        config: ProphetConfig,
        horizon_hours: int = prophet_frame.FORECAST_HOURS,
    ) -> None:
        self.target_path = target_path
        self.output_dir = output_dir
        self.config = config
        self.horizon_hours = horizon_hours

    def run(self, *, overwrite: bool = False) -> dict[str, Path]:
        outputs = {
            "summary": self.output_dir / self.summary_filename,
            "by_lead_day": self.output_dir / self.lead_filename,
            "predictions": self.output_dir / self.predictions_filename,
        }
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{existing[0]} already exists; pass --overwrite to replace it."
            )

        frame = prophet_frame.load(self.target_path)
        cutoff = prophet_frame.forecast_cutoff()
        cutoffs = seasonal_cutoffs(frame, cutoff, self.horizon_hours)
        if not cutoffs:
            raise ValueError(
                "No backtest fold fits inside the history; check the target grid."
            )
        print(
            f"Forecast cutoff {cutoff}; {len(cutoffs)} seasonal fold(s) "
            f"over a {self.horizon_hours}-hour horizon.",
            flush=True,
        )

        backtest = Backtest(self.config, self.horizon_hours)
        results = list(backtest.run(frame, cutoffs, on_fold=_report))

        summary = summarize(results)
        by_lead_day = pd.concat(
            [result.by_lead_day() for result in results], ignore_index=True
        )
        predictions = pd.concat(
            [result.predictions.assign(cutoff=result.cutoff) for result in results],
            ignore_index=True,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write(lambda path: summary.to_csv(path, index=False), outputs["summary"])
        _write(
            lambda path: by_lead_day.to_csv(path, index=False), outputs["by_lead_day"]
        )
        _write(predictions.to_parquet, outputs["predictions"])
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}", flush=True)
        return outputs


def _report(result: FoldResult) -> None:
    metrics = result.metrics()
    print(
        f"  {metrics['cutoff']:%Y-%m-%d}  "
        f"train {metrics['training_years']:>4} yr  "
        f"MAE {metrics['mae_c']:5.2f} C  "
        f"RMSE {metrics['rmse_c']:5.2f} C  "
        f"bias {metrics['bias_c']:+5.2f} C  "
        f"coverage {metrics['interval_coverage']:.0%}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=prophet_frame.DEFAULT_TARGET_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.01)
    parser.add_argument("--yearly-order", type=int, default=6)
    parser.add_argument("--daily-order", type=int, default=6)
    parser.add_argument(
        "--conditional-daily",
        action="store_true",
        help="Fit one daily seasonality per meteorological season instead of "
        "one for the whole year.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = ProphetConfig(
        yearly_fourier_order=args.yearly_order,
        daily_fourier_order=args.daily_order,
        changepoint_prior_scale=args.changepoint_prior_scale,
        conditional_daily=args.conditional_daily,
    )
    BacktestApp(args.target, args.output_dir, config).run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
