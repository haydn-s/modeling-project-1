"""Score every model on past hours without touching the forecast period.

Run from the repository root with:

    python -m rdu_temperature.evaluation.backtest

Two fold sets, because they answer different questions and neither answers
both.

The **seasonal** folds reproduce the task. Each takes a cutoff on September 17
of an earlier year, forecasts the same 336 hours, and trains only on the hours
before it. These are the headline numbers: the model's record on reruns of the
problem it will actually be asked to solve, rather than on a fortnight in
March. There are only four of them, which is the catch.

The **rolling** folds exist to make small differences measurable. Four folds
cannot resolve anything: the spread between them is larger than most effects
worth testing, so a change of a tenth of a degree is indistinguishable from
which Septembers happened to land in the sample. Stepping a cutoff through the
whole history gives a hundred or so folds and the power to tell a real
improvement from noise. They are not the task — they average over seasons the
project will never forecast — so they inform decisions rather than report
results.

Folds in both sets are held strictly before the real cutoff, so none can see an
hour from the forecast period.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from rdu_temperature.features import prophet_frame
from rdu_temperature.models.baselines import Climatology, SeasonalPersistence
from rdu_temperature.models.prophet_model import ProphetConfig, TemperatureProphet

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "metrics"

# One full annual cycle before a model has any yearly seasonality to estimate.
MINIMUM_TRAINING_DAYS = 365

# How far the rolling cutoff advances between folds. Fourteen days keeps
# consecutive horizons from overlapping, so no observed hour is scored twice
# within a fold set and the folds stay close to independent.
ROLLING_STEP_DAYS = 14

HOURS_PER_DAY = 24


class ForecastModel(Protocol):
    """What the backtest needs of a model: fit on history, predict a horizon."""

    def fit(self, history: pd.DataFrame) -> Any: ...

    def forecast(self, cutoff: pd.Timestamp, hours: int) -> pd.DataFrame: ...


ModelFactory = Callable[[], ForecastModel]

MODELS: Mapping[str, ModelFactory] = {
    "prophet": lambda: TemperatureProphet(ProphetConfig()),
    "climatology": Climatology,
    "persistence": SeasonalPersistence,
}


def split(
    frame: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide the frame at a cutoff into history and held-out truth.

    The split is strictly before and on-or-after the cutoff, matching the
    half-open ingestion window, so no hour can land in both halves.
    """
    before = frame[prophet_frame.DS] < cutoff
    return frame.loc[before].copy(), frame.loc[~before].copy()


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


def rolling_cutoffs(
    frame: pd.DataFrame,
    horizon_hours: int = prophet_frame.FORECAST_HOURS,
    step_days: int = ROLLING_STEP_DAYS,
    minimum_training_days: int = MINIMUM_TRAINING_DAYS,
) -> list[pd.Timestamp]:
    """Step a cutoff through the whole history, oldest first.

    The first cutoff sits one annual cycle after the data begins and the last
    one full horizon before it ends, so every fold trains on a complete year
    and scores a complete horizon.
    """
    first = frame[prophet_frame.DS].min()
    frame_end = frame[prophet_frame.DS].max() + pd.Timedelta(hours=1)
    horizon = pd.Timedelta(hours=horizon_hours)
    step = pd.Timedelta(days=step_days)

    folds: list[pd.Timestamp] = []
    candidate = first + pd.Timedelta(days=minimum_training_days)
    while candidate + horizon <= frame_end:
        folds.append(pd.Timestamp(candidate))
        candidate += step
    return folds


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
        return {
            "cutoff": self.cutoff,
            "training_hours": self.training_hours,
            "training_years": round(self.training_hours / (365.25 * HOURS_PER_DAY), 2),
            "scored_hours": len(scored),
            "mae_c": float(error.abs().mean()),
            "rmse_c": float(np.sqrt((error**2).mean())),
            "bias_c": float(error.mean()),
            "max_abs_error_c": float(error.abs().max()),
            "interval_coverage": _coverage(scored),
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


def _coverage(scored: pd.DataFrame) -> float:
    """Share of observations inside the interval, or missing if there is none.

    A baseline that estimated no spread reports no coverage. Treating its
    absent bounds as a failed interval would score it as nought per cent and
    read as though it had predicted badly rather than not at all.
    """
    bounds = scored.loc[:, ["yhat_lower", "yhat_upper"]]
    if bounds.isna().all(axis=None):
        return float("nan")
    inside = scored[prophet_frame.Y].between(scored["yhat_lower"], scored["yhat_upper"])
    return float(inside.mean())


@dataclass(frozen=True)
class Backtest:
    """Refit a model at each cutoff and score the horizon that follows."""

    model_factory: ModelFactory
    horizon_hours: int = prophet_frame.FORECAST_HOURS

    def run_fold(self, frame: pd.DataFrame, cutoff: pd.Timestamp) -> FoldResult:
        history, held_out = split(frame, cutoff)
        model = self.model_factory()
        model.fit(history)
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
        total[column] = _weighted_mean(per_fold[column], weights)
    return pd.concat([per_fold, pd.DataFrame([total])], ignore_index=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean over the folds that reported the metric at all."""
    present = values.notna()
    if not present.any():
        return float("nan")
    return float(np.average(values[present], weights=weights[present]))


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


@dataclass(frozen=True)
class FoldSet:
    """A named set of cutoffs and what it is for."""

    name: str
    cutoffs: list[pd.Timestamp]


class BacktestApp:
    """Run every model over every fold set and write one set of metrics."""

    summary_filename = "backtest_summary.csv"
    lead_filename = "backtest_by_lead_day.csv"
    predictions_filename = "backtest_predictions.parquet"

    def __init__(
        self,
        target_path: Path,
        output_dir: Path,
        models: Mapping[str, ModelFactory],
        horizon_hours: int = prophet_frame.FORECAST_HOURS,
        step_days: int = ROLLING_STEP_DAYS,
    ) -> None:
        self.target_path = target_path
        self.output_dir = output_dir
        self.models = models
        self.horizon_hours = horizon_hours
        self.step_days = step_days

    def fold_sets(self, frame: pd.DataFrame) -> list[FoldSet]:
        cutoff = prophet_frame.forecast_cutoff()
        return [
            FoldSet("seasonal", seasonal_cutoffs(frame, cutoff, self.horizon_hours)),
            FoldSet(
                "rolling",
                rolling_cutoffs(frame, self.horizon_hours, self.step_days),
            ),
        ]

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
        fold_sets = self.fold_sets(frame)
        if not any(folds.cutoffs for folds in fold_sets):
            raise ValueError(
                "No backtest fold fits inside the history; check the target grid."
            )

        summaries: list[pd.DataFrame] = []
        leads: list[pd.DataFrame] = []
        predictions: list[pd.DataFrame] = []

        for folds in fold_sets:
            print(
                f"\n{folds.name}: {len(folds.cutoffs)} fold(s) over a "
                f"{self.horizon_hours}-hour horizon.",
                flush=True,
            )
            for name, factory in self.models.items():
                results = list(
                    Backtest(factory, self.horizon_hours).run(frame, folds.cutoffs)
                )
                summary = summarize(results).assign(model=name, folds=folds.name)
                summaries.append(summary)
                leads.append(
                    pd.concat(
                        [result.by_lead_day() for result in results], ignore_index=True
                    ).assign(model=name, folds=folds.name)
                )
                predictions.append(
                    pd.concat(
                        [
                            result.predictions.assign(cutoff=result.cutoff)
                            for result in results
                        ],
                        ignore_index=True,
                    ).assign(model=name, folds=folds.name)
                )
                _report(name, summary.iloc[-1])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write(
            lambda path: pd.concat(summaries, ignore_index=True).to_csv(
                path, index=False
            ),
            outputs["summary"],
        )
        _write(
            lambda path: pd.concat(leads, ignore_index=True).to_csv(path, index=False),
            outputs["by_lead_day"],
        )
        _write(
            pd.concat(predictions, ignore_index=True).to_parquet, outputs["predictions"]
        )
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}", flush=True)
        return outputs


def _report(model: str, total: pd.Series) -> None:
    coverage = total["interval_coverage"]
    coverage_text = "     —" if pd.isna(coverage) else f"{coverage:6.0%}"
    print(
        f"  {model:<12s} MAE {total['mae_c']:5.2f} C  "
        f"RMSE {total['rmse_c']:5.2f} C  "
        f"bias {total['bias_c']:+5.2f} C  "
        f"coverage {coverage_text}  "
        f"({int(total['scored_hours']):,} hours)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=prophet_frame.DEFAULT_TARGET_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODELS),
        help="Restrict the run to one model; repeat for several.",
    )
    parser.add_argument("--step-days", type=int, default=ROLLING_STEP_DAYS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chosen = args.model or list(MODELS)
    models = {name: MODELS[name] for name in chosen}
    BacktestApp(args.target, args.output_dir, models, step_days=args.step_days).run(
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
