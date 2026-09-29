"""Test whether a difference between two models is real or is fold noise.

Run from the repository root with:

    python -m rdu_temperature.evaluation.compare

Comparing headline averages is not enough. Folds vary enormously — the spread
between September backtests is larger than most differences worth testing — so
a model can look half a degree better purely because of which fortnights landed
in the sample. The comparison has to be paired: every model sees the same
cutoffs, so the difference is taken fold by fold and the shared difficulty of
each fortnight cancels out.

Significance is read off a bootstrap over folds rather than a t-test. Per-fold
errors are not normal and a handful of folds are far worse than the rest, which
is exactly the case where resampling behaves and a t statistic does not. A
confidence interval that contains zero means the folds cannot tell the two
models apart, which is a finding and not a failure to find one.

This is what the rolling fold set is for. Four seasonal folds cannot resolve a
tenth of a degree; a hundred can, and knowing which differences are real is
what keeps tuning from chasing noise.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "backtest_summary.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "model_comparison.csv"

TOTAL_ROW = "all"
DEFAULT_RESAMPLES = 20_000
DEFAULT_SEED = 20260917
CONFIDENCE = 95.0


@dataclass(frozen=True)
class PairedComparison:
    """One model measured against another over the folds they share."""

    folds: str
    metric: str
    challenger: str
    reference: str
    n_folds: int
    mean_difference: float
    ci_low: float
    ci_high: float
    challenger_wins: int

    @property
    def significant(self) -> bool:
        """True when the interval sits wholly on one side of zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "metric": self.metric,
            "challenger": self.challenger,
            "reference": self.reference,
            "n_folds": self.n_folds,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "challenger_wins": self.challenger_wins,
            "win_rate": self.challenger_wins / self.n_folds,
            "significant": self.significant,
        }


def per_fold(summary: pd.DataFrame) -> pd.DataFrame:
    """Drop the weighted total rows, keeping one row per fold."""
    return summary.loc[summary["cutoff"] != TOTAL_ROW]


def compare(
    summary: pd.DataFrame,
    challenger: str,
    reference: str,
    folds: str,
    metric: str = "mae_c",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Bootstrap the paired per-fold difference between two models.

    A negative mean difference means the challenger beat the reference, since
    every metric here is an error that is better when smaller.
    """
    scoped = per_fold(summary)
    scoped = scoped.loc[scoped["folds"] == folds]
    wide = scoped.pivot(index="cutoff", columns="model", values=metric)
    missing = {challenger, reference} - set(wide.columns)
    if missing:
        raise ValueError(f"No {metric} for {sorted(missing)} on the {folds!r} folds.")

    paired = wide.loc[:, [challenger, reference]].dropna()
    if paired.empty:
        raise ValueError(f"No shared {folds!r} folds for {challenger} and {reference}.")

    difference = (paired[challenger] - paired[reference]).to_numpy()
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        difference, size=(resamples, difference.size), replace=True
    ).mean(axis=1)
    tail = (100.0 - CONFIDENCE) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])

    return PairedComparison(
        folds=folds,
        metric=metric,
        challenger=challenger,
        reference=reference,
        n_folds=int(difference.size),
        mean_difference=float(difference.mean()),
        ci_low=float(low),
        ci_high=float(high),
        challenger_wins=int((difference < 0).sum()),
    )


def compare_all(
    summary: pd.DataFrame,
    reference: str,
    metric: str = "mae_c",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare every other model against one reference, on every fold set."""
    scoped = per_fold(summary)
    models = [name for name in scoped["model"].unique() if name != reference]
    rows = [
        compare(summary, challenger, reference, folds, metric, resamples, seed).as_row()
        for folds, challenger in itertools.product(
            sorted(scoped["folds"].unique()), sorted(models)
        )
    ]
    return pd.DataFrame(rows)


def _write(writer: Callable[[Path], Any], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


def _report(row: pd.Series) -> None:
    verdict = (
        "real difference"
        if row["significant"]
        else "indistinguishable from the reference"
    )
    direction = "better" if row["mean_difference"] < 0 else "worse"
    print(
        f"  {row['challenger']:<12s} vs {row['reference']:<12s} "
        f"{row['mean_difference']:+6.3f} C ({direction})  "
        f"95% CI [{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]  "
        f"wins {row['challenger_wins']:>3d}/{row['n_folds']:<3d}  "
        f"{verdict}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--reference", default="prophet")
    parser.add_argument("--metric", default="mae_c")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(
            f"No backtest summary at {args.summary}; run "
            "rdu_temperature.evaluation.backtest."
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it."
        )

    summary = pd.read_csv(args.summary)
    comparisons = compare_all(summary, args.reference, args.metric, args.resamples)
    for folds, group in comparisons.groupby("folds", sort=True):
        print(f"\n{folds}: paired {args.metric} against {args.reference}", flush=True)
        for _, row in group.iterrows():
            _report(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(lambda path: comparisons.to_csv(path, index=False), args.output)
    print(f"\nWrote comparison: {args.output}", flush=True)


if __name__ == "__main__":
    main()
