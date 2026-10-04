"""Run the linear regression model end to end. Open in VS Code and press Run.

1. Backtest on earlier Septembers (cutoff Sep 17 of each past year, 336 hours),
   comparing the linear model with and without the anomaly correction against
   the two baselines.
2. Fit on all history and forecast the real period, Sep 17-30 2026.

Needs data/processed/rdu_hourly_target.parquet, produced by the data pipeline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rdu_temperature.features import prophet_frame
from rdu_temperature.models.baselines import (
    Climatology,
    SeasonalPersistence,
)
from rdu_temperature.models.linear_model import LinearTemperatureModel

HORIZON = prophet_frame.FORECAST_HOURS
OUTPUT_DIR = ROOT / "artifacts" / "metrics"

MODELS = {
    "linear": LinearTemperatureModel,
    "linear_no_anomaly": lambda: LinearTemperatureModel(use_anomaly=False),
    "climatology": Climatology,
    "persistence": SeasonalPersistence,
}


def seasonal_cutoffs(frame: pd.DataFrame, cutoff: pd.Timestamp) -> list:
    """Sep 17 of each earlier year with a full horizon and a year of history."""
    first, last = frame[prophet_frame.DS].min(), frame[prophet_frame.DS].max()
    folds, candidate = [], cutoff - pd.DateOffset(years=1)
    while candidate - first >= pd.Timedelta(days=365):
        if candidate + pd.Timedelta(hours=HORIZON - 1) <= last:
            folds.append(pd.Timestamp(candidate))
        candidate -= pd.DateOffset(years=1)
    return sorted(folds)


def backtest(frame: pd.DataFrame, cutoffs: list) -> pd.DataFrame:
    rows = []
    for name, factory in MODELS.items():
        for cutoff in cutoffs:
            history = frame[frame[prophet_frame.DS] < cutoff]
            truth = frame[frame[prophet_frame.DS] >= cutoff][
                [prophet_frame.DS, prophet_frame.Y]
            ]
            forecast = factory().fit(history).forecast(cutoff, HORIZON)
            scored = forecast.merge(truth, on=prophet_frame.DS).dropna(
                subset=[prophet_frame.Y]
            )
            error = scored["yhat"] - scored[prophet_frame.Y]
            rows.append(
                {
                    "model": name,
                    "cutoff": cutoff,
                    "mae_c": error.abs().mean(),
                    "rmse_c": np.sqrt((error**2).mean()),
                    "bias_c": error.mean(),
                    "hours": len(scored),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    frame = prophet_frame.load()
    cutoff = prophet_frame.forecast_cutoff()
    print(f"Loaded {len(frame):,} hours; real forecast starts {cutoff} UTC.")

    cutoffs = seasonal_cutoffs(frame, cutoff)
    print(f"Backtest folds: {[c.date() for c in cutoffs]}\n")
    results = backtest(frame, cutoffs)

    print("Per fold:")
    print(results.round({"mae_c": 2, "rmse_c": 2, "bias_c": 2}).to_string(index=False))
    print("\nAverage over folds (lower is better):")
    print(
        results.groupby("model")[["mae_c", "rmse_c", "bias_c"]]
        .mean()
        .sort_values("mae_c")
        .round(2)
        .to_string()
    )

    model = LinearTemperatureModel().fit(frame)
    forecast = model.forecast(cutoff, HORIZON)
    print("\nCoefficients:")
    print(model.coefficients.round(3).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "linear_backtest.csv", index=False)
    forecast.to_csv(OUTPUT_DIR / "linear_forecast_2026.csv", index=False)
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
