"""Train the GFS-correcting XGBoost model and write its final forecast.

Run from the repository root with:

    python -m rdu_temperature.models.run_xgboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdu_temperature.features import forecast_covariates, prophet_frame, xgboost_frame
from rdu_temperature.models.xgboost_model import (
    XGBoostFeatureSelector,
    XGBoostForecaster,
    XGBoostForecastPipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "rdu_hourly_target.parquet"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "xgboost_gfs"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "xgboost_gfs.json"
DEFAULT_FORECAST_PATH = (
    PROJECT_ROOT / "artifacts" / "predictions" / "xgboost_gfs_forecast.csv"
)

# Shallow trees did not consistently correct GFS across historical runs. Depth
# four with a small leaf-size guard improved pooled forward-run MAE and RMSE
# while remaining modest for roughly two thousand training rows.
DEFAULT_GFS_MODEL_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "min_child_weight": 5,
}


def _metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae_c": float(error.abs().mean()),
        "rmse_c": float(np.sqrt(error.pow(2).mean())),
        "bias_c": float(error.mean()),
    }


def run(
    *,
    panel_path: Path = forecast_covariates.DEFAULT_PANEL_PATH,
    target_path: Path = DEFAULT_TARGET_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    forecast_path: Path = DEFAULT_FORECAST_PATH,
    max_features: int = 8,
) -> dict[str, Path]:
    """Validate on the latest historical run, refit, and forecast 336 hours."""
    if not target_path.exists():
        raise FileNotFoundError(
            f"No cleaned target at {target_path}; run clean_weather first."
        )
    panel = forecast_covariates.load_panel(panel_path)
    target = pd.read_parquet(target_path)
    cutoff = prophet_frame.forecast_cutoff()
    data = xgboost_frame.build_training_data(panel, target, cutoff)

    validation_run = data.run_times.max()
    validation_mask = data.run_times.eq(validation_run).to_numpy()
    training_mask = ~validation_mask
    train_features = data.features.loc[training_mask]
    train_target = data.target.loc[training_mask]
    validation_features = data.features.loc[validation_mask]
    validation_target = data.target.loc[validation_mask]
    if train_features.empty or validation_features.empty:
        raise ValueError("At least two historical GFS runs are required.")

    raw_train = train_features[xgboost_frame.GFS_TEMPERATURE]
    raw_validation = validation_features[xgboost_frame.GFS_TEMPERATURE]
    training_residual = train_target - raw_train
    validation_residual = validation_target - raw_validation

    validation_pipeline = XGBoostForecastPipeline(
        selector=XGBoostFeatureSelector(
            max_features=max_features,
            model_params=DEFAULT_GFS_MODEL_PARAMS,
        ),
        forecaster=XGBoostForecaster(model_params=DEFAULT_GFS_MODEL_PARAMS),
    ).fit(
        train_features,
        training_residual,
        validation_features=validation_features,
        validation_target=validation_residual,
    )
    corrected = raw_validation + validation_pipeline.predict(validation_features)

    fitted = validation_pipeline.forecaster.model_
    best_iteration = getattr(fitted, "best_iteration", None)
    final_params: dict[str, Any] = dict(DEFAULT_GFS_MODEL_PARAMS)
    if best_iteration is not None:
        final_params["n_estimators"] = int(best_iteration) + 1

    final_pipeline = XGBoostForecastPipeline(
        selector=XGBoostFeatureSelector(
            max_features=max_features,
            model_params=DEFAULT_GFS_MODEL_PARAMS,
        ),
        forecaster=XGBoostForecaster(
            model_params=final_params,
            early_stopping_rounds=None,
        ),
    ).fit(
        data.features,
        data.target - data.features[xgboost_frame.GFS_TEMPERATURE],
    )

    future = xgboost_frame.build_forecast_data(
        panel,
        cutoff,
        prophet_frame.FORECAST_HOURS,
    )
    raw_future = future.features[xgboost_frame.GFS_TEMPERATURE]
    future_prediction = raw_future + final_pipeline.predict(future.features)
    utc_times = pd.to_datetime(future.valid_times, utc=True)
    local_times = utc_times.dt.tz_convert("America/New_York")
    forecast = pd.DataFrame(
        {
            "timestamp_utc": utc_times.map(
                lambda value: value.isoformat().replace("+00:00", "Z")
            ).to_numpy(),
            "timestamp_local": local_times.map(pd.Timestamp.isoformat).to_numpy(),
            "predicted_temperature_c": future_prediction.to_numpy(),
            "raw_gfs_temperature_c": raw_future.to_numpy(),
        }
    )

    model_paths = final_pipeline.save(model_dir)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(forecast_path, index=False)

    report = {
        "training_rows": len(data.features),
        "historical_runs": int(data.run_times.nunique()),
        "validation_run": str(validation_run),
        "validation_rows": len(validation_features),
        "target_definition": "observed_temperature_c - gfs_temperature_c",
        "xgboost_validation": _metrics(corrected, validation_target),
        "raw_gfs_validation": _metrics(raw_validation, validation_target),
        "best_iteration": best_iteration,
        "selected_features": final_pipeline.selected_features_,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote forecast: {forecast_path}", flush=True)
    return {
        **model_paths,
        "metrics": metrics_path,
        "forecast": forecast_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=forecast_covariates.DEFAULT_PANEL_PATH
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST_PATH)
    parser.add_argument("--max-features", type=int, default=8)
    args = parser.parse_args()
    run(
        panel_path=args.panel,
        target_path=args.target,
        model_dir=args.model_dir,
        metrics_path=args.metrics,
        forecast_path=args.forecast,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
