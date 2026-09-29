from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from rdu_temperature.evaluation import figures
from rdu_temperature.features import prophet_frame as pf

CUTOFFS = ("2024-09-17 04:00:00", "2025-09-17 04:00:00")


def _predictions(hours: int = 48) -> pd.DataFrame:
    """Predictions as the backtest stores them: every model and fold set.

    The distractor rows matter. The figure reports one model on one fold set,
    so a fixture holding only those rows would pass even if the filter were
    dropped entirely.
    """
    frames = []
    for model, folds, offset in (
        (figures.FIGURE_MODEL, figures.FIGURE_FOLDS, 2.0),
        (figures.FIGURE_MODEL, "rolling", 9.0),
        ("climatology", figures.FIGURE_FOLDS, 9.0),
    ):
        for cutoff in CUTOFFS:
            start = pd.Timestamp(cutoff)
            timestamps = pd.date_range(start, periods=hours, freq="h")
            frames.append(
                pd.DataFrame(
                    {
                        pf.DS: timestamps,
                        "timestamp_local": pf.to_local(pd.Series(timestamps)),
                        "yhat": [20.0 + offset] * hours,
                        "yhat_lower": [17.0] * hours,
                        "yhat_upper": [27.0] * hours,
                        pf.Y: [20.0] * hours,
                        "cutoff": start,
                        "model": model,
                        "folds": folds,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _write(frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path)
    return path


def test_load_numbers_lead_days_from_one(tmp_path) -> None:
    loaded = figures.load(_write(_predictions(hours=49), tmp_path))

    first = loaded.loc[loaded["cutoff"] == pd.Timestamp(CUTOFFS[0])]

    # The cutoff hour itself is day one, not day zero.
    assert first["lead_day"].iloc[0] == 1
    assert first["lead_day"].iloc[23] == 1
    assert first["lead_day"].iloc[24] == 2
    assert first["lead_day"].iloc[48] == 3


def test_load_signs_the_error_forecast_minus_observed(tmp_path) -> None:
    loaded = figures.load(_write(_predictions(), tmp_path))

    # A forecast above the observation is a positive error, so a warm bias
    # reads positive throughout the figure.
    assert loaded["error_c"].to_numpy() == pytest.approx(2.0)


def test_load_selects_one_model_on_one_fold_set(tmp_path) -> None:
    stored = _predictions()
    loaded = figures.load(_write(stored, tmp_path))

    assert set(loaded["model"]) == {figures.FIGURE_MODEL}
    assert set(loaded["folds"]) == {figures.FIGURE_FOLDS}
    # A third of the stored rows, the other two being the distractors.
    assert len(loaded) == len(stored) // 3


def test_load_can_select_another_model(tmp_path) -> None:
    path = _write(_predictions(), tmp_path)

    loaded = figures.load(path, model="climatology")

    assert set(loaded["model"]) == {"climatology"}


def test_load_refuses_a_selection_that_matches_nothing(tmp_path) -> None:
    path = _write(_predictions(), tmp_path)

    with pytest.raises(ValueError, match="persistence"):
        figures.load(path, model="persistence")


def test_load_reports_missing_predictions(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="backtest"):
        figures.load(tmp_path / "absent.parquet")


def test_the_figure_draws_every_panel(tmp_path) -> None:
    loaded = figures.load(_write(_predictions(), tmp_path))

    figure = figures.ErrorFigure().draw(loaded)

    try:
        # One small multiple per fold, plus the two summary panels.
        assert len(figure.axes) == len(CUTOFFS) + 2
    finally:
        plt.close(figure)
