from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdu_temperature.evaluation.compare import (
    TOTAL_ROW,
    compare,
    compare_all,
    per_fold,
)

# A small resample count keeps the tests quick; the interval is still stable
# because every fixture below uses a fixed seed.
RESAMPLES = 2000


def _summary(errors: dict[str, list[float]], folds: str = "rolling") -> pd.DataFrame:
    rows = []
    for model, per_fold_mae in errors.items():
        for index, mae in enumerate(per_fold_mae):
            rows.append(
                {
                    "cutoff": f"2024-01-{index + 1:02d} 00:00:00",
                    "model": model,
                    "folds": folds,
                    "mae_c": mae,
                    "scored_hours": 336,
                }
            )
        rows.append(
            {
                "cutoff": TOTAL_ROW,
                "model": model,
                "folds": folds,
                "mae_c": float(np.mean(per_fold_mae)),
                "scored_hours": 336 * len(per_fold_mae),
            }
        )
    return pd.DataFrame(rows)


def test_per_fold_drops_the_weighted_total() -> None:
    summary = _summary({"prophet": [1.0, 2.0]})

    assert TOTAL_ROW not in set(per_fold(summary)["cutoff"])
    assert len(per_fold(summary)) == 2


def test_a_consistently_better_challenger_reads_negative_and_significant() -> None:
    summary = _summary(
        {
            "prophet": [3.0, 3.2, 3.1, 2.9, 3.3, 3.0, 3.1, 3.2],
            "challenger": [2.0, 2.2, 2.1, 1.9, 2.3, 2.0, 2.1, 2.2],
        }
    )

    result = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)

    # Every metric is an error, so better is smaller and the difference is
    # negative.
    assert result.mean_difference == pytest.approx(-1.0)
    assert result.challenger_wins == 8
    assert result.significant


def test_a_wash_is_reported_as_indistinguishable() -> None:
    summary = _summary(
        {
            "prophet": [3.0, 2.0, 4.0, 3.0, 2.5, 3.5, 2.8, 3.2],
            "challenger": [2.0, 3.0, 3.0, 4.0, 3.5, 2.5, 3.2, 2.8],
        }
    )

    result = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)

    # The folds disagree about which model won, so the interval spans zero.
    assert result.ci_low < 0.0 < result.ci_high
    assert not result.significant


def test_comparison_is_paired_rather_than_averaged() -> None:
    # Both models average 3.0, but the challenger wins every fold by a hair
    # once the fold's own difficulty is taken out.
    summary = _summary(
        {
            "prophet": [1.1, 3.1, 5.1, 1.1, 3.1, 5.1],
            "challenger": [1.0, 3.0, 5.0, 1.0, 3.0, 5.0],
        }
    )

    result = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)

    assert result.challenger_wins == 6
    assert result.mean_difference == pytest.approx(-0.1)
    # An unpaired comparison of these two spreads could not resolve this.
    assert result.significant


def test_compare_is_reproducible() -> None:
    summary = _summary({"prophet": [3.0, 2.0, 4.0], "challenger": [2.0, 3.0, 3.0]})

    first = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)
    second = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)

    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_compare_refuses_a_model_the_folds_do_not_hold() -> None:
    summary = _summary({"prophet": [1.0, 2.0]})

    with pytest.raises(ValueError, match="absent"):
        compare(summary, "absent", "prophet", "rolling", resamples=RESAMPLES)


def test_compare_all_covers_every_model_and_fold_set() -> None:
    rolling = _summary(
        {"prophet": [3.0, 3.0], "climatology": [2.0, 2.0], "persistence": [4.0, 4.0]},
        folds="rolling",
    )
    seasonal = _summary(
        {"prophet": [3.0, 3.0], "climatology": [2.0, 2.0], "persistence": [4.0, 4.0]},
        folds="seasonal",
    )

    comparisons = compare_all(
        pd.concat([rolling, seasonal], ignore_index=True),
        reference="prophet",
        resamples=RESAMPLES,
    )

    # Two challengers on each of two fold sets; the reference never compares
    # against itself.
    assert len(comparisons) == 4
    assert set(comparisons["challenger"]) == {"climatology", "persistence"}
    assert set(comparisons["folds"]) == {"rolling", "seasonal"}
    assert "prophet" not in set(comparisons["challenger"])


def test_win_rate_matches_the_fold_count() -> None:
    summary = _summary(
        {"prophet": [3.0, 3.0, 3.0, 3.0], "challenger": [2.0, 2.0, 4.0, 4.0]}
    )

    result = compare(summary, "challenger", "prophet", "rolling", resamples=RESAMPLES)
    row = result.as_row()

    assert row["challenger_wins"] == 2
    assert row["win_rate"] == pytest.approx(0.5)
