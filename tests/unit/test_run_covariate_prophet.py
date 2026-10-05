"""The covariate backtest's wiring, which a long run would otherwise expose.

Everything here fails in under a second or twenty minutes in, depending on
whether it is tested: the reference lookup and the fold selection both happen
at the ends of the run, after every model has been refit.
"""

from __future__ import annotations

import pandas as pd

from rdu_temperature.features import prophet_frame as pf
from rdu_temperature.models import run_covariate_prophet as runner
from rdu_temperature.models.covariate_prophet import CovariateProphet
from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_gfs import INTERPOLATED


def _panel(init: str, first_valid: str, hours: int) -> pd.DataFrame:
    init_time = pd.Timestamp(init)
    valid = pd.date_range(pd.Timestamp(first_valid), periods=hours, freq="h")
    return pd.DataFrame(
        {
            schema.SOURCE: "noaa_gfs",
            schema.STATION_ID: "KRDU",
            schema.INIT_TIME_UTC: init_time,
            schema.VALID_TIME_UTC: valid,
            schema.LEAD_HOURS: ((valid - init_time).total_seconds() // 3600).astype(
                int
            ),
            schema.TEMPERATURE_C: [20.0] * hours,
            schema.DEWPOINT_C: [10.0] * hours,
            schema.WIND_SPEED_MS: [2.0] * hours,
            schema.WIND_DIRECTION_DEG: [180.0] * hours,
            schema.CLOUD_COVER_PCT: [50.0] * hours,
            schema.PRECIPITATION_MM: [0.0] * hours,
            INTERPOLATED: [False] * hours,
        }
    )


def test_the_comparison_reference_is_one_of_the_scored_models() -> None:
    models = runner.build_models(_panel("2026-09-16 12:00", "2026-09-17 04:00", 24))

    # compare_all resolves the reference by name, at the very end of a run
    # that has already refit every model over every fold. A rename here would
    # surface as a ValueError twenty minutes in.
    assert runner.REFERENCE_MODEL in models


def test_the_default_covariate_model_is_the_flat_correction() -> None:
    models = runner.build_models(_panel("2026-09-16 12:00", "2026-09-17 04:00", 24))

    plain = models["prophet_gfs"]()
    seasonal = models["prophet_gfs_seasonal"]()

    assert isinstance(plain, CovariateProphet)
    # The two differ only in the switch, which is the comparison the fold sets
    # are there to settle.
    assert plain.seasonality is False
    assert seasonal.seasonality is True
    assert plain.regressors == seasonal.regressors


def test_fold_sets_are_empty_when_no_run_covers_a_fold() -> None:
    frame = pf.add_season_flags(
        pd.DataFrame(
            {
                pf.DS: pd.date_range("2021-09-17 04:00", periods=24 * 400, freq="h"),
                pf.Y: [18.0] * (24 * 400),
            }
        )
    )
    # A single run covering a day, nowhere near a fold's full horizon.
    panel = _panel("2022-03-01 12:00", "2022-03-02 00:00", 24)

    sets = runner.fold_sets(frame, panel, step_days=14)

    # Named, empty, and in a fixed order, so the runner reports the skip
    # rather than failing on an empty cutoff list.
    assert [folds.name for folds in sets] == ["seasonal", "rolling"]
    assert all(folds.cutoffs == [] for folds in sets)


def _difference(n_folds: int, significant: bool) -> pd.Series:
    return pd.Series(
        {
            "challenger": "prophet_gfs",
            "mean_difference": -0.436,
            "ci_low": -0.572,
            "ci_high": -0.301,
            "challenger_wins": n_folds,
            "n_folds": n_folds,
            "significant": significant,
        }
    )


def test_a_difference_too_few_folds_can_resolve_claims_nothing(capsys) -> None:
    # The seasonal set really does come out at two folds on the current
    # archive, and resampling two numbers yields an interval tight enough to
    # read as a confident finding. The verdict is withheld rather than shown.
    runner._report_difference(_difference(2, significant=True))

    printed = capsys.readouterr().out
    assert "too few folds to resolve (2)" in printed
    assert "real difference" not in printed


def test_a_difference_the_folds_can_resolve_is_reported(capsys) -> None:
    runner._report_difference(_difference(19, significant=True))

    assert "real difference" in capsys.readouterr().out
