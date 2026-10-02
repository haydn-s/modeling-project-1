from __future__ import annotations

import pandas as pd
import pytest

from rdu_temperature.pipeline.ingest_gfs import (
    COARSE_LEAD_STEP,
    HOURLY_LEAD_LIMIT,
    GfsRun,
    RunPlan,
    available_leads,
    leads_covering,
    parse_index,
)

CUTOFF = pd.Timestamp("2026-09-17 04:00:00")
HORIZON = 336

# Three records in the order a .idx sidecar lists them, plus one that shares a
# name with the first through an averaging window.
INDEX_TEXT = """\
1:0:d=2026091612:TMP:2 m above ground:24 hour fcst:
2:500:d=2026091612:DPT:2 m above ground:24 hour fcst:
3:1200:d=2026091612:TMP:2 m above ground:18-24 hour ave fcst:
4:1800:d=2026091612:PRATE:surface:24 hour fcst:
"""


def test_leads_are_hourly_then_three_hourly() -> None:
    leads = available_leads(0, 132)

    # The product changes resolution at 120 hours; this is its property, not a
    # preference, so asking for 121 would 404.
    assert leads[:4] == (0, 1, 2, 3)
    assert HOURLY_LEAD_LIMIT in leads
    assert HOURLY_LEAD_LIMIT + 1 not in leads
    assert HOURLY_LEAD_LIMIT + COARSE_LEAD_STEP in leads
    beyond = [lead for lead in leads if lead > HOURLY_LEAD_LIMIT]
    assert all(lead % COARSE_LEAD_STEP == 0 for lead in beyond)


def test_leads_cover_the_half_open_window() -> None:
    init_time = pd.Timestamp("2026-09-16 12:00:00")

    leads = leads_covering(init_time, CUTOFF, CUTOFF + pd.Timedelta(hours=HORIZON))

    first = init_time + pd.Timedelta(hours=leads[0])
    last = init_time + pd.Timedelta(hours=leads[-1])
    # The window is half-open, so the final hour needing a prediction is one
    # hour before its end.
    assert first <= CUTOFF
    assert last >= CUTOFF + pd.Timedelta(hours=HORIZON - 1)


def test_leads_never_go_negative() -> None:
    init_time = pd.Timestamp("2026-09-17 04:00:00")

    leads = leads_covering(init_time, CUTOFF, CUTOFF + pd.Timedelta(hours=24))

    assert min(leads) >= 0


def test_parse_index_gives_byte_ranges_between_offsets() -> None:
    entries = parse_index(INDEX_TEXT)

    assert entries["TMP:2 m above ground"].byte_range == "0-499"
    assert entries["DPT:2 m above ground"].byte_range == "500-1199"
    # The last record runs to the end of the file, which has no next offset.
    assert entries["PRATE:surface"].byte_range == "1800-"


def test_parse_index_keeps_the_instantaneous_record() -> None:
    entries = parse_index(INDEX_TEXT)

    # The averaged variant shares the name and sits later in the file; taking
    # it would silently swap an instantaneous field for a time average.
    assert entries["TMP:2 m above ground"].start == 0


def test_parse_index_ignores_malformed_lines() -> None:
    entries = parse_index("garbage\n\n" + INDEX_TEXT)

    assert len(entries) == 3


def test_run_url_encodes_the_initialisation() -> None:
    run = GfsRun(pd.Timestamp("2026-09-16 12:00:00"), (24,))

    url = run.url(24)

    # The path is what records when the run started, which is the whole basis
    # for the covariate being legitimate.
    assert "gfs.20260916/12/atmos/gfs.t12z.pgrb2.0p25.f024" in url
    assert run.slug == "20260916T1200Z"


def test_the_chosen_run_starts_before_the_cutoff() -> None:
    plan = RunPlan(cutoff=CUTOFF, horizon_hours=HORIZON)

    run = plan.run_for(CUTOFF)

    assert run.init_time == pd.Timestamp("2026-09-16 12:00:00")
    assert run.init_time < CUTOFF


def test_a_training_fold_gets_the_same_lead_in_as_the_real_forecast() -> None:
    plan = RunPlan(cutoff=CUTOFF, horizon_hours=HORIZON)
    fold = pd.Timestamp("2025-09-17 04:00:00")

    run = plan.run_for(fold)

    # A fold trained on a fresher run than the real forecast can have would
    # learn from a covariate better than the one it will be used with.
    assert run.init_time == pd.Timestamp("2025-09-16 12:00:00")
    assert (fold - run.init_time) == (CUTOFF - plan.run_for(CUTOFF).init_time)


def test_a_run_after_the_cutoff_is_refused() -> None:
    plan = RunPlan(cutoff=CUTOFF, horizon_hours=HORIZON)
    later = CUTOFF + pd.Timedelta(days=30)

    # The cutoff is the contract: a run initialised inside the forecast period
    # has seen what the project is meant to predict.
    with pytest.raises(ValueError, match="not before the data cutoff"):
        plan.run_for(later)


def test_cutoffs_sharing_a_run_are_fetched_once() -> None:
    plan = RunPlan(cutoff=CUTOFF, horizon_hours=HORIZON)
    close_together = [
        pd.Timestamp("2025-09-17 04:00:00"),
        pd.Timestamp("2025-09-17 10:00:00"),
    ]

    runs = plan.for_cutoffs(close_together)

    assert len(runs) == 1
