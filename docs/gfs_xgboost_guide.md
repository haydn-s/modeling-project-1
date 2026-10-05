# GFS Data + XGBoost Guide

Teammate handbook for the GFS forecast archive and the XGBoost models built on top of it.

**Status:** the XGBoost work is done and submitted as PR #5 (September version)
and PR #6 (full-year rolling version, stacked on #5). This guide covers what was
built, how to reuse the GFS data in your own model, and how to reproduce everything.

---

## 1. TL;DR

- **GFS archive:** 30 complete NOAA GFS runs in `data/raw/noaa_gfs/` (see §2).
  The download-timeout problem is fixed by a resumable downloader (see §3).
- **Panels:** `data/processed/gfs_forecast_panel.parquet` (September experiment)
  and `data/processed/gfs_rolling_panel.parquet` (rolling experiment) hold matched
  forecast/observation pairs, ready to join with the target on `valid_time_utc`.
- **Models:** `src/rdu_temperature/models/run_xgboost.py` (September) and
  `run_xgboost_rolling.py` (rolling). Both write a saved model, metrics JSON,
  and a 336-hour forecast CSV under `artifacts/`.
- **No leakage:** every GFS run was initialized before the 2026-09-17 00:00
  cutoff, so all of this is legal to use for the forecast period.

---

## 2. The GFS archive (`data/raw/noaa_gfs/`)

30 CSV files, one per GFS run, all complete (no `.partial.csv` leftovers):

| Runs | Inits | Purpose |
|---|---|---|
| 26 rolling | 2022-09-16 → 2026-07-17, ~56 days apart | Full-year training data (Hayden's original plan) |
| 3 September | 2023/2024/2025 September runs | September backtest folds |
| 1 forecast | 2026-09-16 12Z | Covers the 2026-09-17 → 09-30 forecast window |

Each file holds 336 hourly leads for station KRDU (RDU airport). File naming:
`gfs_<init-slug>.csv`. A file only appears as `.csv` once the whole run has
downloaded — anything still in flight is a `.partial.csv`, so "30 CSVs, 0 partials"
means the archive is complete.

Verified: 30/30 complete, 5,460 published leads, panel builds to 10,080 hourly
rows (30 × 336) with zero missing `temperature_c` and zero screening masks.

---

## 3. The resumable downloader (timeout fix)

**Script:** `src/rdu_temperature/pipeline/ingest_gfs.py`

The old problem: downloading 26 runs in one shot timed out partway and the whole
run was lost. The fix has three layers:

1. **Retries with backoff** — `urllib3 Retry` on the HTTPS session, generous
   timeouts `(connect=15s, read=180s)`.
2. **Per-lead checkpointing** — the `.partial.csv` is updated atomically after
   *each lead*, so a retry loses at most the single request in flight.
3. **Atomic completion** — the partial file is renamed to `.csv` only when the
   run is fully downloaded. Re-running the command resumes where it stopped.

**Commands** (from the repo root):

```bash
# The single run covering the forecast period
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_gfs --runs forecast

# Forecast run + one September run per backtest fold
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_gfs --runs seasonal

# Forecast run + rolling folds (every 56 days; --fold-step 4 = every 4th fold)
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_gfs --runs rolling --fold-step 4

# Start over for a run that looks corrupt
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_gfs --runs rolling --overwrite
```

Config lives in `config/weather_sources.json` (`noaa_gfs` section): station
coordinates, cutoff, horizon. Station defaults to `KRDU`.

---

## 4. Panel building

**Script:** `src/rdu_temperature/pipeline/clean_gfs.py`

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.clean_gfs
```

Reads every `gfs_*.csv` in `data/raw/noaa_gfs/` and writes:

- `data/processed/gfs_forecast_panel.parquet` — matched forecast/observation panel
- `data/processed/gfs_screening_report.csv` — what screening masked

(The rolling script builds its own panel as `gfs_rolling_panel.parquet`; see §5.)

**Dedupe rule (important):** one valid hour can appear in two runs (e.g. the
2024-09-13 rolling run and the 2024-09-16 September run overlap on 264 hours).
Before training, the panel is deduplicated on `valid_time_utc`, keeping the
**latest `init_time_utc`** for each hour. This is handled automatically inside
the rolling training script.

---

## 5. The XGBoost models

### 5a. September version — `src/rdu_temperature/models/run_xgboost.py`

Trains on September-period GFS runs from prior years only.

```bash
PYTHONPATH=src .VENV/bin/python -m rdu_temperature.models.run_xgboost
```

Writes (paths configurable via `--model-dir`, `--metrics`, `--forecast`):

- `artifacts/models/xgboost_gfs/` — saved model (`model.ubj`) + `metadata.json`
- `artifacts/metrics/xgboost_gfs.json`
- `artifacts/predictions/xgboost_gfs_forecast.csv` — 336-hour forecast

Metrics (holdout run `2025-09-16 12:00`, 2,015 training rows / 6 runs):

| | MAE | RMSE | bias |
|---|---|---|---|
| XGBoost | 2.43°C | 3.15°C | −1.14°C |
| raw GFS | 2.49°C | 3.31°C | −1.12°C |

### 5b. Rolling version — `src/rdu_temperature/models/run_xgboost_rolling.py`

Trains on the full 30-run archive.

```bash
PYTHONPATH=src .VENV/bin/python -m rdu_temperature.models.run_xgboost_rolling
```

Writes:

- `artifacts/models/xgboost_rolling/` — `model.ubj` + `metadata.json`
- `artifacts/metrics/xgboost_rolling.json`
- `artifacts/predictions/xgboost_rolling_forecast.csv`

Metrics (holdout run `2026-07-17 12:00`, 9,471 training rows / 29 runs):

| | MAE | RMSE | bias |
|---|---|---|---|
| XGBoost | 1.79°C | 2.39°C | +0.22°C |
| raw GFS | 2.70°C | 3.50°C | +1.21°C |

Selected features: `target_day_of_year_sin/cos`, `gfs_temperature_c`,
`gfs_dewpoint_c`, `gfs_precipitation_mm`, `target_hour_sin`,
`forecast_lead_hour`, `gfs_interpolated`.

> Note: the September and rolling numbers use *different* holdout runs, so the
> 1.79 vs 2.43 gap is indicative, not a paired comparison.

---

## 6. Using the GFS data in YOUR model

This is the part for George (linear regression) and Haydn (Prophet). You do **not**
need to re-download anything.

**Minimal recipe (pandas):**

```python
import pandas as pd

gfs = pd.read_parquet("data/processed/gfs_rolling_panel.parquet")
target = pd.read_parquet("data/processed/rdu_hourly_target.parquet")

# One row per valid hour, GFS columns are prefixed gfs_
df = target.merge(gfs, on="valid_time_utc", how="left")

# Example: linear regression on the GFS temperature + calendar features
X = df[["gfs_temperature_c", "gfs_dewpoint_c", "forecast_lead_hour"]]
y = df["temperature_c"]  # your target column
```

**What you get for free:**

- 30 runs × 336 leads of GFS covariates, already cleaned and screened.
- The `forecast` run (2026-09-16 12Z) gives you covariates for the actual
  2026-09-17 → 09-30 scoring window — just filter
  `init_time_utc == "2026-09-16 12:00"`.
- If you ever need more runs: run the downloader in §3, then `clean_gfs`.

**For the final 336-hour forecast**, train on everything before the cutoff,
then predict with the 2026-09-16 12Z run's 336 leads — exactly what
`run_xgboost*.py` do (see their `--forecast` output).

---

## 7. Gotchas

- **Python:** the project env is `.VENV` (conda-style, no `bin/activate`).
  Always run with `.VENV/bin/python`, not bare `python`.
- **`.partial.csv` files** are normal mid-download. Only worry if one is stale
  (older than a day) — then re-run the downloader or pass `--overwrite`.
- **Overlapping valid hours** between runs are deduped by latest init
  (automatic in the rolling script; do the same in your own code if you build
  a custom panel).
- **CI** runs `ruff format --check`, `ruff check`, and `pytest` on every PR —
  run `ruff format --check src tests && ruff check src tests && pytest -q`
  locally before pushing.
- The main README's *"Deferred: forecast covariates"* section predates this
  work — the GFS pipeline described here is what un-defers it.
