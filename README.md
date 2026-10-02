# RDU Hourly Temperature Forecasting

[![CI](https://github.com/haydn-s/modeling-project-1/actions/workflows/ci.yml/badge.svg)](https://github.com/haydn-s/modeling-project-1/actions/workflows/ci.yml)

This project predicts the hourly temperature measured at Raleigh-Durham International Airport (RDU) from September 17, 2026 at 12:00 a.m. through September 30, 2026 at 11:00 p.m.

Only information available before September 17, 2026 at 12:00 a.m. may be used to build the forecasts.

## Team

- George Deng
- Cheney Li
- Haydn Stucker

## Project requirements

- Build at least one linear regression model.
- Build at least one additional model.
- Evaluate the models without using information from the forecast period.
- Deliver a presentation, code repository, and a 2–4 page writeup.

## Repository structure

- `src/rdu_temperature/`: Reusable project source code
- `src/rdu_temperature/pipeline/`: Data collection and preparation code
- `src/rdu_temperature/features/`: Feature engineering code
- `src/rdu_temperature/models/`: Model implementations
- `src/rdu_temperature/evaluation/`: Model evaluation code
- `tests/unit/`: Unit tests
- `tests/integration/`: Integration tests
- `notebooks/`: Exploratory analysis and experiments
- `config/`: Project configuration
- `data/raw/`: Original source data
- `data/processed/`: Prepared modeling data
- `artifacts/models/`: Saved model artifacts
- `artifacts/metrics/`: Saved evaluation metrics
- `reports/figures/`: Figures for the presentation and writeup
- `reports/tables/`: Tables for the presentation and writeup
- `docs/`: Project documentation and requirements
- `.github/workflows/`: Continuous integration
- `.env.example`: Template for local environment variables
- `requirements.txt`: Python dependencies

## Setup

1. Activate the virtual environment:

   ```bash
   source .VENV/bin/activate
   ```

2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On macOS, XGBoost also needs the OpenMP runtime, which is a system package
   rather than a wheel. Without it `import xgboost` fails, and because the
   `models` and `evaluation` packages re-export the XGBoost classes, that
   failure takes the whole test suite with it:

   ```bash
   brew install libomp
   ```

   Linux wheels carry their own OpenMP, so CI needs nothing extra.

3. Install the commit-message hook:

   ```bash
   pre-commit install --hook-type commit-msg
   ```

4. Create a local environment file:

   ```bash
   cp .env.example .env
   ```

5. Add any required local values to `.env`.

## Conventional commits

This repository uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to keep the Git history consistent and readable. Commitlint checks each commit message through the configured `commit-msg` hook.

Use this format:

```text
<type>(optional-scope): <description>
```

Common commit types include:

- `feat`: Add or change project functionality
- `fix`: Correct a defect
- `docs`: Change documentation only
- `test`: Add or update tests
- `refactor`: Restructure code without changing its behavior
- `chore`: Perform maintenance or repository setup
- `build`: Change dependencies or build configuration
- `ci`: Change continuous-integration configuration

Examples:

```text
chore(scaffold): add initial project structure
feat(pipeline): collect historical weather observations
feat(models): add linear regression baseline
fix(features): prevent forecast-period data leakage
docs(readme): document the evaluation approach
```

Use `!` before the colon for a breaking change, or add a `BREAKING CHANGE:` footer to the commit body.

## Continuous integration

Every pull request, and every push to `main`, runs three jobs:

- **Lint**: `ruff format --check` and `ruff check` over `src` and `tests`
- **Tests**: the full `pytest` suite, then `--help` on each command-line entry
  point
- **Commit messages**: the repository's own commitlint hook, over every commit
  in the pull request

The jobs reproduce locally with no extra tooling:

```bash
ruff format --check src tests && ruff check src tests && pytest -q
```

Two things worth knowing about what CI can and cannot check here.

The generated datasets are not in the repository, so no job runs the pipeline,
the backtest, or the figures. The test suite builds every fixture it needs,
which is what makes it meaningful without the data; the entry-point check
covers argument parsing only. Anything that depends on real observations has to
be run and reviewed by hand.

The lint and commit-message jobs install a single pinned tool read straight out
of `requirements.txt`, rather than repeating a version in the workflow. CI
cannot drift from what runs locally without the pin changing first.

## Data sources

The initial ingestion window is September 17, 2021 at 12:00 a.m. through
September 16, 2026 at 11:59 p.m. in `America/New_York`. The equivalent UTC
interval is `[2021-09-17T04:00:00Z, 2026-09-17T04:00:00Z)`, which prevents
forecast-period information from entering the training data.

The pipeline uses four complementary sources:

- NOAA GHCN-hourly for quality-controlled RDU and surrounding airport-station
  observations
- NC State ECONet for nearby research-grade, non-airport observations
- Open-Meteo ERA5-Seamless for a consistent gridded regional baseline. It uses
  ERA5-Land for higher-resolution temperature and humidity, complemented by ERA5
  for precipitation, pressure, clouds, wind, and solar radiation.
- Iowa Environmental Mesonet for a convenient METAR cross-check

Five years is the initial analysis window. The project may expand this to ten
years later to evaluate whether the additional annual cycles improve model
performance.

## GFS forecast covariates

NOAA GFS supplies the one thing observations cannot: a prediction for a period
that has not happened. A model run records when it was initialised, so a run
that started before the data cutoff knew nothing after it, and its forecasts
for the scored window are fair to use. This is shared infrastructure — the
panel it produces is model-agnostic, and both the Prophet and XGBoost work read
from it.

Fetch the run that covers the forecast period, then build the panel:

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_gfs --runs forecast
```

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.clean_gfs
```

For covariates over the backtest folds as well, pass `--runs seasonal` (one run
per September fold) or `--runs rolling --fold-step 4` (every fourth rolling
fold). Runs are written one file per initialisation and an existing file is
skipped, so a long fetch can be interrupted and resumed, or built up in stages.

`data/processed/gfs_forecast_panel.parquet` holds one row per station, run, and
valid hour, in the same column vocabulary as the observation panel.

### What to know before using it

- **Bandwidth is the cost, not disk.** A GFS 0.25-degree file is about 520 MB
  and the project wants six fields at one grid point, so each file is read by
  byte range through its `.idx` sidecar: about half a megabyte per field
  instead of 520. Only the extracted point values are kept, so the forecast run
  is 28 KB on disk against roughly 0.76 GB transferred. A full rolling fetch
  moves about 20 GB and stores under a megabyte.
- **The back of the horizon is coarser than the front.** The product is hourly
  to 120 hours and three-hourly beyond, so 154 of the forecast window's 336
  hours are interpolated between published leads. Every row carries an
  `interpolated` flag, so a result that leans on them can be told from one that
  does not.
- **Wind is interpolated as components, then converted.** Bearings cannot be
  interpolated across north, where the arithmetic midpoint of 350 and 10
  degrees is due south. Eastward and northward components are filled first and
  the speed and bearing derived afterwards.
- **Training folds get the same lead time as the real forecast.** Each fold's
  covariate comes from a run initialised the same number of hours before its
  cutoff as the real run is before the real cutoff. A fold handed a fresher run
  would learn from a covariate better than the one it will be used with.
- **`PRATE`, not `APCP`.** Accumulated precipitation is published over windows
  that change with lead, which cannot be placed on an hourly grid without
  inventing a disaggregation. `PRATE` is the rate at the valid hour.
- **The archive begins in spring 2021**, which covers this project's window.
  The note below previously said January 2021; probing the bucket puts the
  earliest 0.25-degree run between March and June of that year.

### Superseded: forecast covariates

Numerical weather prediction archives encode each run's initialization time in
its file path, so a run initialized before the cutoff can legitimately supply
predicted covariates for the forecast period. NOAA GFS (0.25 degree, 384-hour
lead, archived from 2021-01-01), NOAA GEFS, and ECMWF IFS and AIFS (360-hour
lead, archived from 2023-01-18) each cover the full target window from the
2026-09-16 12Z run. GFS is the strongest candidate because its archive aligns
with this project's 2021 ingestion window, which is what makes it possible to
train on matched forecast and observation pairs.

GFS is now implemented; see the section above. Two details in this note turned
out to be wrong once the archive was probed: the 0.25-degree archive begins in
spring 2021 rather than on January 1, and the product is hourly only to 120
hours, so most of a fourteen-day horizon arrives three-hourly. Neither changes
the conclusion that GFS was the right candidate.

Open-Meteo is not a safe substitute. Its archive endpoint returns ERA5
reanalysis, and its previous-runs endpoint returns a `temperature_2m` series
that is analysis assimilating observations. The `previous_dayN` series are
genuine forecasts, but their lead rolls with valid time, so late-window values
come from runs initialized inside the forecast period.

## Data pipeline

Copy `.env.example` to `.env`, then add the CLOUDS API hash associated with your
NC State Climate Office account. Run all ingestion sources from the repository
root:

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_weather
```

Run one or more public sources without an ECONet credential by repeating
`--source`:

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.ingest_weather \
  --source noaa-ghcnh \
  --source open-meteo \
  --source iem-asos
```

Each source writes a separate timestamped CSV beneath `data/raw/`. Existing
files are protected unless `--overwrite` is supplied. Station selections,
variables, the model version, and the exact data cutoff are maintained in
`config/weather_sources.json`.

## Data cleaning

Run the cleaning pipeline from the repository root once ingestion has finished:

```bash
PYTHONPATH=src python -m rdu_temperature.pipeline.clean_weather
```

Existing outputs are protected unless `--overwrite` is supplied. The run writes
three files beneath `data/processed/`:

- `hourly_panel.parquet`: every source and station on one hourly grid, in a
  single column vocabulary and a single set of units
- `rdu_hourly_target.parquet`: the hourly RDU temperature the project predicts
- `screening_report.csv`: what quality screening masked, by rule and station

Cleaning runs in five stages. Sources are normalized onto the canonical schema
in `schema.py`, aligned to the hourly grid, screened for implausible readings,
and coalesced into the target series, which is then written with the panel.

### Decisions worth knowing

- **Observation times are floored.** The 23:51 report describes hour 23. Where
  a station reports more than once an hour, the last non-missing reading of
  each variable wins. Averaging was rejected because wind direction does not
  average linearly.
- **The target is a measurement, never an estimate.** RDU temperature comes
  from GHCNh, with IEM filling the hours GHCNh omits. The two publish the same
  observation and agree to 0.028 degrees Celsius. GHCNh supplies 43,650 hours
  and IEM a further 151, leaving 23 of 43,824 hours uncovered. Those stay
  missing rather than being interpolated.
- **Timestamps are indexed on UTC and reported in both.** That keeps the
  repeated wall clock hour at the autumn transition distinct, while the local
  column carries the clock hours the project is scored on.
- **Screening masks readings, not rows.** One bad variable never discards the
  rest of an observation. Across the window this masks 11 readings out of
  7,315,298, and no temperature.
- **KJNX reports whole degrees Celsius** where every other station reports
  tenths, so it holds one value for up to 21 consecutive hours. Those runs are
  quantization rather than a stuck sensor, which is why the flatline threshold
  is a deliberately loose 24 hours.

## Modeling approach

To be completed.

## Evaluation

To be completed.

## Results

To be completed.
