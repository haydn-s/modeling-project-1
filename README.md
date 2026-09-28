# RDU Hourly Temperature Forecasting

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

### Deferred: forecast covariates

Numerical weather prediction archives encode each run's initialization time in
its file path, so a run initialized before the cutoff can legitimately supply
predicted covariates for the forecast period. NOAA GFS (0.25 degree, 384-hour
lead, archived from 2021-01-01), NOAA GEFS, and ECMWF IFS and AIFS (360-hour
lead, archived from 2023-01-18) each cover the full target window from the
2026-09-16 12Z run. GFS is the strongest candidate because its archive aligns
with this project's 2021 ingestion window, which is what makes it possible to
train on matched forecast and observation pairs.

This is deferred. The current approach uses observations only.

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
