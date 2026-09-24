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

To be completed.

## Data pipeline

To be completed.

## Modeling approach

To be completed.

## Evaluation

To be completed.

## Results

To be completed.
