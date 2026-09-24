from __future__ import annotations

import pandas as pd
import pytest

from rdu_temperature.pipeline import schema


def test_canonical_columns_are_unique_and_lead_with_identity() -> None:
    assert len(set(schema.CANONICAL_COLUMNS)) == len(schema.CANONICAL_COLUMNS)
    assert schema.CANONICAL_COLUMNS[:3] == schema.IDENTITY_COLUMNS
    assert set(schema.CANONICAL_COLUMNS) == set(
        schema.IDENTITY_COLUMNS + schema.MEASUREMENT_COLUMNS
    )


@pytest.mark.parametrize(
    ("fahrenheit", "celsius"),
    [(32.0, 0.0), (212.0, 100.0), (-40.0, -40.0), (70.0, 21.111111)],
)
def test_fahrenheit_to_celsius(fahrenheit: float, celsius: float) -> None:
    result = schema.fahrenheit_to_celsius(pd.Series([fahrenheit]))
    assert result.iloc[0] == pytest.approx(celsius, abs=1e-6)


def test_conversions_match_reference_values() -> None:
    def convert(function, value: float) -> float:
        return function(pd.Series([value])).iloc[0]

    assert convert(schema.knots_to_meters_per_second, 1.0) == pytest.approx(
        0.514444, abs=1e-6
    )
    assert convert(schema.inches_of_mercury_to_hectopascals, 29.9213) == pytest.approx(
        1013.25, abs=1e-2
    )
    assert convert(schema.inches_to_millimeters, 1.0) == 25.4
    assert convert(schema.miles_to_meters, 1.0) == 1609.344
    assert convert(schema.kilometers_to_meters, 16.093) == 16093.0


def test_conversions_propagate_missing_values() -> None:
    values = pd.Series([32.0, None, 212.0])

    result = schema.fahrenheit_to_celsius(values)

    assert result.isna().tolist() == [False, True, False]


def test_sky_cover_is_ordered_and_rejects_unknown_codes() -> None:
    codes = pd.Series(["CLR", "FEW", "SCT", "BKN", "OVC", "VV ", "XXX", None])

    result = schema.sky_cover_to_percent(codes)

    assert result.tolist()[:6] == [0.0, 18.75, 43.75, 75.0, 100.0, 100.0]
    assert result.isna().tolist()[6:] == [True, True]
    ordered = [schema.SKY_COVER_PCT[c] for c in ("CLR", "FEW", "SCT", "BKN", "OVC")]
    assert ordered == sorted(ordered)
