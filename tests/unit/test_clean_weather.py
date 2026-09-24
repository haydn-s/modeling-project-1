from __future__ import annotations

import pandas as pd
import pytest

from rdu_temperature.pipeline import schema
from rdu_temperature.pipeline.clean_weather import (
    EconetNormalizer,
    HourlyGrid,
    IemAsosNormalizer,
    NoaaGhcnhNormalizer,
    OpenMeteoNormalizer,
    StationRegistry,
)

CONFIG = {
    "airport_stations": [
        {"name": "RDU", "icao": "KRDU", "iem_id": "RDU", "ghcnh_id": "USW00013722"},
        {"name": "TTA", "icao": "KTTA", "iem_id": "TTA", "ghcnh_id": "USW00003723"},
    ],
    "econet_stations": [{"id": "REED", "name": "Reedy Creek"}],
}


@pytest.fixture
def registry() -> StationRegistry:
    return StationRegistry.from_config(CONFIG)


def test_registry_maps_each_source_alias_to_one_canonical_id(
    registry: StationRegistry,
) -> None:
    assert registry.resolve(pd.Series(["USW00013722"]), "ghcnh").iloc[0] == "KRDU"
    assert registry.resolve(pd.Series(["RDU"]), "iem").iloc[0] == "KRDU"
    assert registry.resolve(pd.Series(["REED"]), "econet").iloc[0] == "REED"
    assert registry.resolve(pd.Series(["NOPE"]), "iem").isna().all()


def test_ghcnh_masks_only_the_variable_its_quality_code_describes(
    registry: StationRegistry,
) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["USW00013722"] * 3,
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:51Z", "2026-01-01T01:51Z", "2026-01-01T02:51Z"],
                utc=True,
            ),
            "temperature": [10.0, 11.0, 12.0],
            "temperature_Quality_Code": ["5", "2", "A"],
            "dew_point_temperature": [5.0, 6.0, 7.0],
            "dew_point_temperature_Quality_Code": [None, None, None],
            "visibility": [16.093, 16.093, 16.093],
        }
    )

    result = NoaaGhcnhNormalizer(registry).normalize(frame)

    assert result[schema.TEMPERATURE_C].tolist()[0] == 10.0
    assert result[schema.TEMPERATURE_C].isna().tolist() == [False, True, True]
    assert result[schema.DEWPOINT_C].tolist() == [5.0, 6.0, 7.0]
    assert result[schema.VISIBILITY_M].tolist() == [16093.0] * 3
    assert result[schema.STATION_ID].unique().tolist() == ["KRDU"]


def test_iem_converts_units_and_takes_greatest_sky_layer(
    registry: StationRegistry,
) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["RDU", "TTA", "XXX"],
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:51Z", "2026-01-01T00:51Z", "2026-01-01T00:51Z"],
                utc=True,
            ),
            "tmpf": [32.0, 212.0, 50.0],
            "sknt": [1.0, 0.0, 0.0],
            "p01i": [1.0, 0.0, 0.0],
            "vsby": [1.0, 10.0, 10.0],
            "mslp": [1013.2, 1000.0, 1000.0],
            "skyc1": ["FEW", "CLR", "CLR"],
            "skyc2": ["OVC", None, None],
            "skyc3": [None, None, None],
            "skyc4": [None, None, None],
        }
    )

    result = IemAsosNormalizer(registry).normalize(frame)

    assert result[schema.STATION_ID].tolist() == ["KRDU", "KTTA"]
    assert result[schema.TEMPERATURE_C].tolist() == [0.0, 100.0]
    assert result[schema.WIND_SPEED_MS].iloc[0] == pytest.approx(0.514444, abs=1e-6)
    assert result[schema.PRECIPITATION_MM].iloc[0] == pytest.approx(25.4)
    assert result[schema.VISIBILITY_M].iloc[0] == pytest.approx(1609.344)
    # Layers are cumulative, so OVC above FEW means overcast.
    assert result[schema.CLOUD_COVER_PCT].tolist() == [100.0, 0.0]


def test_iem_unreadable_sky_layers_stay_missing(registry: StationRegistry) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["RDU"],
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:51Z"], utc=True),
            "tmpf": [50.0],
            "skyc1": ["///"],
            "skyc2": [None],
        }
    )

    result = IemAsosNormalizer(registry).normalize(frame)

    assert result[schema.CLOUD_COVER_PCT].isna().all()


def test_econet_pivots_long_rows_and_drops_failed_grades(
    registry: StationRegistry,
) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["REED"] * 4,
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:00Z"] * 2 + ["2026-01-01T01:00Z"] * 2, utc=True
            ),
            "var": ["airtemp2m|C", "dewtemp2m|C"] * 2,
            "value": ["12.5", "8.0", "QCF", "9.0"],
            "score": [0, 1, 3, 0],
        }
    )

    result = EconetNormalizer(registry).normalize(frame)

    assert list(result.columns) == [
        schema.SOURCE,
        schema.STATION_ID,
        schema.TIMESTAMP_UTC,
        schema.TEMPERATURE_C,
        schema.DEWPOINT_C,
    ]
    assert result[schema.TEMPERATURE_C].tolist()[0] == 12.5
    assert result[schema.TEMPERATURE_C].isna().tolist() == [False, True]
    assert result[schema.DEWPOINT_C].tolist() == [8.0, 9.0]


def test_open_meteo_renames_without_converting(registry: StationRegistry) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["USW00013722"],
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00Z"], utc=True),
            "temperature_2m": [20.0],
            "cloud_cover": [55.0],
            "shortwave_radiation": [0.0],
        }
    )

    result = OpenMeteoNormalizer(registry).normalize(frame)

    assert result[schema.TEMPERATURE_C].tolist() == [20.0]
    assert result[schema.CLOUD_COVER_PCT].tolist() == [55.0]
    assert result[schema.SOURCE].tolist() == ["open_meteo"]


def test_every_normalizer_emits_identity_columns_first(
    registry: StationRegistry,
) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["USW00013722"],
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00Z"], utc=True),
            "temperature_2m": [20.0],
        }
    )

    result = OpenMeteoNormalizer(registry).normalize(frame)

    assert tuple(result.columns[:3]) == schema.IDENTITY_COLUMNS


@pytest.fixture
def grid() -> HourlyGrid:
    return HourlyGrid.from_config(
        {
            "history": {
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-01-01T04:00:00Z",
            }
        }
    )


def test_grid_is_half_open(grid: HourlyGrid) -> None:
    assert grid.index.tolist() == [
        pd.Timestamp("2026-01-01T00:00Z"),
        pd.Timestamp("2026-01-01T01:00Z"),
        pd.Timestamp("2026-01-01T02:00Z"),
        pd.Timestamp("2026-01-01T03:00Z"),
    ]


def test_alignment_floors_the_hour_and_keeps_the_last_reading(
    grid: HourlyGrid,
) -> None:
    frame = pd.DataFrame(
        {
            schema.SOURCE: ["iem_asos"] * 3,
            schema.STATION_ID: ["KRDU"] * 3,
            schema.TIMESTAMP_UTC: pd.to_datetime(
                ["2026-01-01T00:15Z", "2026-01-01T00:35Z", "2026-01-01T00:55Z"],
                utc=True,
            ),
            schema.TEMPERATURE_C: [10.0, 11.0, 12.0],
            # The final report omits the dew point, so the earlier one stands.
            schema.DEWPOINT_C: [5.0, 6.0, None],
        }
    )

    aligned = grid.align(frame, "iem_asos")
    first = aligned.iloc[0]

    assert len(aligned) == 4
    assert first[schema.TIMESTAMP_UTC] == pd.Timestamp("2026-01-01T00:00Z")
    assert first[schema.TEMPERATURE_C] == 12.0
    assert first[schema.DEWPOINT_C] == 6.0
    assert aligned[schema.TEMPERATURE_C].isna().tolist() == [False, True, True, True]


def test_alignment_gives_every_station_every_hour(grid: HourlyGrid) -> None:
    frame = pd.DataFrame(
        {
            schema.SOURCE: ["iem_asos"] * 2,
            schema.STATION_ID: ["KRDU", "KTTA"],
            schema.TIMESTAMP_UTC: pd.to_datetime(
                ["2026-01-01T00:51Z", "2026-01-01T02:51Z"], utc=True
            ),
            schema.TEMPERATURE_C: [10.0, 20.0],
        }
    )

    aligned = grid.align(frame, "iem_asos")

    assert len(aligned) == 8
    assert aligned[schema.STATION_ID].value_counts().to_dict() == {"KRDU": 4, "KTTA": 4}
    assert aligned[schema.SOURCE].unique().tolist() == ["iem_asos"]
    rdu = aligned[aligned[schema.STATION_ID] == "KRDU"]
    assert rdu[schema.TEMPERATURE_C].tolist()[0] == 10.0
    assert rdu[schema.TEMPERATURE_C].isna().sum() == 3


def test_alignment_excludes_observations_outside_the_grid(grid: HourlyGrid) -> None:
    frame = pd.DataFrame(
        {
            schema.SOURCE: ["iem_asos"] * 2,
            schema.STATION_ID: ["KRDU"] * 2,
            schema.TIMESTAMP_UTC: pd.to_datetime(
                ["2026-01-01T01:51Z", "2026-01-01T04:51Z"], utc=True
            ),
            schema.TEMPERATURE_C: [10.0, 99.0],
        }
    )

    aligned = grid.align(frame, "iem_asos")

    assert len(aligned) == 4
    assert 99.0 not in aligned[schema.TEMPERATURE_C].dropna().tolist()
