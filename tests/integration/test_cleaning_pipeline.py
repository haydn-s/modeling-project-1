"""End-to-end cleaning run over miniature copies of each raw source."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rdu_temperature.pipeline.clean_weather import WeatherCleaningApp

CONFIG = {
    "history": {
        "start_utc": "2026-01-01T00:00:00Z",
        "end_utc": "2026-01-01T04:00:00Z",
        "local_timezone": "America/New_York",
    },
    "airport_stations": [
        {
            "name": "Raleigh-Durham International Airport",
            "icao": "KRDU",
            "iem_id": "RDU",
            "ghcnh_id": "USW00013722",
            "latitude": 35.8922,
            "longitude": -78.7819,
        }
    ],
    "econet_stations": [
        {"id": "REED", "name": "Reedy Creek", "latitude": 35.8, "longitude": -78.7}
    ],
}

HOURS = pd.date_range("2026-01-01T00:00Z", periods=4, freq="h")


def _write_sources(raw_dir: Path) -> None:
    # GHCNh reports at :51. The third hour is missing and the fourth is
    # flagged suspect, so both must fall through to IEM.
    ghcnh = pd.DataFrame(
        {
            "station_id": ["USW00013722"] * 3,
            "timestamp_utc": pd.DatetimeIndex([HOURS[0], HOURS[1], HOURS[3]])
            + pd.Timedelta(minutes=51),
            "temperature": [10.0, 11.0, 13.0],
            "temperature_Quality_Code": [5.0, 5.0, 2.0],
            "dew_point_temperature": [5.0, 6.0, 8.0],
            "wind_direction": [999.0, 180.0, 90.0],
            "wind_direction_Quality_Code": [9.0, 9.0, 9.0],
        }
    )
    iem = pd.DataFrame(
        {
            "station_id": ["RDU"] * 4,
            "timestamp_utc": HOURS + pd.Timedelta(minutes=51),
            "tmpf": [50.0, 51.8, 53.6, 55.4],
            "skyc1": ["CLR", "FEW", "OVC", "CLR"],
        }
    )
    open_meteo = pd.DataFrame(
        {
            "station_id": ["USW00013722"] * 4,
            "timestamp_utc": HOURS,
            "temperature_2m": [9.5, 10.5, 11.5, 12.5],
            "cloud_cover": [0.0, 20.0, 100.0, 0.0],
        }
    )
    econet = pd.DataFrame(
        {
            "station_id": ["REED"] * 4,
            "timestamp_utc": list(HOURS[:2]) * 2,
            "var": ["airtemp2m|C"] * 2 + ["dewtemp2m|C"] * 2,
            "value": ["9.0", "QCF", "4.0", "5.0"],
            "score": [0, 3, 0, 0],
        }
    )
    for name, frame in {
        "noaa_ghcnh": ghcnh,
        "iem_asos": iem,
        "open_meteo": open_meteo,
        "ncsco_econet": econet,
    }.items():
        directory = raw_dir / name
        directory.mkdir(parents=True)
        frame.to_csv(
            directory / "weather_20260101T000000Z_20260101T040000Z.csv", index=False
        )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "weather_sources.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    _write_sources(raw_dir)
    return config_path, raw_dir, tmp_path / "processed"


def test_cleaning_run_writes_a_panel_target_and_report(
    workspace: tuple[Path, Path, Path],
) -> None:
    config_path, raw_dir, output_dir = workspace

    outputs = WeatherCleaningApp(config_path, raw_dir, output_dir).run()

    assert all(path.exists() for path in outputs.values())
    panel = pd.read_parquet(outputs["panel"])
    target = pd.read_parquet(outputs["target"])

    # Four sources over the stations each one covers, across four hours.
    assert sorted(panel["source"].unique()) == [
        "iem_asos",
        "ncsco_econet",
        "noaa_ghcnh",
        "open_meteo",
    ]
    assert sorted(panel["station_id"].unique()) == ["KRDU", "REED"]
    assert len(target) == 4


def test_target_prefers_ghcnh_and_falls_through_to_iem(
    workspace: tuple[Path, Path, Path],
) -> None:
    config_path, raw_dir, output_dir = workspace

    outputs = WeatherCleaningApp(config_path, raw_dir, output_dir).run()
    target = pd.read_parquet(outputs["target"]).set_index("timestamp_utc")

    origins = target["temperature_source"].tolist()
    # Hours 0 and 1 come from GHCNh; hour 2 is absent there and hour 3 was
    # flagged suspect, so both fall through to IEM.
    assert origins == ["noaa_ghcnh", "noaa_ghcnh", "iem_asos", "iem_asos"]
    assert target["temperature_c"].tolist() == [10.0, 11.0, pytest.approx(12.0), 13.0]


def test_cleaning_run_records_local_time_and_protects_existing_output(
    workspace: tuple[Path, Path, Path],
) -> None:
    config_path, raw_dir, output_dir = workspace
    app = WeatherCleaningApp(config_path, raw_dir, output_dir)

    outputs = app.run()
    target = pd.read_parquet(outputs["target"])

    # January at RDU is standard time, five hours behind UTC.
    assert str(target["timestamp_local"].dt.tz) == "America/New_York"
    assert target["timestamp_local"].iloc[0].hour == 19
    with pytest.raises(FileExistsError):
        app.run()
    app.run(overwrite=True)
