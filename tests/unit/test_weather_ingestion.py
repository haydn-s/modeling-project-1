from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest

from rdu_temperature.pipeline.ingest_weather import (
    DateWindow,
    IemAsosSource,
    NoaaGhcnhSource,
    OpenMeteoSource,
    WeatherSource,
)


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        payload: dict[str, Any] | list[dict[str, Any]] | None = None,
        content: bytes = b"",
    ) -> None:
        self.text = text
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any] | list[dict[str, Any]]:
        assert self._payload is not None
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


class ExampleSource(WeatherSource):
    source_name = "example"

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "station_id": ["B", "A", "A", "A"],
                "timestamp_utc": [
                    "2026-01-01T01:00:00Z",
                    "2025-12-31T23:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                ],
                "temperature": [2.0, -1.0, 1.0, 3.0],
            }
        )


def test_date_window_splits_on_calendar_boundaries() -> None:
    window = DateWindow.from_strings("2025-12-15T04:00:00Z", "2026-02-02T04:00:00Z")

    years = list(window.calendar_years())
    chunks = list(window.fixed_days(20))

    assert years == [
        (pd.Timestamp("2025-12-15T04:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
        (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-02-02T04:00:00Z")),
    ]
    assert len(chunks) == 3
    assert chunks[1] == (
        pd.Timestamp("2026-01-04T04:00:00Z"),
        pd.Timestamp("2026-01-24T04:00:00Z"),
    )


def test_shared_ingestion_filters_cutoff_sorts_and_writes_csv(tmp_path) -> None:
    window = DateWindow.from_strings("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    source = ExampleSource(window, tmp_path)

    output_path = source.ingest()
    output = pd.read_csv(output_path)

    assert output["source"].tolist() == ["example", "example"]
    assert output["station_id"].tolist() == ["A", "B"]
    assert output["temperature"].tolist() == [1.0, 2.0]
    with pytest.raises(FileExistsError):
        source.ingest()


def test_iem_uses_exact_utc_interval_and_parses_csv(tmp_path) -> None:
    response = FakeResponse(
        text=(
            "station,valid,tmpf,lat,lon,elevation\n"
            "RDU,2026-09-16 23:51,70.0,35.8922,-78.7819,120.0\n"
        )
    )
    session = FakeSession([response])
    window = DateWindow.from_strings("2026-09-16T04:00:00Z", "2026-09-17T04:00:00Z")
    source = IemAsosSource(
        window,
        tmp_path,
        session=session,
        stations=[{"iem_id": "RDU", "name": "RDU"}],
    )

    frame = source.fetch()
    params = dict(session.requests[0]["params"])

    assert frame.loc[0, "station_id"] == "RDU"
    assert frame.loc[0, "timestamp_utc"] == pd.Timestamp("2026-09-16T23:51:00Z")
    assert params["sts"] == "2026-09-16T04:00:00Z"
    assert params["ets"] == "2026-09-17T04:00:00Z"
    assert params["report_type"] == "3"


def test_noaa_keeps_routine_hourly_reports_and_configured_variables(tmp_path) -> None:
    parquet_buffer = io.BytesIO()
    pd.DataFrame(
        {
            "DATE": [
                "2026-09-16T04:51:00",
                "2026-09-16T05:00:00",
                "2026-09-16T05:51:00",
            ],
            "STATION": ["USW00013722"] * 3,
            "NAME": ["RALEIGH AP"] * 3,
            "temperature": [20.0, 20.0, 21.0],
            "temperature_Report_Type": ["FM15", "FM12", "FM15"],
            "temperature_Quality_Code": [None, None, None],
            "dew_point_temperature": [15.0, 15.0, 16.0],
            "unused_variable": [1, 2, 3],
        }
    ).to_parquet(parquet_buffer, index=False)
    session = FakeSession([FakeResponse(content=parquet_buffer.getvalue())])
    station = {
        "ghcnh_id": "USW00013722",
        "name": "RDU",
        "icao": "KRDU",
    }
    source = NoaaGhcnhSource(
        DateWindow.from_strings("2026-09-16T04:00:00Z", "2026-09-17T04:00:00Z"),
        tmp_path,
        session=session,
        stations=[station],
        variables=["temperature", "dew_point_temperature"],
    )

    frame = source.fetch()

    assert frame["temperature"].tolist() == [20.0, 21.0]
    assert frame["temperature_Report_Type"].tolist() == ["FM15", "FM15"]
    assert "dew_point_temperature" in frame
    assert "unused_variable" not in frame


def test_open_meteo_maps_each_grid_response_to_its_station(tmp_path) -> None:
    hourly = {
        "time": ["2026-09-16T04:00"],
        "temperature_2m": [20.0],
    }
    response = FakeResponse(
        payload=[
            {"latitude": 35.9, "longitude": -78.8, "elevation": 120, "hourly": hourly},
            {"latitude": 35.6, "longitude": -79.1, "elevation": 73, "hourly": hourly},
        ]
    )
    session = FakeSession([response])
    stations = [
        {
            "ghcnh_id": "USW00013722",
            "name": "RDU",
            "latitude": 35.8922,
            "longitude": -78.7819,
        },
        {
            "ghcnh_id": "USW00003723",
            "name": "TTA",
            "latitude": 35.5764,
            "longitude": -79.1034,
        },
    ]
    source = OpenMeteoSource(
        DateWindow.from_strings("2026-09-16T04:00:00Z", "2026-09-17T04:00:00Z"),
        tmp_path,
        session=session,
        stations=stations,
        variables=["temperature_2m"],
        model="era5_seamless",
    )

    frame = source.fetch()

    assert frame["station_id"].tolist() == ["USW00013722", "USW00003723"]
    assert frame["model"].unique().tolist() == ["era5_seamless"]
    assert session.requests[0]["params"]["models"] == "era5_seamless"
