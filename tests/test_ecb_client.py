from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.ecb_client import (
    ECBClient,
    ECBDataValidationError,
    ECBSeriesRequest,
)


VALID_CSV = """\
KEY,FREQ,REF_AREA,BENCHMARK_ITEM,DATA_TYPE_EST,TIME_PERIOD,OBS_VALUE
EST.B.EU000A2X2A25.WT,B,U2,EU000A2X2A25,WT,2026-07-23,2.100
EST.B.EU000A2X2A25.WT,B,U2,EU000A2X2A25,WT,2026-07-24,2.101
"""


def test_series_request_rejects_empty_dataflow() -> None:
    with pytest.raises(
        ValueError,
        match="dataflow",
    ):
        ECBSeriesRequest(
            dataflow="",
            series_key="B.EU000A2X2A25.WT",
        )


def test_series_request_rejects_empty_key() -> None:
    with pytest.raises(
        ValueError,
        match="series_key",
    ):
        ECBSeriesRequest(
            dataflow="EST",
            series_key="",
        )


def test_series_request_rejects_reversed_dates() -> None:
    with pytest.raises(
        ValueError,
        match="start_period",
    ):
        ECBSeriesRequest(
            dataflow="EST",
            series_key="B.EU000A2X2A25.WT",
            start_period=date(
                2026,
                7,
                25,
            ),
            end_period=date(
                2026,
                7,
                24,
            ),
        )


def test_series_request_rejects_invalid_last_n() -> None:
    with pytest.raises(
        ValueError,
        match="last_n_observations",
    ):
        ECBSeriesRequest(
            dataflow="EST",
            series_key="B.EU000A2X2A25.WT",
            last_n_observations=0,
        )


def test_build_url() -> None:
    client = ECBClient()

    request = ECBSeriesRequest(
        dataflow="EST",
        series_key="B.EU000A2X2A25.WT",
    )

    assert client.build_url(request) == (
        "https://data-api.ecb.europa.eu/"
        "service/data/EST/"
        "B.EU000A2X2A25.WT"
    )


def test_build_parameters() -> None:
    client = ECBClient()

    request = ECBSeriesRequest(
        dataflow="EST",
        series_key="B.EU000A2X2A25.WT",
        start_period=date(
            2026,
            1,
            1,
        ),
        end_period=date(
            2026,
            7,
            24,
        ),
        last_n_observations=20,
    )

    assert client.build_parameters(
        request
    ) == {
        "format": "csvdata",
        "detail": "full",
        "startPeriod": "2026-01-01",
        "endPeriod": "2026-07-24",
        "lastNObservations": 20,
    }


def test_parse_csv_response() -> None:
    observations = ECBClient.parse_csv_response(
        VALID_CSV
    )

    assert len(observations) == 2

    assert list(
        observations.columns[:2]
    ) == [
        "observation_date",
        "value",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        observations["observation_date"]
    )

    assert observations.iloc[-1]["value"] == pytest.approx(
        2.101
    )


def test_parse_csv_removes_invalid_rows() -> None:
    csv_text = """\
TIME_PERIOD,OBS_VALUE
2026-07-23,2.100
not-a-date,2.200
2026-07-24,not-a-number
"""

    observations = ECBClient.parse_csv_response(
        csv_text
    )

    assert len(observations) == 1
    assert observations.iloc[0]["value"] == pytest.approx(
        2.100
    )


def test_parse_csv_rejects_empty_response() -> None:
    with pytest.raises(
        ECBDataValidationError,
        match="empty",
    ):
        ECBClient.parse_csv_response("")


def test_parse_csv_rejects_missing_columns() -> None:
    with pytest.raises(
        ECBDataValidationError,
        match="missing required columns",
    ):
        ECBClient.parse_csv_response(
            "DATE,RATE\n2026-07-24,2.1\n"
        )


def test_save_csv(
    tmp_path,
) -> None:
    observations = ECBClient.parse_csv_response(
        VALID_CSV
    )

    from src.ecb_client import ECBSeriesResult

    result = ECBSeriesResult(
        dataflow="EST",
        series_key="B.EU000A2X2A25.WT",
        source_url="https://example.com",
        retrieved_at_utc=pd.Timestamp.now(
            tz="UTC"
        ),
        observations=observations,
    )

    output_path = (
        tmp_path
        / "nested"
        / "estr.csv"
    )

    saved_path = ECBClient.save_csv(
        result=result,
        output_path=output_path,
    )

    assert saved_path.exists()

    saved_data = pd.read_csv(
        saved_path
    )

    assert len(saved_data) == 2