from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src.bundesbank_client import (
    BundesbankClient,
    BundesbankRequestError,
    BundesbankResponseError,
    BundesbankSeriesRequest,
    combine_benchmark_series,
)
from src.sovereign_market_data import (
    GERMANY_10Y,
)


SAMPLE_SDMX_CSV = """\
FREQ,BBK_STD_MATURITY,SECURITY_TYPE,UNIT,SECURITY_CATEGORY,VALUATION,OBS_STATUS,TIME_PERIOD,OBS_VALUE
D,REN,EUR,A630,000000WT1010,A,A,2026-07-23,2.810
D,REN,EUR,A630,000000WT1010,A,A,2026-07-24,2.850
"""


class FakeResponse:
    """
    Minimal deterministic HTTP response for client tests.
    """

    def __init__(
        self,
        status_code: int,
        text: str,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url


class FakeSession:
    """
    Minimal requests-compatible session.
    """

    def __init__(
        self,
        response: FakeResponse | None = None,
        error: requests.RequestException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[
            dict[str, object]
        ] = []

    def get(
        self,
        url: str,
        params: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
            }
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError(
                "FakeSession has no configured response."
            )

        return self.response


def test_request_rejects_invalid_date_range() -> None:
    with pytest.raises(
        ValueError,
        match="start_period",
    ):
        BundesbankSeriesRequest(
            dataflow="BBSSY",
            series_key="D.TEST",
            start_period=date(
                2026,
                7,
                28,
            ),
            end_period=date(
                2026,
                7,
                27,
            ),
        )


def test_build_url_and_params() -> None:
    client = BundesbankClient(
        base_url="https://example.test/rest/data",
        session=FakeSession(),
    )

    request = BundesbankSeriesRequest(
        dataflow="BBSSY",
        series_key="D.TEST",
        start_period=date(
            2025,
            1,
            1,
        ),
        end_period=date(
            2026,
            7,
            24,
        ),
    )

    assert client.build_url(
        request
    ) == (
        "https://example.test/rest/data/"
        "BBSSY/D.TEST"
    )

    assert client.build_params(
        request
    ) == {
        "format": "sdmx_csv",
        "lang": "en",
        "startPeriod": "2025-01-01",
        "endPeriod": "2026-07-24",
    }


def test_parse_sdmx_csv() -> None:
    observations = BundesbankClient.parse_sdmx_csv(
        SAMPLE_SDMX_CSV
    )

    assert list(
        observations.columns
    ) == [
        "observation_date",
        "value",
    ]

    assert len(
        observations
    ) == 2

    assert observations.iloc[-1][
        "value"
    ] == pytest.approx(
        2.85
    )


def test_empty_response_is_rejected() -> None:
    with pytest.raises(
        BundesbankResponseError,
        match="empty response",
    ):
        BundesbankClient.parse_sdmx_csv(
            ""
        )


def test_response_missing_observation_columns_is_rejected() -> None:
    with pytest.raises(
        BundesbankResponseError,
        match="missing required",
    ):
        BundesbankClient.parse_sdmx_csv(
            "DATE,RATE\n2026-07-24,2.85\n"
        )


def test_fetch_series_uses_expected_endpoint() -> None:
    response = FakeResponse(
        status_code=200,
        text=SAMPLE_SDMX_CSV,
        url=(
            "https://example.test/rest/data/"
            "BBSSY/D.TEST?format=sdmx_csv"
        ),
    )

    session = FakeSession(
        response=response
    )

    client = BundesbankClient(
        base_url="https://example.test/rest/data",
        session=session,
    )

    result = client.fetch_series(
        BundesbankSeriesRequest(
            dataflow="BBSSY",
            series_key="D.TEST",
        )
    )

    assert len(
        result.observations
    ) == 2

    assert session.calls[0][
        "url"
    ] == (
        "https://example.test/rest/data/"
        "BBSSY/D.TEST"
    )


def test_http_error_is_rejected() -> None:
    session = FakeSession(
        response=FakeResponse(
            status_code=404,
            text="No matching data",
            url="https://example.test/not-found",
        )
    )

    client = BundesbankClient(
        session=session
    )

    with pytest.raises(
        BundesbankRequestError,
        match="HTTP 404",
    ):
        client.fetch_series(
            BundesbankSeriesRequest(
                dataflow="BBSSY",
                series_key="D.TEST",
            )
        )


def test_network_error_is_wrapped() -> None:
    session = FakeSession(
        error=requests.ConnectionError(
            "Network unavailable"
        )
    )

    client = BundesbankClient(
        session=session
    )

    with pytest.raises(
        BundesbankRequestError,
        match="before a response",
    ):
        client.fetch_series(
            BundesbankSeriesRequest(
                dataflow="BBSSY",
                series_key="D.TEST",
            )
        )


def test_fetch_benchmark_builds_standard_contract() -> None:
    response = FakeResponse(
        status_code=200,
        text=SAMPLE_SDMX_CSV,
        url="https://example.test/10y",
    )

    client = BundesbankClient(
        base_url="https://example.test/rest/data",
        session=FakeSession(
            response=response
        ),
    )

    benchmark = client.fetch_benchmark(
        definition=GERMANY_10Y,
        as_of_date=date(
            2026,
            7,
            27,
        ),
    )

    assert {
        "country",
        "tenor_years",
        "yield_percent",
        "source_series",
        "data_status",
        "business_days_stale",
    }.issubset(
        benchmark.columns
    )

    latest = benchmark.iloc[-1]

    assert latest[
        "country"
    ] == "Germany"

    assert latest[
        "tenor_years"
    ] == 10

    assert latest[
        "yield_percent"
    ] == pytest.approx(
        2.85
    )

    assert latest[
        "business_days_stale"
    ] == 1

    assert latest[
        "data_status"
    ] == "OFFICIAL_DAILY"


def test_save_csv(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        status_code=200,
        text=SAMPLE_SDMX_CSV,
        url="https://example.test/10y",
    )

    client = BundesbankClient(
        session=FakeSession(
            response=response
        )
    )

    benchmark = client.fetch_benchmark(
        definition=GERMANY_10Y,
        as_of_date=date(
            2026,
            7,
            27,
        ),
    )

    output_path = (
        tmp_path
        / "german_yields.csv"
    )

    saved_path = client.save_csv(
        observations=benchmark,
        output_path=output_path,
    )

    assert saved_path.exists()

    saved_data = pd.read_csv(
        saved_path
    )

    assert len(
        saved_data
    ) == 2


def test_combine_benchmark_series() -> None:
    base = pd.DataFrame(
        {
            "observation_date": [
                "2026-07-24",
            ],
            "country": [
                "Germany",
            ],
            "country_code": [
                "DE",
            ],
            "tenor_years": [
                2,
            ],
            "benchmark_name": [
                "2Y",
            ],
            "yield_percent": [
                2.1,
            ],
            "source_name": [
                "Deutsche Bundesbank",
            ],
            "source_series": [
                "BBSSY.D.TEST2",
            ],
            "source_timestamp": [
                "2026-07-27T10:00:00Z",
            ],
            "data_status": [
                "OFFICIAL_DAILY",
            ],
            "business_days_stale": [
                1,
            ],
        }
    )

    second = base.copy()

    second["tenor_years"] = 10
    second["benchmark_name"] = "10Y"
    second["yield_percent"] = 2.85
    second["source_series"] = "BBSSY.D.TEST10"

    combined = combine_benchmark_series(
        (
            base,
            second,
        )
    )

    assert len(
        combined
    ) == 2

    assert set(
        combined[
            "tenor_years"
        ]
    ) == {
        2,
        10,
    }