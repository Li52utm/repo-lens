from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.download_estr_market_quality import (
    download_market_quality_series,
    download_series,
)
from src.ecb_series import (
    ECBSeriesDefinition,
    ESTR_MARKET_QUALITY_SERIES,
)


class FakeECBClient:
    """
    Deterministic ECB client used by downloader tests.
    """

    def __init__(self) -> None:
        self.requested_keys: list[str] = []

    def fetch_series(
        self,
        request,
    ):
        self.requested_keys.append(
            request.series_key
        )

        observations = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    [
                        "2026-07-23",
                        "2026-07-24",
                    ]
                ),
                "value": [
                    100.0,
                    101.0,
                ],
            }
        )

        return SimpleNamespace(
            observations=observations,
        )

    @staticmethod
    def save_csv(
        result,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.observations.to_csv(
            output_path,
            index=False,
        )

        return output_path


def test_market_quality_registry_has_unique_keys() -> None:
    keys = [
        definition.series_key
        for definition in ESTR_MARKET_QUALITY_SERIES
    ]

    assert len(keys) == 5
    assert len(set(keys)) == len(keys)


def test_market_quality_registry_has_unique_outputs() -> None:
    output_names = [
        definition.output_filename
        for definition in ESTR_MARKET_QUALITY_SERIES
    ]

    assert len(set(output_names)) == len(
        output_names
    )


def test_market_quality_registry_contains_verified_keys() -> None:
    keys = {
        definition.series_key
        for definition in ESTR_MARKET_QUALITY_SERIES
    }

    assert keys == {
        "B.EU000A2X2A25.TT",
        "B.EU000A2X2A25.R25",
        "B.EU000A2X2A25.R75",
        "B.EU000A2X2A25.NB",
        "B.EU000A2X2A25.NT",
    }


def test_download_series_saves_expected_file(
    tmp_path: Path,
) -> None:
    client = FakeECBClient()

    definition = ECBSeriesDefinition(
        name="Test series",
        dataflow="EST",
        series_key="B.TEST.KEY",
        unit="Count",
        classification="Official",
        description="Test description.",
        output_filename="test_series.csv",
    )

    record = download_series(
        client=client,
        definition=definition,
        output_directory=tmp_path,
    )

    assert record.output_path == (
        tmp_path
        / "test_series.csv"
    )

    assert record.output_path.exists()
    assert record.observation_count == 2

    assert record.latest_value == pytest.approx(
        101.0
    )

    assert client.requested_keys == [
        "B.TEST.KEY"
    ]


def test_download_all_market_quality_series(
    tmp_path: Path,
) -> None:
    client = FakeECBClient()

    records = download_market_quality_series(
        client=client,
        output_directory=tmp_path,
    )

    assert len(records) == 5

    expected_keys = [
        definition.series_key
        for definition in ESTR_MARKET_QUALITY_SERIES
    ]

    assert client.requested_keys == expected_keys

    for definition in ESTR_MARKET_QUALITY_SERIES:
        assert (
            tmp_path
            / definition.output_filename
        ).exists()