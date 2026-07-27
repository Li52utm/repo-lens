from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ecb_client import (
    ECBClient,
    ECBSeriesRequest,
)
from src.ecb_series import (
    ECBSeriesDefinition,
    ESTR_MARKET_QUALITY_SERIES,
)


OUTPUT_DIRECTORY = Path(
    "data/raw/ecb"
)


@dataclass(frozen=True)
class DownloadRecord:
    """
    Record one completed ECB market-quality download.
    """

    series_name: str
    series_key: str
    output_path: Path
    observation_count: int
    first_observation_date: pd.Timestamp
    latest_observation_date: pd.Timestamp
    latest_value: float


def download_series(
    client: ECBClient,
    definition: ECBSeriesDefinition,
    output_directory: Path,
) -> DownloadRecord:
    """
    Download, validate and save one approved ECB series.
    """
    request = ECBSeriesRequest(
        dataflow=definition.dataflow,
        series_key=definition.series_key,
    )

    result = client.fetch_series(
        request=request,
    )

    output_path = (
        output_directory
        / definition.output_filename
    )

    saved_path = client.save_csv(
        result=result,
        output_path=output_path,
    )

    observations = result.observations

    latest = observations.iloc[-1]

    return DownloadRecord(
        series_name=definition.name,
        series_key=definition.series_key,
        output_path=saved_path,
        observation_count=len(observations),
        first_observation_date=pd.Timestamp(
            observations[
                "observation_date"
            ].min()
        ),
        latest_observation_date=pd.Timestamp(
            latest[
                "observation_date"
            ]
        ),
        latest_value=float(
            latest[
                "value"
            ]
        ),
    )


def download_market_quality_series(
    client: ECBClient | None = None,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> tuple[DownloadRecord, ...]:
    """
    Download all approved €STR market-quality series.
    """
    effective_client = (
        client
        if client is not None
        else ECBClient()
    )

    records = tuple(
        download_series(
            client=effective_client,
            definition=definition,
            output_directory=output_directory,
        )
        for definition in ESTR_MARKET_QUALITY_SERIES
    )

    expected_count = len(
        ESTR_MARKET_QUALITY_SERIES
    )

    if len(records) != expected_count:
        raise RuntimeError(
            "Not all €STR market-quality series "
            "were downloaded successfully."
        )

    return records


def main() -> None:
    """
    Download the official €STR market-quality dataset.
    """
    records = download_market_quality_series()

    print()
    print("REPOLENS €STR MARKET QUALITY DOWNLOAD")
    print("=" * 78)

    for record in records:
        print()
        print(f"Series: {record.series_name}")
        print(f"Key: {record.series_key}")
        print(
            "Observations: "
            f"{record.observation_count:,}"
        )
        print(
            "First observation: "
            f"{record.first_observation_date.date()}"
        )
        print(
            "Latest observation: "
            f"{record.latest_observation_date.date()}"
        )
        print(
            "Latest value: "
            f"{record.latest_value:,.3f}"
        )
        print(
            "Saved to: "
            f"{record.output_path}"
        )

    print()
    print(
        "SUCCESS: downloaded "
        f"{len(records)} official ECB series."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()