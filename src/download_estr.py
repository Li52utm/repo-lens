from __future__ import annotations

from pathlib import Path

from src.ecb_client import (
    ECBClient,
    ECBSeriesRequest,
)
from src.ecb_series import ESTR_RATE


OUTPUT_PATH = Path(
    "data/raw/ecb/estr_rate.csv"
)


def main() -> None:
    """
    Download and save the official €STR time series.
    """
    client = ECBClient()

    request = ECBSeriesRequest(
        dataflow=ESTR_RATE.dataflow,
        series_key=ESTR_RATE.series_key,
    )

    result = client.fetch_series(
        request=request,
    )

    saved_path = client.save_csv(
        result=result,
        output_path=OUTPUT_PATH,
    )

    latest_observation = (
        result.observations.iloc[-1]
    )

    print()
    print("REPOLENS ECB DATA DOWNLOAD")
    print("=" * 60)
    print(f"Series: {ESTR_RATE.name}")
    print(f"Classification: {ESTR_RATE.classification}")
    print(f"Unit: {ESTR_RATE.unit}")
    print(
        "Observations: "
        f"{len(result.observations):,}"
    )
    print(
        "First observation: "
        f"{result.observations['observation_date'].min().date()}"
    )
    print(
        "Latest observation: "
        f"{latest_observation['observation_date'].date()}"
    )
    print(
        "Latest value: "
        f"{latest_observation['value']:.3f}%"
    )
    print(f"Source: {result.source_url}")
    print(f"Saved to: {saved_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()