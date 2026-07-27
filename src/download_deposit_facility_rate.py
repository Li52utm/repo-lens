from __future__ import annotations

from pathlib import Path

from src.ecb_client import (
    ECBClient,
    ECBSeriesRequest,
)


DATAFLOW = "FM"
SERIES_KEY = "B.U2.EUR.4F.KR.DFR.LEV"

OUTPUT_PATH = Path(
    "data/raw/ecb/deposit_facility_rate.csv"
)


def main() -> None:
    """
    Download the official ECB deposit facility rate.
    """
    client = ECBClient()

    request = ECBSeriesRequest(
        dataflow=DATAFLOW,
        series_key=SERIES_KEY,
    )

    result = client.fetch_series(
        request=request,
    )

    saved_path = client.save_csv(
        result=result,
        output_path=OUTPUT_PATH,
    )

    latest = result.observations.iloc[-1]

    print()
    print("REPOLENS ECB DEPOSIT FACILITY RATE")
    print("=" * 64)
    print("Series: ECB deposit facility rate")
    print("Classification: Official")
    print("Unit: Percent per annum")
    print(
        f"Observations: "
        f"{len(result.observations):,}"
    )
    print(
        f"First observation: "
        f"{result.observations['observation_date'].min().date()}"
    )
    print(
        f"Latest observation: "
        f"{latest['observation_date'].date()}"
    )
    print(
        f"Latest value: "
        f"{latest['value']:.3f}%"
    )
    print(
        f"Source: "
        f"{result.source_url}"
    )
    print(
        f"Saved to: "
        f"{saved_path}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()