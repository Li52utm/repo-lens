from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.bundesbank_client import (
    BundesbankClient,
    combine_benchmark_series,
)
from src.sovereign_market_data import (
    GERMAN_BENCHMARKS,
)


DEFAULT_START_PERIOD = date(
    2015,
    1,
    1,
)

OUTPUT_PATH = Path(
    "data/raw/sovereign/germany_benchmark_yields.csv"
)


def download_german_benchmark_yields(
    client: BundesbankClient | None = None,
    start_period: date = DEFAULT_START_PERIOD,
    end_period: date | None = None,
    as_of_date: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Download Germany's official benchmark sovereign yield history.
    """
    effective_client = (
        client
        if client is not None
        else BundesbankClient()
    )

    benchmark_frames = tuple(
        effective_client.fetch_benchmark(
            definition=definition,
            start_period=start_period,
            end_period=end_period,
            as_of_date=as_of_date,
        )
        for definition in GERMAN_BENCHMARKS
    )

    combined = combine_benchmark_series(
        benchmark_frames
    )

    downloaded_tenors = set(
        combined[
            "tenor_years"
        ].unique()
    )

    expected_tenors = {
        definition.tenor_years
        for definition in GERMAN_BENCHMARKS
    }

    if downloaded_tenors != expected_tenors:
        raise RuntimeError(
            "Downloaded German benchmark tenors do not match "
            f"the approved registry. Expected {sorted(expected_tenors)}, "
            f"received {sorted(downloaded_tenors)}."
        )

    return combined


def latest_curve_snapshot(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the latest available observation for each German tenor.
    """
    ordered = observations.sort_values(
        [
            "tenor_years",
            "observation_date",
        ]
    )

    latest = ordered.groupby(
        "tenor_years",
        as_index=False,
    ).tail(
        1
    )

    return latest.sort_values(
        "tenor_years"
    ).reset_index(
        drop=True
    )


def main() -> None:
    """
    Download, save and display Germany's official benchmark curve.
    """
    client = BundesbankClient()

    observations = download_german_benchmark_yields(
        client=client
    )

    saved_path = client.save_csv(
        observations=observations,
        output_path=OUTPUT_PATH,
    )

    latest = latest_curve_snapshot(
        observations
    )

    print()
    print("REPOLENS GERMAN SOVEREIGN BENCHMARK CURVE")
    print("=" * 84)
    print(
        "Source: Deutsche Bundesbank official daily statistics"
    )
    print(
        "Status: OFFICIAL_DAILY unless marked STALE"
    )
    print("-" * 84)

    for row in latest.itertuples(
        index=False
    ):
        print(
            f"{int(row.tenor_years):>2}Y  "
            f"{float(row.yield_percent):>8.3f}%  "
            f"{pd.Timestamp(row.observation_date).date()}  "
            f"{row.data_status:<14}  "
            f"{int(row.business_days_stale)} business day(s) old"
        )

    print("-" * 84)
    print(
        f"Rows downloaded: {len(observations):,}"
    )
    print(
        f"Saved to: {saved_path}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()