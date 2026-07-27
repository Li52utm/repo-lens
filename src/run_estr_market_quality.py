from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.estr_market_quality import (
    build_estr_market_quality,
    load_market_quality_csv,
    save_estr_market_quality,
    summarise_estr_market_quality,
)


RAW_DIRECTORY = Path(
    "data/raw/ecb"
)

OUTPUT_PATH = Path(
    "data/processed/funding/estr_market_quality.csv"
)


def format_optional_number(
    value: float,
    decimals: int = 2,
) -> str:
    """
    Format a number while handling insufficient history.
    """
    if pd.isna(
        value
    ):
        return "Insufficient history"

    return f"{value:.{decimals}f}"


def main() -> None:
    """
    Build and save the RepoLens €STR market-quality dataset.
    """
    estr_rate_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_rate.csv"
    )

    total_volume_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_total_volume.csv"
    )

    percentile_25_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_rate_25th_percentile.csv"
    )

    percentile_75_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_rate_75th_percentile.csv"
    )

    active_banks_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_active_banks.csv"
    )

    transaction_count_data = load_market_quality_csv(
        RAW_DIRECTORY
        / "estr_transaction_count.csv"
    )

    market_quality = build_estr_market_quality(
        estr_rate_data=estr_rate_data,
        total_volume_data=total_volume_data,
        percentile_25_data=percentile_25_data,
        percentile_75_data=percentile_75_data,
        active_banks_data=active_banks_data,
        transaction_count_data=transaction_count_data,
    )

    saved_path = save_estr_market_quality(
        market_quality=market_quality,
        output_path=OUTPUT_PATH,
    )

    summary = summarise_estr_market_quality(
        market_quality=market_quality
    )

    alert_count = int(
        market_quality[
            "is_market_quality_alert"
        ].sum()
    )

    print()
    print("REPOLENS €STR MARKET QUALITY")
    print("=" * 72)
    print(
        "Latest observation: "
        f"{summary.latest_observation_date.date()}"
    )
    print(
        "Latest €STR: "
        f"{summary.latest_estr_rate:.3f}%"
    )
    print(
        "Eligible transaction volume: "
        f"€{summary.latest_total_volume_eur_mn:,.0f} million"
    )
    print(
        "Active banks: "
        f"{summary.latest_active_banks:,.0f}"
    )
    print(
        "Eligible transactions: "
        f"{summary.latest_transaction_count:,.0f}"
    )
    print(
        "25th to 75th percentile dispersion: "
        f"{summary.latest_rate_dispersion_bp:.3f} bp"
    )
    print(
        "RepoLens market-quality score: "
        f"{format_optional_number(summary.latest_quality_score)} / 100"
    )
    print(
        "Market-quality regime: "
        f"{summary.quality_regime}"
    )
    print(
        "Current alert: "
        f"{summary.is_market_quality_alert}"
    )
    print(
        "Available score components: "
        f"{summary.available_component_count} / 4"
    )
    print(
        "Data freshness: "
        f"{summary.freshness_status} "
        f"({summary.business_days_stale} business days old)"
    )
    print(
        "Historical alerts: "
        f"{alert_count:,}"
    )
    print(
        "Classification: "
        "Official ECB inputs; RepoLens derived score"
    )
    print(
        "Saved to: "
        f"{saved_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()