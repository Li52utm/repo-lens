from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.morning_sheet import (
    build_morning_sheet,
    load_processed_csv,
    save_morning_sheet,
    summarise_morning_sheet,
)


PROCESSED_FUNDING_DIRECTORY = Path(
    "data/processed/funding"
)

FUNDING_CONDITIONS_PATH = (
    PROCESSED_FUNDING_DIRECTORY
    / "estr_funding_conditions.csv"
)

POLICY_SPREAD_PATH = (
    PROCESSED_FUNDING_DIRECTORY
    / "estr_policy_spread.csv"
)

MARKET_QUALITY_PATH = (
    PROCESSED_FUNDING_DIRECTORY
    / "estr_market_quality.csv"
)

OUTPUT_PATH = Path(
    "outputs/morning_sheet/repolens_morning_sheet.csv"
)


def format_optional_number(
    value: float,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """
    Format a number while safely handling missing history.
    """
    if pd.isna(
        value
    ):
        return "Insufficient history"

    return f"{value:.{decimals}f}{suffix}"


def main() -> None:
    """
    Build and save the consolidated RepoLens Morning Sheet.
    """
    funding_conditions = load_processed_csv(
        FUNDING_CONDITIONS_PATH
    )

    policy_spread = load_processed_csv(
        POLICY_SPREAD_PATH
    )

    market_quality = load_processed_csv(
        MARKET_QUALITY_PATH
    )

    morning_sheet = build_morning_sheet(
        funding_conditions=funding_conditions,
        policy_spread=policy_spread,
        market_quality=market_quality,
    )

    saved_path = save_morning_sheet(
        morning_sheet=morning_sheet,
        output_path=OUTPUT_PATH,
    )

    summary = summarise_morning_sheet(
        morning_sheet=morning_sheet
    )

    print()
    print("REPOLENS EURO FUNDING MORNING SHEET")
    print("=" * 78)
    print(
        "Observation date: "
        f"{summary.observation_date.date()}"
    )
    print(
        "Overall status: "
        f"{summary.overall_status}"
    )
    print(
        "Active alerts: "
        f"{summary.alert_count}"
    )
    print("-" * 78)
    print(
        "€STR: "
        f"{summary.estr_rate:.3f}%"
    )
    print(
        "Daily move: "
        f"{format_optional_number(summary.daily_change_bp, 3, ' bp')}"
    )
    print(
        "Funding regime: "
        f"{summary.funding_regime}"
    )
    print("-" * 78)
    print(
        "ECB deposit facility rate: "
        f"{summary.deposit_facility_rate:.3f}%"
    )
    print(
        "€STR policy spread: "
        f"{summary.policy_spread_bp:.3f} bp"
    )
    print(
        "Policy-spread z-score: "
        f"{format_optional_number(summary.policy_spread_z_score, 2)}"
    )
    print(
        "Transmission regime: "
        f"{summary.transmission_regime}"
    )
    print(
        "Policy rate changed today: "
        f"{summary.policy_rate_changed_today}"
    )
    print("-" * 78)
    print(
        "Market-quality score: "
        f"{format_optional_number(summary.market_quality_score, 2)} / 100"
    )
    print(
        "Daily quality-score change: "
        f"{format_optional_number(summary.market_quality_change, 2)}"
    )
    print(
        "Quality regime: "
        f"{summary.quality_regime}"
    )
    print(
        "Eligible volume: "
        f"€{summary.total_volume_eur_mn:,.0f} million"
    )
    print(
        "Daily volume change: "
        f"{format_optional_number(summary.volume_change_pct, 2, '%')}"
    )
    print(
        "Active banks: "
        f"{summary.active_banks:,.0f}"
    )
    print(
        "Eligible transactions: "
        f"{summary.transaction_count:,.0f}"
    )
    print(
        "Rate dispersion: "
        f"{summary.rate_dispersion_bp:.3f} bp"
    )
    print("-" * 78)
    print(
        "Classification: "
        "Official ECB inputs; RepoLens derived analytics"
    )
    print(
        "Saved to: "
        f"{saved_path}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()