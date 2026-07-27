from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.estr_policy_spread import (
    build_estr_policy_spread,
    load_rate_csv,
    save_policy_spread,
    summarise_estr_policy_spread,
)


ESTR_INPUT_PATH = Path(
    "data/raw/ecb/estr_rate.csv"
)

DEPOSIT_FACILITY_INPUT_PATH = Path(
    "data/raw/ecb/deposit_facility_rate.csv"
)

OUTPUT_PATH = Path(
    "data/processed/funding/estr_policy_spread.csv"
)


def format_optional_number(
    value: float,
    decimals: int = 3,
) -> str:
    """
    Format a value while handling insufficient history.
    """
    if pd.isna(value):
        return "Insufficient history"

    return f"{value:.{decimals}f}"


def main() -> None:
    """
    Build and save the €STR policy transmission dataset.
    """
    estr_data = load_rate_csv(
        input_path=ESTR_INPUT_PATH
    )

    deposit_facility_data = load_rate_csv(
        input_path=DEPOSIT_FACILITY_INPUT_PATH
    )

    spread_data = build_estr_policy_spread(
        estr_data=estr_data,
        policy_rate_data=deposit_facility_data,
    )

    saved_path = save_policy_spread(
        spread_data=spread_data,
        output_path=OUTPUT_PATH,
    )

    summary = summarise_estr_policy_spread(
        spread_data=spread_data
    )

    unusual_observations = int(
        spread_data[
            "is_unusual_spread"
        ].sum()
    )

    policy_change_observations = int(
        spread_data[
            "is_policy_change_day"
        ].sum()
    )

    print()
    print("REPOLENS EURO POLICY TRANSMISSION")
    print("=" * 68)
    print(
        "Latest observation: "
        f"{summary.latest_observation_date.date()}"
    )
    print(
        "Latest €STR: "
        f"{summary.latest_estr_rate:.3f}%"
    )
    print(
        "Deposit facility rate: "
        f"{summary.latest_deposit_facility_rate:.3f}%"
    )
    print(
        "€STR minus deposit facility rate: "
        f"{summary.latest_spread_bp:.3f} bp"
    )
    print(
        "60-day mean spread: "
        f"{format_optional_number(summary.rolling_mean_spread_bp)} bp"
    )
    print(
        "60-day spread volatility: "
        f"{format_optional_number(summary.rolling_volatility_bp)} bp"
    )
    print(
        "Latest spread z-score: "
        f"{format_optional_number(summary.latest_spread_z_score, 2)}"
    )
    print(
        "Transmission regime: "
        f"{summary.transmission_regime}"
    )
    print(
        "Latest date is a policy change date: "
        f"{summary.is_policy_change_day}"
    )
    print(
        "Latest spread is unusual: "
        f"{summary.is_unusual_spread}"
    )
    print(
        "Historical unusual observations: "
        f"{unusual_observations:,}"
    )
    print(
        "Matched policy change observations: "
        f"{policy_change_observations:,}"
    )
    print(
        "Saved to: "
        f"{saved_path}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()