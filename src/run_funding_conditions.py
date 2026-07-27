from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.funding_conditions import (
    build_funding_conditions,
    load_estr_csv,
    save_funding_conditions,
    summarise_funding_conditions,
)


INPUT_PATH = Path(
    "data/raw/ecb/estr_rate.csv"
)

OUTPUT_PATH = Path(
    "data/processed/funding/estr_funding_conditions.csv"
)


def format_optional_number(
    value: float,
    decimals: int = 3,
) -> str:
    """
    Format a number while handling missing values clearly.
    """
    if pd.isna(
        value
    ):
        return "Insufficient history"

    return f"{value:.{decimals}f}"


def main() -> None:
    """
    Build and save the RepoLens €STR funding-condition dataset.
    """
    raw_data = load_estr_csv(
        input_path=INPUT_PATH
    )

    conditions = build_funding_conditions(
        data=raw_data,
    )

    output_path = save_funding_conditions(
        conditions=conditions,
        output_path=OUTPUT_PATH,
    )

    summary = summarise_funding_conditions(
        conditions=conditions,
    )

    abnormal_moves = int(
        conditions["is_abnormal_move"]
        .sum()
    )

    month_end_observations = int(
        conditions["is_month_end"]
        .sum()
    )

    quarter_end_observations = int(
        conditions["is_quarter_end"]
        .sum()
    )

    print()
    print("REPOLENS FUNDING CONDITIONS")
    print("=" * 64)
    print(
        "Latest observation: "
        f"{summary.latest_observation_date.date()}"
    )
    print(
        "Latest €STR: "
        f"{summary.latest_estr_rate:.3f}%"
    )
    print(
        "Latest daily change: "
        f"{format_optional_number(summary.latest_daily_change_bp)} bp"
    )
    print(
        "20-day change volatility: "
        f"{format_optional_number(summary.rolling_change_volatility_bp)} bp"
    )
    print(
        "Latest move z-score: "
        f"{format_optional_number(summary.latest_change_z_score, 2)}"
    )
    print(
        "Funding regime: "
        f"{summary.funding_regime}"
    )
    print(
        "Data freshness: "
        f"{summary.freshness_status} "
        f"({summary.business_days_stale} business days old)"
    )
    print(
        "Latest observation is month end: "
        f"{summary.is_month_end}"
    )
    print(
        "Latest observation is quarter end: "
        f"{summary.is_quarter_end}"
    )
    print(
        "Historical abnormal moves: "
        f"{abnormal_moves:,}"
    )
    print(
        "Month end observations: "
        f"{month_end_observations:,}"
    )
    print(
        "Quarter end observations: "
        f"{quarter_end_observations:,}"
    )
    print(
        "Saved to: "
        f"{output_path}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()