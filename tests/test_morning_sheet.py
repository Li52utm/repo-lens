from __future__ import annotations

import pandas as pd
import pytest

from src.morning_sheet import (
    MorningSheetValidationError,
    build_morning_sheet,
    classify_overall_status,
    prepare_policy_spread,
    summarise_morning_sheet,
)


def create_dates() -> pd.DatetimeIndex:
    """
    Create deterministic common business dates.
    """
    return pd.bdate_range(
        start="2026-07-20",
        periods=5,
    )


def create_funding_conditions() -> pd.DataFrame:
    """
    Create deterministic funding-condition data.
    """
    return pd.DataFrame(
        {
            "observation_date": create_dates(),
            "estr_rate": [
                2.180,
                2.181,
                2.182,
                2.185,
                2.186,
            ],
            "daily_change_bp": [
                0.0,
                0.1,
                0.1,
                0.3,
                0.1,
            ],
            "rolling_change_volatility_bp": [
                0.10,
                0.10,
                0.11,
                0.11,
                0.09,
            ],
            "change_z_score": [
                0.0,
                0.5,
                0.4,
                2.5,
                0.8,
            ],
            "funding_regime": [
                "Normal",
                "Normal",
                "Normal",
                "Volatile",
                "Normal",
            ],
            "is_abnormal_move": [
                False,
                False,
                False,
                True,
                False,
            ],
            "is_month_end": [
                False,
                False,
                False,
                False,
                True,
            ],
            "is_quarter_end": [
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


def create_policy_spread() -> pd.DataFrame:
    """
    Create deterministic policy-spread data.

    The deliberately incorrect source flag demonstrates that the
    Morning Sheet recalculates policy changes from the rate itself.
    """
    return pd.DataFrame(
        {
            "observation_date": create_dates(),
            "deposit_facility_rate": [
                2.25,
                2.25,
                2.00,
                2.00,
                2.00,
            ],
            "estr_policy_spread_bp": [
                -7.0,
                -6.9,
                18.2,
                18.5,
                18.6,
            ],
            "spread_z_score": [
                0.0,
                0.2,
                3.0,
                2.0,
                1.0,
            ],
            "transmission_regime": [
                "Normal transmission",
                "Normal transmission",
                "Unusually above policy rate",
                "Unusually above policy rate",
                "Above policy rate",
            ],
            "is_unusual_spread": [
                False,
                False,
                True,
                True,
                False,
            ],
            "is_policy_change_day": [
                False,
                False,
                True,
                True,
                True,
            ],
        }
    )


def create_market_quality() -> pd.DataFrame:
    """
    Create deterministic market-quality data.
    """
    return pd.DataFrame(
        {
            "observation_date": create_dates(),
            "total_volume_eur_mn": [
                60_000.0,
                61_000.0,
                62_000.0,
                64_000.0,
                63_789.0,
            ],
            "active_banks": [
                45.0,
                45.0,
                46.0,
                46.0,
                46.0,
            ],
            "transaction_count": [
                850.0,
                860.0,
                880.0,
                900.0,
                908.0,
            ],
            "rate_dispersion_bp": [
                4.5,
                4.4,
                4.3,
                4.2,
                4.0,
            ],
            "market_quality_score": [
                49.0,
                50.0,
                51.0,
                52.0,
                51.8566,
            ],
            "quality_regime": [
                "Normal market quality",
                "Normal market quality",
                "Normal market quality",
                "Normal market quality",
                "Normal market quality",
            ],
            "is_market_quality_alert": [
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_policy_change_is_recalculated_correctly() -> None:
    prepared = prepare_policy_spread(
        create_policy_spread()
    )

    assert prepared[
        "policy_rate_changed_today"
    ].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_build_morning_sheet_combines_common_dates() -> None:
    morning_sheet = build_morning_sheet(
        funding_conditions=create_funding_conditions(),
        policy_spread=create_policy_spread(),
        market_quality=create_market_quality(),
    )

    assert len(
        morning_sheet
    ) == 5

    assert morning_sheet[
        "observation_date"
    ].is_monotonic_increasing


def test_build_morning_sheet_contains_runner_contract() -> None:
    morning_sheet = build_morning_sheet(
        funding_conditions=create_funding_conditions(),
        policy_spread=create_policy_spread(),
        market_quality=create_market_quality(),
    )

    required_runner_columns = {
        "observation_date",
        "estr_rate",
        "daily_change_bp",
        "funding_regime",
        "deposit_facility_rate",
        "policy_rate_changed_today",
        "estr_policy_spread_bp",
        "spread_z_score",
        "transmission_regime",
        "market_quality_score",
        "market_quality_change",
        "quality_regime",
        "total_volume_eur_mn",
        "volume_change_pct",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "alert_count",
        "overall_status",
    }

    assert required_runner_columns.issubset(
        morning_sheet.columns
    )


def test_market_quality_changes_are_calculated() -> None:
    morning_sheet = build_morning_sheet(
        funding_conditions=create_funding_conditions(),
        policy_spread=create_policy_spread(),
        market_quality=create_market_quality(),
    )

    assert morning_sheet.iloc[-1][
        "market_quality_change"
    ] == pytest.approx(
        51.8566 - 52.0
    )

    assert morning_sheet.iloc[1][
        "volume_change_pct"
    ] == pytest.approx(
        (
            61_000.0
            / 60_000.0
            - 1.0
        )
        * 100.0
    )


def test_alert_count_combines_three_alert_sources() -> None:
    funding = create_funding_conditions()
    policy = create_policy_spread()
    quality = create_market_quality()

    funding.loc[
        3,
        "is_abnormal_move",
    ] = True

    policy.loc[
        3,
        "is_unusual_spread",
    ] = True

    quality.loc[
        3,
        "is_market_quality_alert",
    ] = True

    morning_sheet = build_morning_sheet(
        funding_conditions=funding,
        policy_spread=policy,
        market_quality=quality,
    )

    assert morning_sheet.iloc[3][
        "alert_count"
    ] == 3

    assert morning_sheet.iloc[3][
        "overall_status"
    ] == "High attention"


def test_classify_overall_status() -> None:
    assert classify_overall_status(
        alert_count=2,
        is_month_end=False,
        is_quarter_end=False,
    ) == "High attention"

    assert classify_overall_status(
        alert_count=1,
        is_month_end=False,
        is_quarter_end=False,
    ) == "Monitor"

    assert classify_overall_status(
        alert_count=0,
        is_month_end=False,
        is_quarter_end=True,
    ) == "Quarter-end watch"

    assert classify_overall_status(
        alert_count=0,
        is_month_end=True,
        is_quarter_end=False,
    ) == "Month-end watch"

    assert classify_overall_status(
        alert_count=0,
        is_month_end=False,
        is_quarter_end=False,
    ) == "Normal"


def test_summary_returns_latest_observation() -> None:
    morning_sheet = build_morning_sheet(
        funding_conditions=create_funding_conditions(),
        policy_spread=create_policy_spread(),
        market_quality=create_market_quality(),
    )

    summary = summarise_morning_sheet(
        morning_sheet
    )

    latest = morning_sheet.iloc[-1]

    assert summary.observation_date == pd.Timestamp(
        latest[
            "observation_date"
        ]
    )

    assert summary.estr_rate == pytest.approx(
        latest[
            "estr_rate"
        ]
    )

    assert summary.market_quality_score == pytest.approx(
        latest[
            "market_quality_score"
        ]
    )

    assert summary.alert_count == int(
        latest[
            "alert_count"
        ]
    )


def test_missing_input_column_is_rejected() -> None:
    funding = create_funding_conditions().drop(
        columns=[
            "funding_regime",
        ]
    )

    with pytest.raises(
        MorningSheetValidationError,
        match="missing required columns",
    ):
        build_morning_sheet(
            funding_conditions=funding,
            policy_spread=create_policy_spread(),
            market_quality=create_market_quality(),
        )


def test_no_common_dates_is_rejected() -> None:
    quality = create_market_quality()

    quality["observation_date"] = (
        pd.bdate_range(
            start="2030-01-01",
            periods=5,
        )
    )

    with pytest.raises(
        MorningSheetValidationError,
        match="no common observation dates",
    ):
        build_morning_sheet(
            funding_conditions=create_funding_conditions(),
            policy_spread=create_policy_spread(),
            market_quality=quality,
        )