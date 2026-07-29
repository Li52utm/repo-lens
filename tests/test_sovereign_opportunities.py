from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.sovereign_opportunities import (
    OpportunityConviction,
    OpportunityDirection,
    SovereignOpportunityInput,
    SovereignOpportunityValidationError,
    build_opportunity,
    build_opportunity_set,
    classify_conviction,
    curve_slope_frame,
    determine_trade_direction,
    opportunities_to_frame,
)


SETTLEMENT_DATE = date(
    2026,
    7,
    28,
)


def create_german_curve() -> pd.DataFrame:
    """
    Create deterministic German benchmark observations.
    """
    return pd.DataFrame(
        [
            {
                "observation_date": "2026-07-27",
                "country_code": "DE",
                "tenor_years": 2,
                "yield_percent": 2.15,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 2,
                "yield_percent": 2.20,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 5,
                "yield_percent": 2.45,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 10,
                "yield_percent": 2.85,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 30,
                "yield_percent": 3.15,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
        ]
    )


def create_opportunity_inputs() -> tuple[
    SovereignOpportunityInput,
    ...,
]:
    """
    Create complete 2Y, 5Y, 10Y and 30Y opportunity inputs.
    """
    return (
        SovereignOpportunityInput(
            tenor_years=2,
            italian_yield_percent=2.70,
            target_spread_bp=45.0,
        ),
        SovereignOpportunityInput(
            tenor_years=5,
            italian_yield_percent=3.20,
            target_spread_bp=65.0,
        ),
        SovereignOpportunityInput(
            tenor_years=10,
            italian_yield_percent=3.85,
            target_spread_bp=85.0,
        ),
        SovereignOpportunityInput(
            tenor_years=30,
            italian_yield_percent=4.65,
            target_spread_bp=140.0,
        ),
    )


def test_direction_for_spread_narrowing() -> None:
    direction = determine_trade_direction(
        target_spread_change_bp=-10.0,
        no_trade_threshold_bp=2.0,
    )

    assert direction == (
        OpportunityDirection
        .LONG_ITALY_SHORT_GERMANY
    )


def test_direction_for_spread_widening() -> None:
    direction = determine_trade_direction(
        target_spread_change_bp=10.0,
        no_trade_threshold_bp=2.0,
    )

    assert direction == (
        OpportunityDirection
        .SHORT_ITALY_LONG_GERMANY
    )


def test_small_dislocation_returns_no_trade() -> None:
    direction = determine_trade_direction(
        target_spread_change_bp=1.0,
        no_trade_threshold_bp=2.0,
    )

    assert direction == OpportunityDirection.NO_TRADE


def test_conviction_classification() -> None:
    assert classify_conviction(
        absolute_dislocation_bp=30.0,
        no_trade_threshold_bp=2.0,
    ) == OpportunityConviction.HIGH

    assert classify_conviction(
        absolute_dislocation_bp=15.0,
        no_trade_threshold_bp=2.0,
    ) == OpportunityConviction.MEDIUM

    assert classify_conviction(
        absolute_dislocation_bp=5.0,
        no_trade_threshold_bp=2.0,
    ) == OpportunityConviction.LOW

    assert classify_conviction(
        absolute_dislocation_bp=1.0,
        no_trade_threshold_bp=2.0,
    ) == OpportunityConviction.NEUTRAL


def test_build_narrowing_opportunity() -> None:
    result = build_opportunity(
        opportunity_input=(
            SovereignOpportunityInput(
                tenor_years=10,
                italian_yield_percent=3.85,
                target_spread_bp=85.0,
            )
        ),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        anchor_notional_eur=10_000_000.0,
    )

    assert result.current_spread_bp == pytest.approx(
        100.0
    )

    assert result.target_spread_change_bp == pytest.approx(
        -15.0
    )

    assert result.trade_direction == (
        OpportunityDirection
        .LONG_ITALY_SHORT_GERMANY
    )

    assert result.target_pnl_eur > 0.0
    assert result.adverse_pnl_eur < 0.0
    assert result.reward_to_risk > 0.0
    assert result.hedge_notional_eur > 0.0

    assert result.net_dv01_eur == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_build_widening_opportunity() -> None:
    result = build_opportunity(
        opportunity_input=(
            SovereignOpportunityInput(
                tenor_years=10,
                italian_yield_percent=3.85,
                target_spread_bp=115.0,
            )
        ),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    assert result.trade_direction == (
        OpportunityDirection
        .SHORT_ITALY_LONG_GERMANY
    )

    assert result.target_spread_change_bp == pytest.approx(
        15.0
    )

    assert result.target_pnl_eur > 0.0
    assert result.adverse_pnl_eur < 0.0


def test_no_trade_has_zero_scenario_values() -> None:
    result = build_opportunity(
        opportunity_input=(
            SovereignOpportunityInput(
                tenor_years=10,
                italian_yield_percent=3.85,
                target_spread_bp=101.0,
            )
        ),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        no_trade_threshold_bp=2.0,
    )

    assert result.trade_direction == (
        OpportunityDirection.NO_TRADE
    )

    assert result.conviction == (
        OpportunityConviction.NEUTRAL
    )

    assert result.target_pnl_eur == pytest.approx(
        0.0
    )

    assert result.adverse_pnl_eur == pytest.approx(
        0.0
    )


def test_opportunity_set_requires_all_tenors() -> None:
    with pytest.raises(
        SovereignOpportunityValidationError,
        match="cover exactly",
    ):
        build_opportunity_set(
            opportunity_inputs=(
                SovereignOpportunityInput(
                    tenor_years=10,
                    italian_yield_percent=3.85,
                    target_spread_bp=85.0,
                ),
            ),
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_opportunity_set_is_ranked() -> None:
    opportunities = build_opportunity_set(
        opportunity_inputs=create_opportunity_inputs(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    assert len(
        opportunities
    ) == 4

    active_opportunities = [
        opportunity
        for opportunity in opportunities
        if opportunity.trade_direction
        != OpportunityDirection.NO_TRADE
    ]

    reward_to_risk_values = [
        opportunity.reward_to_risk
        for opportunity
        in active_opportunities
    ]

    assert reward_to_risk_values == sorted(
        reward_to_risk_values,
        reverse=True,
    )


def test_opportunities_frame_contract() -> None:
    opportunities = build_opportunity_set(
        opportunity_inputs=create_opportunity_inputs(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    frame = opportunities_to_frame(
        opportunities
    )

    assert len(
        frame
    ) == 4

    assert {
        "rank",
        "tenor_years",
        "italian_yield_percent",
        "german_yield_percent",
        "current_spread_bp",
        "target_spread_bp",
        "trade_direction",
        "conviction",
        "hedge_notional_eur",
        "target_pnl_eur",
        "adverse_pnl_eur",
        "reward_to_risk",
    }.issubset(
        frame.columns
    )


def test_curve_slope_frame() -> None:
    slopes = curve_slope_frame(
        opportunity_inputs=create_opportunity_inputs(),
        german_curve=create_german_curve(),
    )

    assert list(
        slopes[
            "curve_segment"
        ]
    ) == [
        "2s5s",
        "2s10s",
        "5s10s",
        "10s30s",
    ]

    germany_2s10s = slopes.loc[
        slopes[
            "curve_segment"
        ].eq(
            "2s10s"
        )
    ].iloc[0]

    assert germany_2s10s[
        "germany_slope_bp"
    ] == pytest.approx(
        65.0
    )


def test_duplicate_tenors_are_rejected() -> None:
    duplicate = (
        SovereignOpportunityInput(
            tenor_years=2,
            italian_yield_percent=2.70,
            target_spread_bp=45.0,
        ),
        SovereignOpportunityInput(
            tenor_years=2,
            italian_yield_percent=2.75,
            target_spread_bp=50.0,
        ),
        SovereignOpportunityInput(
            tenor_years=10,
            italian_yield_percent=3.85,
            target_spread_bp=85.0,
        ),
        SovereignOpportunityInput(
            tenor_years=30,
            italian_yield_percent=4.65,
            target_spread_bp=140.0,
        ),
    )

    with pytest.raises(
        SovereignOpportunityValidationError,
        match="unique",
    ):
        build_opportunity_set(
            opportunity_inputs=duplicate,
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_missing_german_tenor_is_rejected() -> None:
    broken_curve = create_german_curve().loc[
        lambda frame: frame[
            "tenor_years"
        ].ne(
            30
        )
    ]

    with pytest.raises(
        SovereignOpportunityValidationError,
        match="missing supported tenors",
    ):
        build_opportunity_set(
            opportunity_inputs=create_opportunity_inputs(),
            german_curve=broken_curve,
            settlement_date=SETTLEMENT_DATE,
        )


def test_invalid_anchor_notional_is_rejected() -> None:
    with pytest.raises(
        SovereignOpportunityValidationError,
        match="anchor_notional_eur must be positive",
    ):
        build_opportunity(
            opportunity_input=(
                SovereignOpportunityInput(
                    tenor_years=10,
                    italian_yield_percent=3.85,
                    target_spread_bp=85.0,
                )
            ),
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
            anchor_notional_eur=0.0,
        )