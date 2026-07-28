from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.sovereign_instruments import (
    GERMANY_5Y_BOBL,
    GERMANY_10Y_BUND,
    ITALY_10Y_BTP,
)
from src.sovereign_relative_value import (
    PositionDirection,
    RelativeValueLeg,
    RelativeValueValidationError,
    build_dv01_neutral_position,
    build_parallel_scenarios,
    build_spread_scenarios,
    calculate_dv01_neutral_hedge_notional,
    position_to_frame,
)
from src.sovereign_snapshot import SovereignYieldInput


SETTLEMENT_DATE = date(
    2026,
    7,
    28,
)


def create_german_curve() -> pd.DataFrame:
    """
    Create deterministic official German benchmark observations.
    """
    return pd.DataFrame(
        [
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


def create_btp_yield_input() -> SovereignYieldInput:
    """
    Create a deterministic desk-input BTP yield.
    """
    return SovereignYieldInput(
        isin=ITALY_10Y_BTP.isin,
        yield_percent=3.85,
        observation_date=SETTLEMENT_DATE,
        source_name="Desk input",
    )


def create_standard_legs() -> tuple[
    RelativeValueLeg,
    RelativeValueLeg,
]:
    """
    Create a long BTP versus short Bund trade.
    """
    anchor_leg = RelativeValueLeg(
        instrument=ITALY_10Y_BTP,
        direction=PositionDirection.LONG,
        yield_input=create_btp_yield_input(),
    )

    hedge_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.SHORT,
    )

    return (
        anchor_leg,
        hedge_leg,
    )


def test_hedge_notional_matches_dv01_ratio() -> None:
    hedge_notional = (
        calculate_dv01_neutral_hedge_notional(
            anchor_dv01_per_eur_1m=850.0,
            hedge_dv01_per_eur_1m=800.0,
            anchor_notional_eur=10_000_000.0,
        )
    )

    assert hedge_notional == pytest.approx(
        10_625_000.0
    )


def test_zero_anchor_notional_is_rejected() -> None:
    with pytest.raises(
        RelativeValueValidationError,
        match="must be positive",
    ):
        calculate_dv01_neutral_hedge_notional(
            anchor_dv01_per_eur_1m=850.0,
            hedge_dv01_per_eur_1m=800.0,
            anchor_notional_eur=0.0,
        )


def test_zero_hedge_dv01_is_rejected() -> None:
    with pytest.raises(
        RelativeValueValidationError,
        match="hedge_dv01_per_eur_1m must be positive",
    ):
        calculate_dv01_neutral_hedge_notional(
            anchor_dv01_per_eur_1m=850.0,
            hedge_dv01_per_eur_1m=0.0,
            anchor_notional_eur=10_000_000.0,
        )


def test_same_instrument_is_rejected() -> None:
    anchor_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.LONG,
    )

    hedge_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.SHORT,
    )

    with pytest.raises(
        RelativeValueValidationError,
        match="must be different",
    ):
        build_dv01_neutral_position(
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_same_directions_are_rejected() -> None:
    anchor_leg = RelativeValueLeg(
        instrument=ITALY_10Y_BTP,
        direction=PositionDirection.LONG,
        yield_input=create_btp_yield_input(),
    )

    hedge_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.LONG,
    )

    with pytest.raises(
        RelativeValueValidationError,
        match="opposite directions",
    ):
        build_dv01_neutral_position(
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_position_is_dv01_neutral() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        anchor_notional_eur=10_000_000.0,
    )

    assert position.anchor_notional_eur == pytest.approx(
        10_000_000.0
    )

    assert position.hedge_notional_eur > 0.0

    assert position.anchor_position_dv01_eur == pytest.approx(
        position.hedge_position_dv01_eur,
        rel=1e-10,
    )

    assert position.net_dv01_eur == pytest.approx(
        0.0,
        abs=1e-8,
    )

    assert position.dv01_hedge_error_eur == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_position_spread_is_anchor_minus_hedge() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    assert position.anchor_yield_percent == pytest.approx(
        3.85
    )

    assert position.hedge_yield_percent == pytest.approx(
        2.85
    )

    assert position.spread_bp == pytest.approx(
        100.0
    )


def test_hedge_ratio_scales_independently_of_anchor_notional() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    ten_million = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        anchor_notional_eur=10_000_000.0,
    )

    twenty_million = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        anchor_notional_eur=20_000_000.0,
    )

    assert (
        twenty_million.hedge_notional_eur
        == pytest.approx(
            ten_million.hedge_notional_eur
            * 2.0
        )
    )

    assert (
        twenty_million.hedge_notional_per_anchor_euro
        == pytest.approx(
            ten_million.hedge_notional_per_anchor_euro
        )
    )


def test_missing_btp_yield_is_rejected() -> None:
    anchor_leg = RelativeValueLeg(
        instrument=ITALY_10Y_BTP,
        direction=PositionDirection.LONG,
    )

    hedge_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.SHORT,
    )

    with pytest.raises(
        RelativeValueValidationError,
        match="explicit instrument-level yield",
    ):
        build_dv01_neutral_position(
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_spread_widening_hurts_long_btp_short_bund() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    scenarios = build_spread_scenarios(
        position=position,
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        spread_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    narrowing = scenarios.iloc[0]
    widening = scenarios.iloc[1]

    assert narrowing[
        "total_pnl_eur"
    ] > 0.0

    assert widening[
        "total_pnl_eur"
    ] < 0.0


def test_spread_scenario_changes_spread_by_requested_amount() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    scenarios = build_spread_scenarios(
        position=position,
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        spread_shocks_bp=(
            15.0,
        ),
    )

    result = scenarios.iloc[0]

    assert result[
        "shocked_spread_bp"
    ] == pytest.approx(
        position.spread_bp
        + 15.0
    )

    assert result[
        "anchor_yield_shock_bp"
    ] == pytest.approx(
        7.5
    )

    assert result[
        "hedge_yield_shock_bp"
    ] == pytest.approx(
        -7.5
    )


def test_parallel_shock_has_small_first_order_pnl() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    scenarios = build_parallel_scenarios(
        position=position,
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        parallel_shocks_bp=(
            -1.0,
            1.0,
        ),
    )

    assert abs(
        scenarios.iloc[0][
            "total_pnl_eur"
        ]
    ) < 1_000.0

    assert abs(
        scenarios.iloc[1][
            "total_pnl_eur"
        ]
    ) < 1_000.0


def test_position_frame_contains_two_legs() -> None:
    anchor_leg, hedge_leg = create_standard_legs()

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    frame = position_to_frame(
        position
    )

    assert len(
        frame
    ) == 2

    assert list(
        frame["leg"]
    ) == [
        "Anchor",
        "Hedge",
    ]

    assert set(
        frame["direction"]
    ) == {
        "LONG",
        "SHORT",
    }


def test_cross_tenor_trade_can_be_constructed() -> None:
    anchor_leg = RelativeValueLeg(
        instrument=GERMANY_5Y_BOBL,
        direction=PositionDirection.LONG,
    )

    hedge_leg = RelativeValueLeg(
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.SHORT,
    )

    position = build_dv01_neutral_position(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
        anchor_notional_eur=10_000_000.0,
    )

    assert position.anchor_isin == (
        GERMANY_5Y_BOBL.isin
    )

    assert position.hedge_isin == (
        GERMANY_10Y_BUND.isin
    )

    assert position.hedge_notional_eur > 0.0