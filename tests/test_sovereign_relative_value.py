from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sovereign_instruments import (
    GERMANY_5Y_BOBL,
    GERMANY_10Y_BUND,
    ITALY_10Y_BTP,
)
from src.sovereign_relative_value import (
    CurveInstrumentStructure,
    PositionDirection,
    RelativeValueLeg,
    RelativeValueValidationError,
    add_residual_history_statistics,
    build_country_local_curve_relative_value,
    build_country_local_curve_relative_value_with_history,
    build_dv01_neutral_position,
    build_parallel_scenarios,
    build_spread_scenarios,
    calculate_dv01_neutral_hedge_notional,
    calculate_local_curve_fair_values,
    classify_curve_instrument_structure,
    load_local_curve_residual_history,
    persist_local_curve_residual_history,
    position_to_frame,
    rank_historical_local_curve_opportunities,
    rank_local_curve_opportunities,
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

def create_local_curve_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 2Y",
                "isin": "IT0000000001",
                "maturity_date": date(
                    2028,
                    7,
                    28,
                ),
                "years_to_maturity": 2.0,
                "yield_percent": 3.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 5Y",
                "isin": "IT0000000002",
                "maturity_date": date(
                    2031,
                    7,
                    28,
                ),
                "years_to_maturity": 5.0,
                "yield_percent": 3.60,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 7Y",
                "isin": "IT0000000003",
                "maturity_date": date(
                    2033,
                    7,
                    28,
                ),
                "years_to_maturity": 7.0,
                "yield_percent": 3.70,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 10Y",
                "isin": "IT0000000004",
                "maturity_date": date(
                    2036,
                    7,
                    28,
                ),
                "years_to_maturity": 10.0,
                "yield_percent": 4.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
        ]
    )


def test_local_curve_fair_value_is_leave_one_out_interpolation() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    five_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000002"
        )
    ].iloc[
        0
    ]

    # 5Y is bracketed by 2Y at 3.00% and 7Y at 3.70%.
    # Weight on 7Y = (5 - 2) / (7 - 2) = 0.6.
    expected_fair_yield = (
        0.4
        * 3.00
        + 0.6
        * 3.70
    )

    assert five_year[
        "fair_yield_percent"
    ] == pytest.approx(
        expected_fair_yield
    )

    assert five_year[
        "shorter_isin"
    ] == "IT0000000001"

    assert five_year[
        "longer_isin"
    ] == "IT0000000003"


def test_positive_curve_residual_is_cheap() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    five_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000002"
        )
    ].iloc[
        0
    ]

    expected_fair_yield = (
        0.4
        * 3.00
        + 0.6
        * 3.70
    )

    expected_residual_bp = (
        3.60
        - expected_fair_yield
    ) * 100.0

    assert five_year[
        "residual_bp"
    ] == pytest.approx(
        expected_residual_bp
    )

    assert five_year[
        "relative_value_label"
    ] == "CHEAP"


def test_negative_curve_residual_is_rich() -> None:
    points = create_local_curve_points()

    points.loc[
        points[
            "isin"
        ].eq(
            "IT0000000003"
        ),
        "yield_percent",
    ] = 3.40

    fair_values = calculate_local_curve_fair_values(
        points
    )

    seven_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000003"
        )
    ].iloc[
        0
    ]

    # 7Y target is bracketed by 5Y at 3.60% and 10Y at 4.00%.
    assert seven_year[
        "fair_yield_percent"
    ] == pytest.approx(
        3.76
    )

    assert seven_year[
        "residual_bp"
    ] == pytest.approx(
        -36.0
    )

    assert seven_year[
        "relative_value_label"
    ] == "RICH"


def test_curve_edges_are_not_extrapolated() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    shortest = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000001"
        )
    ].iloc[
        0
    ]

    longest = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000004"
        )
    ].iloc[
        0
    ]

    for row in (
        shortest,
        longest,
    ):
        assert not bool(
            row[
                "fair_value_available"
            ]
        )

        assert pd.isna(
            row[
                "fair_yield_percent"
            ]
        )

        assert row[
            "fair_value_exclusion_reason"
        ] == "NO_TWO_SIDED_MATURITY_BRACKET"


def test_target_bond_is_not_used_as_its_own_curve_anchor() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    five_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000002"
        )
    ].iloc[
        0
    ]

    assert five_year[
        "shorter_isin"
    ] != five_year[
        "isin"
    ]

    assert five_year[
        "longer_isin"
    ] != five_year[
        "isin"
    ]


def test_local_curve_requires_one_country_and_currency() -> None:
    points = create_local_curve_points()

    points.loc[
        points.index[
            -1
        ],
        "currency",
    ] = "GBP"

    with pytest.raises(
        RelativeValueValidationError,
        match="one country and one currency",
    ):
        calculate_local_curve_fair_values(
            points
        )


def test_duplicate_isins_are_rejected_for_curve_fair_value() -> None:
    points = create_local_curve_points()

    points.loc[
        points.index[
            -1
        ],
        "isin",
    ] = points.loc[
        points.index[
            0
        ],
        "isin",
    ]

    with pytest.raises(
        RelativeValueValidationError,
        match="duplicate ISINs",
    ):
        calculate_local_curve_fair_values(
            points
        )


def test_rank_local_curve_opportunities_uses_absolute_residual() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    ranked = rank_local_curve_opportunities(
        fair_values,
        top_n=2,
    )

    assert len(
        ranked
    ) == 2

    assert list(
        ranked[
            "rank"
        ]
    ) == [
        1,
        2,
    ]

    assert ranked[
        "absolute_residual_bp"
    ].is_monotonic_decreasing


def test_rank_local_curve_opportunities_rejects_nonpositive_top_n() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    with pytest.raises(
        RelativeValueValidationError,
        match="top_n must be positive",
    ):
        rank_local_curve_opportunities(
            fair_values,
            top_n=0,
        )


def test_country_curve_builder_reuses_snapshot_curve_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import sovereign_relative_value

    monkeypatch.setattr(
        sovereign_relative_value,
        "sovereign_curve_points",
        lambda **kwargs: create_local_curve_points(),
    )

    fair_values = build_country_local_curve_relative_value(
        country="Italy",
        currency="EUR",
        as_of_date=SETTLEMENT_DATE,
    )

    assert len(
        fair_values
    ) == 4

    assert fair_values[
        "fair_value_available"
    ].sum() == 2

def test_curve_structure_classifies_plain_fixed_btp() -> None:
    assert classify_curve_instrument_structure(
        "Btp Tf 3,50% Mz30 Eur"
    ) == CurveInstrumentStructure.PLAIN_FIXED_RATE_NOMINAL


@pytest.mark.parametrize(
    "name",
    [
        "Btp Valore Sc Mz30 Eur",
        "Btp Futura Ap37 Eur",
        "Btp Piu' Sc Fb33 Eur",
        "BTP Più Sc Fb33 Eur",
    ],
)
def test_curve_structure_excludes_retail_step_up_products(
    name: str,
) -> None:
    assert classify_curve_instrument_structure(
        name
    ) == CurveInstrumentStructure.STRUCTURED_RETAIL_STEP_UP


def test_structured_retail_bond_is_not_given_plain_curve_fair_value() -> None:
    points = create_local_curve_points()

    points.loc[
        points[
            "isin"
        ].eq(
            "IT0000000002"
        ),
        "display_name",
    ] = "Btp Valore Sc Mz30 Eur"

    fair_values = calculate_local_curve_fair_values(
        points
    )

    structured = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000002"
        )
    ].iloc[
        0
    ]

    assert not bool(
        structured[
            "curve_model_eligible"
        ]
    )

    assert not bool(
        structured[
            "fair_value_available"
        ]
    )

    assert structured[
        "instrument_structure"
    ] == (
        CurveInstrumentStructure
        .STRUCTURED_RETAIL_STEP_UP
        .value
    )

    assert structured[
        "fair_value_exclusion_reason"
    ] == (
        CurveInstrumentStructure
        .STRUCTURED_RETAIL_STEP_UP
        .value
    )


def test_structured_retail_bond_is_not_used_as_interpolation_anchor() -> None:
    points = pd.DataFrame(
        [
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP plain 2Y",
                "isin": "IT0000000101",
                "maturity_date": date(2028, 7, 28),
                "years_to_maturity": 2.0,
                "yield_percent": 3.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "Btp Valore Sc Mz30 Eur",
                "isin": "IT0000000102",
                "maturity_date": date(2030, 7, 28),
                "years_to_maturity": 4.0,
                "yield_percent": 9.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP plain 5Y",
                "isin": "IT0000000103",
                "maturity_date": date(2031, 7, 28),
                "years_to_maturity": 5.0,
                "yield_percent": 3.55,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP plain 7Y",
                "isin": "IT0000000104",
                "maturity_date": date(2033, 7, 28),
                "years_to_maturity": 7.0,
                "yield_percent": 3.70,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP plain 10Y",
                "isin": "IT0000000105",
                "maturity_date": date(2036, 7, 28),
                "years_to_maturity": 10.0,
                "yield_percent": 4.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
        ]
    )

    fair_values = calculate_local_curve_fair_values(
        points
    )

    five_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000103"
        )
    ].iloc[
        0
    ]

    assert five_year[
        "shorter_isin"
    ] == "IT0000000101"

    assert five_year[
        "longer_isin"
    ] == "IT0000000104"

    expected_fair = (
        0.4
        * 3.00
        + 0.6
        * 3.70
    )

    assert five_year[
        "fair_yield_percent"
    ] == pytest.approx(
        expected_fair
    )


def test_ranked_opportunities_exclude_structured_retail_products() -> None:
    points = create_local_curve_points()

    points.loc[
        points[
            "isin"
        ].eq(
            "IT0000000002"
        ),
        "display_name",
    ] = "Btp Futura Ap37 Eur"

    fair_values = calculate_local_curve_fair_values(
        points
    )

    ranked = rank_local_curve_opportunities(
        fair_values,
        top_n=20,
    )

    assert "IT0000000002" not in set(
        ranked[
            "isin"
        ]
    )

def test_curve_geometry_fields_are_reported() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    five_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000002"
        )
    ].iloc[
        0
    ]

    assert five_year[
        "shorter_gap_years"
    ] == pytest.approx(
        3.0
    )

    assert five_year[
        "longer_gap_years"
    ] == pytest.approx(
        2.0
    )

    assert five_year[
        "total_bracket_width_years"
    ] == pytest.approx(
        5.0
    )

    assert five_year[
        "interpolation_balance_ratio"
    ] == pytest.approx(
        2.0 / 3.0
    )

    assert five_year[
        "curve_location"
    ] == "BELLY"

    assert five_year[
        "interpolation_quality"
    ] in {
        "HIGH",
        "MEDIUM",
        "LOW",
    }

    assert 0.0 < five_year[
        "interpolation_quality_score"
    ] <= 1.0


def test_curve_location_labels_front_belly_and_long_end() -> None:
    points = pd.DataFrame(
        [
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 1Y",
                "isin": "IT0000000201",
                "maturity_date": date(2027, 7, 28),
                "years_to_maturity": 1.0,
                "yield_percent": 2.80,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 5Y",
                "isin": "IT0000000202",
                "maturity_date": date(2031, 7, 28),
                "years_to_maturity": 5.0,
                "yield_percent": 3.40,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 12Y",
                "isin": "IT0000000203",
                "maturity_date": date(2038, 7, 28),
                "years_to_maturity": 12.0,
                "yield_percent": 4.10,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 20Y",
                "isin": "IT0000000204",
                "maturity_date": date(2046, 7, 28),
                "years_to_maturity": 20.0,
                "yield_percent": 4.40,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
        ]
    )

    fair_values = calculate_local_curve_fair_values(
        points
    )

    lookup = fair_values.set_index(
        "isin"
    )[
        "curve_location"
    ].to_dict()

    assert lookup[
        "IT0000000201"
    ] == "FRONT_END"

    assert lookup[
        "IT0000000202"
    ] == "BELLY"

    assert lookup[
        "IT0000000203"
    ] == "LONG_END"

    assert lookup[
        "IT0000000204"
    ] == "LONG_END"


def test_more_balanced_interpolation_has_better_quality_score() -> None:
    points = pd.DataFrame(
        [
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 1Y",
                "isin": "IT0000000301",
                "maturity_date": date(2027, 7, 28),
                "years_to_maturity": 1.0,
                "yield_percent": 2.80,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 2Y",
                "isin": "IT0000000302",
                "maturity_date": date(2028, 7, 28),
                "years_to_maturity": 2.0,
                "yield_percent": 3.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 3Y",
                "isin": "IT0000000303",
                "maturity_date": date(2029, 7, 28),
                "years_to_maturity": 3.0,
                "yield_percent": 3.20,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 6Y",
                "isin": "IT0000000304",
                "maturity_date": date(2032, 7, 28),
                "years_to_maturity": 6.0,
                "yield_percent": 3.60,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP 10Y",
                "isin": "IT0000000305",
                "maturity_date": date(2036, 7, 28),
                "years_to_maturity": 10.0,
                "yield_percent": 4.00,
                "observation_date": SETTLEMENT_DATE,
                "market_source": "Public reference",
                "market_status": "PUBLIC_REFERENCE",
            },
        ]
    )

    fair_values = calculate_local_curve_fair_values(
        points
    )

    two_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000302"
        )
    ].iloc[
        0
    ]

    three_year = fair_values.loc[
        fair_values[
            "isin"
        ].eq(
            "IT0000000303"
        )
    ].iloc[
        0
    ]

    assert two_year[
        "interpolation_balance_ratio"
    ] > three_year[
        "interpolation_balance_ratio"
    ]

    assert two_year[
        "interpolation_quality_score"
    ] > three_year[
        "interpolation_quality_score"
    ]


def test_ranker_preserves_quality_context() -> None:
    fair_values = calculate_local_curve_fair_values(
        create_local_curve_points()
    )

    ranked = rank_local_curve_opportunities(
        fair_values,
        top_n=20,
    )

    assert "interpolation_quality" in ranked.columns
    assert "interpolation_quality_score" in ranked.columns
    assert "total_bracket_width_years" in ranked.columns
    assert "curve_location" in ranked.columns




def test_residual_history_upsert_is_idempotent(tmp_path: Path) -> None:
    path=tmp_path/"rv_history.csv"
    current=calculate_local_curve_fair_values(create_local_curve_points())
    first=persist_local_curve_residual_history(current,path=path)
    second=persist_local_curve_residual_history(current,path=path)
    assert len(first)==2
    assert len(second)==2
    assert not second.duplicated(subset=["observation_date","isin","model_version"]).any()


def test_residual_history_excludes_unavailable_edges(tmp_path: Path) -> None:
    path=tmp_path/"rv_history.csv"
    current=calculate_local_curve_fair_values(create_local_curve_points())
    history=persist_local_curve_residual_history(current,path=path)
    assert set(history["isin"])=={"IT0000000002","IT0000000003"}


def test_missing_history_file_returns_empty_schema(tmp_path: Path) -> None:
    history=load_local_curve_residual_history(tmp_path/"missing.csv")
    assert history.empty
    assert "residual_bp" in history.columns
    assert "model_version" in history.columns


def test_z_score_excludes_current_observation_date() -> None:
    current=calculate_local_curve_fair_values(create_local_curve_points())
    target=current.loc[current["isin"].eq("IT0000000002")].copy()
    target.loc[:,"observation_date"]=date(2026,9,4)
    history=pd.DataFrame([
        {"observation_date":date(2026,9,1),"isin":"IT0000000002","residual_bp":1.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
        {"observation_date":date(2026,9,2),"isin":"IT0000000002","residual_bp":2.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
        {"observation_date":date(2026,9,3),"isin":"IT0000000002","residual_bp":3.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
        {"observation_date":date(2026,9,4),"isin":"IT0000000002","residual_bp":999.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
    ])
    row=add_residual_history_statistics(target,history=history,lookback_observations=3,minimum_history_observations=3).iloc[0]
    assert row["history_observation_count"]==3
    assert row["historical_mean_residual_bp"]==pytest.approx(2.0)
    expected_std=float(np.std([1.0,2.0,3.0],ddof=0))
    assert row["historical_std_residual_bp"]==pytest.approx(expected_std)
    assert row["z_score"]==pytest.approx((float(row["residual_bp"])-2.0)/expected_std)


def test_z_score_requires_minimum_history() -> None:
    current=calculate_local_curve_fair_values(create_local_curve_points())
    target=current.loc[current["isin"].eq("IT0000000002")].copy()
    target.loc[:,"observation_date"]=date(2026,9,4)
    history=pd.DataFrame([
        {"observation_date":date(2026,9,1),"isin":"IT0000000002","residual_bp":1.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
        {"observation_date":date(2026,9,2),"isin":"IT0000000002","residual_bp":2.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"},
    ])
    row=add_residual_history_statistics(target,history=history,lookback_observations=60,minimum_history_observations=20).iloc[0]
    assert row["history_observation_count"]==2
    assert not bool(row["history_ready"])
    assert pd.isna(row["z_score"])


def test_zero_volatility_returns_no_z_score() -> None:
    current=calculate_local_curve_fair_values(create_local_curve_points())
    target=current.loc[current["isin"].eq("IT0000000002")].copy()
    target.loc[:,"observation_date"]=date(2026,9,4)
    history=pd.DataFrame([
        {"observation_date":date(2026,9,d),"isin":"IT0000000002","residual_bp":5.0,"model_version":"LOCAL_LINEAR_PLAIN_FIXED_V1"}
        for d in (1,2,3)
    ])
    row=add_residual_history_statistics(target,history=history,lookback_observations=3,minimum_history_observations=3).iloc[0]
    assert bool(row["history_ready"])
    assert row["historical_std_residual_bp"]==pytest.approx(0.0)
    assert pd.isna(row["z_score"])


def test_historical_ranker_orders_by_absolute_z_score() -> None:
    frame=pd.DataFrame([
        {"isin":"IT0000000001","fair_value_available":True,"history_ready":True,"z_score":-2.0,"residual_bp":-5.0,"absolute_residual_bp":5.0,"years_to_maturity":5.0},
        {"isin":"IT0000000002","fair_value_available":True,"history_ready":True,"z_score":3.0,"residual_bp":4.0,"absolute_residual_bp":4.0,"years_to_maturity":7.0},
    ])
    ranked=rank_historical_local_curve_opportunities(frame,top_n=20)
    assert list(ranked["isin"])==["IT0000000002","IT0000000001"]


def test_builder_persists_current_without_same_day_lookahead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import sovereign_relative_value
    current=calculate_local_curve_fair_values(create_local_curve_points())
    monkeypatch.setattr(sovereign_relative_value,"build_country_local_curve_relative_value",lambda **kwargs: current)
    path=tmp_path/"rv_history.csv"
    enriched=build_country_local_curve_relative_value_with_history(
        country="Italy",currency="EUR",history_path=path,persist_current=True,
        lookback_observations=60,minimum_history_observations=20,
    )
    persisted=load_local_curve_residual_history(path)
    assert len(persisted)==2
    available=enriched.loc[enriched["fair_value_available"].eq(True)]
    assert set(available["history_observation_count"])=={0}
    assert available["z_score"].isna().all()
