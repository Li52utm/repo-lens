from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.sovereign_opportunities import (
    OpportunityConviction,
    OpportunityDirection,
    SovereignOpportunityInput,
    SovereignOpportunityValidationError,
    add_cross_sectional_z_scores,
    build_historical_local_curve_scanner,
    build_local_curve_opportunity_scanner,
    build_opportunity,
    build_opportunity_set,
    classify_conviction,
    curve_slope_frame,
    determine_trade_direction,
    opportunities_to_frame,
    selected_local_curve_bond,
    validate_local_curve_scanner_input,
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


def create_local_curve_scanner_frame() -> pd.DataFrame:
    """
    Create deterministic instrument-level RV observations for scanner tests.
    """
    return pd.DataFrame(
        [
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP A",
                "isin": "IT0000001001",
                "maturity_date": date(2028, 9, 1),
                "years_to_maturity": 2.0,
                "actual_yield_percent": 3.10,
                "fair_yield_percent": 3.00,
                "residual_bp": 10.0,
                "absolute_residual_bp": 10.0,
                "relative_value_label": "CHEAP",
                "curve_location": "FRONT_END",
                "fair_value_available": True,
                "interpolation_quality_score": 0.80,
                "interpolation_quality": "HIGH",
                "market_source": "Borsa Italiana / Skipper Informatica",
                "market_status": "PUBLIC_REFERENCE",
                "model_name": "LEAVE_ONE_OUT_LOCAL_LINEAR",
                "model_version": "LOCAL_LINEAR_PLAIN_FIXED_V1",
                "model_status": "REPOLENS_DERIVED",
                "history_observation_count": 0,
                "historical_mean_residual_bp": float("nan"),
                "historical_std_residual_bp": float("nan"),
                "historical_percentile": float("nan"),
                "z_score": float("nan"),
                "history_ready": False,
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP B",
                "isin": "IT0000001002",
                "maturity_date": date(2031, 9, 1),
                "years_to_maturity": 5.0,
                "actual_yield_percent": 3.40,
                "fair_yield_percent": 3.46,
                "residual_bp": -6.0,
                "absolute_residual_bp": 6.0,
                "relative_value_label": "RICH",
                "curve_location": "BELLY",
                "fair_value_available": True,
                "interpolation_quality_score": 0.72,
                "interpolation_quality": "HIGH",
                "market_source": "Borsa Italiana / Skipper Informatica",
                "market_status": "PUBLIC_REFERENCE",
                "model_name": "LEAVE_ONE_OUT_LOCAL_LINEAR",
                "model_version": "LOCAL_LINEAR_PLAIN_FIXED_V1",
                "model_status": "REPOLENS_DERIVED",
                "history_observation_count": 30,
                "historical_mean_residual_bp": -1.0,
                "historical_std_residual_bp": 2.0,
                "historical_percentile": 3.0,
                "z_score": -2.5,
                "history_ready": True,
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP C",
                "isin": "IT0000001003",
                "maturity_date": date(2036, 9, 1),
                "years_to_maturity": 10.0,
                "actual_yield_percent": 4.10,
                "fair_yield_percent": 4.14,
                "residual_bp": -4.0,
                "absolute_residual_bp": 4.0,
                "relative_value_label": "RICH",
                "curve_location": "BELLY",
                "fair_value_available": True,
                "interpolation_quality_score": 0.45,
                "interpolation_quality": "MEDIUM",
                "market_source": "Borsa Italiana / Skipper Informatica",
                "market_status": "PUBLIC_REFERENCE",
                "model_name": "LEAVE_ONE_OUT_LOCAL_LINEAR",
                "model_version": "LOCAL_LINEAR_PLAIN_FIXED_V1",
                "model_status": "REPOLENS_DERIVED",
                "history_observation_count": 30,
                "historical_mean_residual_bp": 0.0,
                "historical_std_residual_bp": 1.0,
                "historical_percentile": 1.0,
                "z_score": -4.0,
                "history_ready": True,
            },
            {
                "country": "Italy",
                "currency": "EUR",
                "display_name": "BTP unavailable edge",
                "isin": "IT0000001004",
                "maturity_date": date(2060, 9, 1),
                "years_to_maturity": 34.0,
                "actual_yield_percent": 4.60,
                "fair_yield_percent": float("nan"),
                "residual_bp": float("nan"),
                "absolute_residual_bp": float("nan"),
                "relative_value_label": "UNAVAILABLE",
                "curve_location": "LONG_END",
                "fair_value_available": False,
                "interpolation_quality_score": float("nan"),
                "interpolation_quality": "UNAVAILABLE",
                "market_source": "Borsa Italiana / Skipper Informatica",
                "market_status": "PUBLIC_REFERENCE",
                "model_name": "LEAVE_ONE_OUT_LOCAL_LINEAR",
                "model_version": "LOCAL_LINEAR_PLAIN_FIXED_V1",
                "model_status": "REPOLENS_DERIVED",
                "history_observation_count": 0,
                "historical_mean_residual_bp": float("nan"),
                "historical_std_residual_bp": float("nan"),
                "historical_percentile": float("nan"),
                "z_score": float("nan"),
                "history_ready": False,
            },
        ]
    )


def test_local_curve_scanner_ranks_current_residuals_separately() -> None:
    scanner = build_local_curve_opportunity_scanner(
        create_local_curve_scanner_frame(),
        top_n=20,
    )

    assert list(
        scanner[
            "isin"
        ]
    ) == [
        "IT0000001001",
        "IT0000001002",
        "IT0000001003",
    ]

    assert list(
        scanner[
            "current_rank"
        ]
    ) == [
        1,
        2,
        3,
    ]


def test_local_curve_scanner_preserves_real_historical_ranks() -> None:
    scanner = build_local_curve_opportunity_scanner(
        create_local_curve_scanner_frame(),
        top_n=20,
    )

    by_isin = scanner.set_index(
        "isin"
    )

    assert pd.isna(
        by_isin.loc[
            "IT0000001001",
            "historical_rank",
        ]
    )

    assert by_isin.loc[
        "IT0000001003",
        "historical_rank",
    ] == 1

    assert by_isin.loc[
        "IT0000001002",
        "historical_rank",
    ] == 2


def test_local_curve_scanner_does_not_fake_z_score_without_history() -> None:
    scanner = build_local_curve_opportunity_scanner(
        create_local_curve_scanner_frame(),
        top_n=20,
    )

    first = scanner.loc[
        scanner[
            "isin"
        ].eq(
            "IT0000001001"
        )
    ].iloc[
        0
    ]

    assert first[
        "history_status"
    ] == "BUILDING_HISTORY"

    assert first[
        "opportunity_basis"
    ] == "LOCAL_CURVE_RESIDUAL_ONLY"

    assert pd.isna(
        first[
            "z_score"
        ]
    )

    assert pd.isna(
        first[
            "absolute_z_score"
        ]
    )


def test_historical_local_curve_scanner_orders_by_absolute_z() -> None:
    scanner = build_historical_local_curve_scanner(
        create_local_curve_scanner_frame(),
        top_n=20,
    )

    assert list(
        scanner[
            "isin"
        ]
    ) == [
        "IT0000001003",
        "IT0000001002",
    ]

    assert list(
        scanner[
            "historical_rank"
        ]
    ) == [
        1,
        2,
    ]


def test_historical_scanner_empty_while_history_is_building() -> None:
    frame = create_local_curve_scanner_frame()

    frame[
        "history_ready"
    ] = False

    frame[
        "z_score"
    ] = float(
        "nan"
    )

    scanner = build_historical_local_curve_scanner(
        frame,
        top_n=20,
    )

    assert scanner.empty


def test_local_curve_scanner_excludes_unavailable_fair_values() -> None:
    scanner = build_local_curve_opportunity_scanner(
        create_local_curve_scanner_frame(),
        top_n=20,
    )

    assert "IT0000001004" not in set(
        scanner[
            "isin"
        ]
    )


def test_history_ready_requires_real_z_score() -> None:
    frame = create_local_curve_scanner_frame()

    frame.loc[
        frame[
            "isin"
        ].eq(
            "IT0000001002"
        ),
        "z_score",
    ] = float(
        "nan"
    )

    with pytest.raises(
        SovereignOpportunityValidationError,
        match="history_ready observations",
    ):
        validate_local_curve_scanner_input(
            frame
        )


def test_local_curve_scanner_rejects_missing_required_columns() -> None:
    frame = create_local_curve_scanner_frame().drop(
        columns=[
            "model_version",
        ]
    )

    with pytest.raises(
        SovereignOpportunityValidationError,
        match="missing required columns",
    ):
        build_local_curve_opportunity_scanner(
            frame,
            top_n=20,
        )


def test_local_curve_scanner_top_n_is_respected() -> None:
    scanner = build_local_curve_opportunity_scanner(
        create_local_curve_scanner_frame(),
        top_n=2,
    )

    assert len(
        scanner
    ) == 2

    assert list(
        scanner[
            "current_rank"
        ]
    ) == [
        1,
        2,
    ]


def test_cross_sectional_z_score_excludes_target_from_peer_distribution() -> None:
    frame = create_local_curve_scanner_frame()

    scored = add_cross_sectional_z_scores(
        frame,
        minimum_peer_observations=2,
    )

    target = scored.loc[
        scored[
            "isin"
        ].eq(
            "IT0000001001"
        )
    ].iloc[
        0
    ]

    peer_values = np.array(
        [
            -6.0,
            -4.0,
        ]
    )

    expected_mean = float(
        np.mean(
            peer_values
        )
    )

    expected_std = float(
        np.std(
            peer_values,
            ddof=0,
        )
    )

    expected_z = (
        10.0
        - expected_mean
    ) / expected_std

    assert target[
        "whole_curve_peer_count"
    ] == 2

    assert target[
        "whole_curve_mean_residual_bp"
    ] == pytest.approx(
        expected_mean
    )

    assert target[
        "whole_curve_std_residual_bp"
    ] == pytest.approx(
        expected_std
    )

    assert target[
        "cross_sectional_z_score"
    ] == pytest.approx(
        expected_z
    )


def test_sector_z_score_uses_only_same_sector_peers() -> None:
    frame = create_local_curve_scanner_frame()

    extra = frame.iloc[
        [
            1
        ]
    ].copy()

    extra.loc[
        :,
        "isin",
    ] = "IT0000001005"

    extra.loc[
        :,
        "display_name",
    ] = "BTP D"

    extra.loc[
        :,
        "residual_bp",
    ] = 2.0

    extra.loc[
        :,
        "absolute_residual_bp",
    ] = 2.0

    extra.loc[
        :,
        "z_score",
    ] = float(
        "nan"
    )

    extra.loc[
        :,
        "history_ready",
    ] = False

    combined = pd.concat(
        [
            frame,
            extra,
        ],
        ignore_index=True,
    )

    scored = add_cross_sectional_z_scores(
        combined,
        minimum_peer_observations=2,
    )

    target = scored.loc[
        scored[
            "isin"
        ].eq(
            "IT0000001003"
        )
    ].iloc[
        0
    ]

    # BTP C is BELLY. Its sector peers are BTP B (-6) and BTP D (+2).
    expected_mean = -2.0
    expected_std = 4.0
    expected_z = (
        -4.0
        - expected_mean
    ) / expected_std

    assert target[
        "sector_peer_count"
    ] == 2

    assert target[
        "sector_mean_residual_bp"
    ] == pytest.approx(
        expected_mean
    )

    assert target[
        "sector_std_residual_bp"
    ] == pytest.approx(
        expected_std
    )

    assert target[
        "sector_z_score"
    ] == pytest.approx(
        expected_z
    )


def test_cross_sectional_z_score_requires_minimum_peer_count() -> None:
    scored = add_cross_sectional_z_scores(
        create_local_curve_scanner_frame(),
        minimum_peer_observations=3,
    )

    front_end = scored.loc[
        scored[
            "isin"
        ].eq(
            "IT0000001001"
        )
    ].iloc[
        0
    ]

    assert front_end[
        "whole_curve_peer_count"
    ] == 2

    assert pd.isna(
        front_end[
            "cross_sectional_z_score"
        ]
    )


def test_scanner_exposes_immediate_cross_sectional_z_scores() -> None:
    frame = create_local_curve_scanner_frame()

    # Add enough available peer observations for the default minimum of 3.
    extra = frame.iloc[
        [
            1
        ]
    ].copy()

    extra.loc[
        :,
        "isin",
    ] = "IT0000001005"

    extra.loc[
        :,
        "display_name",
    ] = "BTP D"

    extra.loc[
        :,
        "residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "absolute_residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "z_score",
    ] = float(
        "nan"
    )

    extra.loc[
        :,
        "history_ready",
    ] = False

    combined = pd.concat(
        [
            frame,
            extra,
        ],
        ignore_index=True,
    )

    scanner = build_local_curve_opportunity_scanner(
        combined,
        top_n=20,
    )

    assert "cross_sectional_z_score" in scanner.columns
    assert "sector_z_score" in scanner.columns
    assert "cross_sectional_rank" in scanner.columns
    assert "sector_rank" in scanner.columns

    assert scanner[
        "cross_sectional_z_score"
    ].notna().all()


def test_selected_local_curve_bond_returns_one_scanner_row() -> None:
    frame = create_local_curve_scanner_frame()

    extra = frame.iloc[
        [
            1
        ]
    ].copy()

    extra.loc[
        :,
        "isin",
    ] = "IT0000001005"

    extra.loc[
        :,
        "display_name",
    ] = "BTP D"

    extra.loc[
        :,
        "residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "absolute_residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "history_ready",
    ] = False

    extra.loc[
        :,
        "z_score",
    ] = float(
        "nan"
    )

    scanner = build_local_curve_opportunity_scanner(
        pd.concat(
            [
                frame,
                extra,
            ],
            ignore_index=True,
        ),
        top_n=20,
    )

    selected = selected_local_curve_bond(
        scanner,
        isin="it0000001002",
    )

    assert selected[
        "isin"
    ] == "IT0000001002"

    assert pd.notna(
        selected[
            "cross_sectional_z_score"
        ]
    )


def test_selected_local_curve_bond_rejects_missing_isin() -> None:
    frame = create_local_curve_scanner_frame()

    extra = frame.iloc[
        [
            1
        ]
    ].copy()

    extra.loc[
        :,
        "isin",
    ] = "IT0000001005"

    extra.loc[
        :,
        "display_name",
    ] = "BTP D"

    extra.loc[
        :,
        "residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "absolute_residual_bp",
    ] = 1.0

    extra.loc[
        :,
        "history_ready",
    ] = False

    extra.loc[
        :,
        "z_score",
    ] = float(
        "nan"
    )

    scanner = build_local_curve_opportunity_scanner(
        pd.concat(
            [
                frame,
                extra,
            ],
            ignore_index=True,
        ),
        top_n=20,
    )

    with pytest.raises(
        SovereignOpportunityValidationError,
        match="Expected exactly one",
    ):
        selected_local_curve_bond(
            scanner,
            isin="IT0000001999",
        )