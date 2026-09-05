import pytest

from src.repo_adjusted_relative_value import (
    RepoAdjustedRelativeValueValidationError,
    RepoFundingLegInput,
    SpreadScenarioPoint,
    analyse_funding_adjusted_spread_breakeven,
    analyse_repo_adjusted_relative_value,
    analyse_repo_funding_leg,
)
from src.sovereign_relative_value import PositionDirection


def make_leg(
    *,
    isin: str,
    direction: PositionDirection,
    face_value_eur: float = 10_000_000.0,
    specific_repo_rate_percent: float = 1.50,
    gc_repo_rate_percent: float = 2.00,
    repo_days: int = 30,
    day_count_basis: int = 360,
) -> RepoFundingLegInput:
    return RepoFundingLegInput(
        isin=isin,
        direction=direction,
        face_value_eur=face_value_eur,
        dirty_price_per_100=100.0,
        haircut_percent=0.0,
        specific_repo_rate_percent=specific_repo_rate_percent,
        gc_repo_rate_percent=gc_repo_rate_percent,
        repo_days=repo_days,
        day_count_basis=day_count_basis,
    )


def test_long_special_collateral_has_positive_signed_funding_impact() -> None:
    result = analyse_repo_funding_leg(
        make_leg(
            isin="DE0000000001",
            direction=PositionDirection.LONG,
        )
    )

    assert result.specialness_bp == pytest.approx(
        50.0
    )

    assert (
        result.unsigned_financing_edge_vs_gc_eur
        > 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        > 0.0
    )


def test_short_special_collateral_reverses_the_funding_sign() -> None:
    result = analyse_repo_funding_leg(
        make_leg(
            isin="DE0000000002",
            direction=PositionDirection.SHORT,
        )
    )

    assert (
        result.unsigned_financing_edge_vs_gc_eur
        > 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        < 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        == pytest.approx(
            -result.unsigned_financing_edge_vs_gc_eur
        )
    )


def test_long_special_anchor_vs_gc_hedge_has_positive_net_overlay() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=1.00,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=2.00,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.anchor.specialness_bp
        == pytest.approx(
            100.0
        )
    )

    assert (
        analysis.hedge.specialness_bp
        == pytest.approx(
            0.0
        )
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        > 0.0
    )

    assert (
        analysis.anchor_minus_hedge_specialness_bp
        == pytest.approx(
            100.0
        )
    )


def test_shorting_more_special_anchor_produces_negative_net_overlay() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=1.00,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=2.00,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        < 0.0
    )


def test_equal_specialness_and_equal_notional_offset_for_opposite_directions() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=1.50,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=1.50,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        == pytest.approx(
            0.0,
            abs=1e-9,
        )
    )

    assert (
        analysis.anchor_minus_hedge_specialness_bp
        == pytest.approx(
            0.0
        )
    )


def test_dv01_sized_unequal_notionals_are_preserved() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            face_value_eur=8_000_000.0,
            specific_repo_rate_percent=1.25,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            face_value_eur=10_000_000.0,
            specific_repo_rate_percent=1.90,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert analysis.anchor.face_value_eur == pytest.approx(
        8_000_000.0
    )

    assert analysis.hedge.face_value_eur == pytest.approx(
        10_000_000.0
    )

    expected_net = (
        analysis.anchor.signed_financing_impact_vs_gc_eur
        + analysis.hedge.signed_financing_impact_vs_gc_eur
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        == pytest.approx(
            expected_net
        )
    )


def test_repo_term_mismatch_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="repo_days must match",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
                repo_days=7,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.SHORT,
                repo_days=30,
            ),
        )


def test_day_count_mismatch_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="day_count_basis must match",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
                day_count_basis=360,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.SHORT,
                day_count_basis=365,
            ),
        )


def test_same_direction_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="directions must be opposite",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.LONG,
            ),
        )


def test_same_instrument_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="different instruments",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
            ),
            hedge=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.SHORT,
            ),
        )



def make_linear_spread_scenarios(
    pnl_per_bp_eur: float,
) -> tuple[SpreadScenarioPoint, ...]:
    return tuple(
        SpreadScenarioPoint(
            spread_shock_bp=shock,
            cash_pnl_eur=(
                shock
                * pnl_per_bp_eur
            ),
        )
        for shock in (
            -25.0,
            -10.0,
            -5.0,
            5.0,
            10.0,
            25.0,
        )
    )


def test_positive_funding_edge_creates_adverse_spread_capacity() -> None:
    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=100.0,
        funding_overlay_eur=10_000.0,
        scenario_points=make_linear_spread_scenarios(
            pnl_per_bp_eur=-5_000.0
        ),
    )

    assert breakeven.interpretation == "ADVERSE_MOVE_CAPACITY"
    assert breakeven.direction == "WIDENING"
    assert breakeven.first_order_equivalent_spread_move_bp == pytest.approx(
        2.0
    )
    assert breakeven.breakeven_spread_shock_bp == pytest.approx(
        2.0
    )
    assert breakeven.breakeven_spread_level_bp == pytest.approx(
        102.0
    )
    assert breakeven.breakeven_move_magnitude_bp == pytest.approx(
        2.0
    )
    assert breakeven.within_scenario_range is True


def test_positive_funding_edge_can_protect_against_narrowing() -> None:
    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=40.0,
        funding_overlay_eur=8_000.0,
        scenario_points=make_linear_spread_scenarios(
            pnl_per_bp_eur=4_000.0
        ),
    )

    assert breakeven.direction == "NARROWING"
    assert breakeven.breakeven_spread_shock_bp == pytest.approx(
        -2.0
    )
    assert breakeven.breakeven_spread_level_bp == pytest.approx(
        38.0
    )


def test_negative_funding_overlay_reports_favourable_move_required() -> None:
    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=100.0,
        funding_overlay_eur=-15_000.0,
        scenario_points=make_linear_spread_scenarios(
            pnl_per_bp_eur=-5_000.0
        ),
    )

    assert breakeven.interpretation == "FAVOURABLE_MOVE_REQUIRED"
    assert breakeven.direction == "NARROWING"
    assert breakeven.breakeven_spread_shock_bp == pytest.approx(
        -3.0
    )
    assert breakeven.breakeven_spread_level_bp == pytest.approx(
        97.0
    )


def test_zero_funding_overlay_has_zero_spread_breakeven_shift() -> None:
    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=75.0,
        funding_overlay_eur=0.0,
        scenario_points=make_linear_spread_scenarios(
            pnl_per_bp_eur=-5_000.0
        ),
    )

    assert breakeven.direction == "NONE"
    assert breakeven.interpretation == (
        "No repo funding edge or drag versus GC is present."
    )
    assert breakeven.breakeven_spread_shock_bp == pytest.approx(
        0.0
    )
    assert breakeven.breakeven_spread_level_bp == pytest.approx(
        75.0
    )


def test_non_linear_curve_uses_bracket_interpolation_not_only_dv01() -> None:
    scenarios = (
        SpreadScenarioPoint(
            spread_shock_bp=-10.0,
            cash_pnl_eur=60_000.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=-5.0,
            cash_pnl_eur=28_000.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=5.0,
            cash_pnl_eur=-22_000.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=10.0,
            cash_pnl_eur=-40_000.0,
        ),
    )

    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=110.0,
        funding_overlay_eur=11_000.0,
        scenario_points=scenarios,
    )

    assert breakeven.direction == "WIDENING"
    assert breakeven.breakeven_spread_shock_bp == pytest.approx(
        2.5
    )
    assert breakeven.breakeven_spread_level_bp == pytest.approx(
        112.5
    )
    assert breakeven.interpolation_width_bp == pytest.approx(
        5.0
    )


def test_breakeven_outside_supplied_scenario_range_is_not_extrapolated() -> None:
    breakeven = analyse_funding_adjusted_spread_breakeven(
        current_spread_bp=100.0,
        funding_overlay_eur=200_000.0,
        scenario_points=make_linear_spread_scenarios(
            pnl_per_bp_eur=-5_000.0
        ),
    )

    assert breakeven.within_scenario_range is False
    assert breakeven.breakeven_spread_shock_bp is None
    assert breakeven.breakeven_spread_level_bp is None
    assert breakeven.breakeven_move_magnitude_bp is None


def test_zero_shock_nonzero_cash_pnl_is_rejected() -> None:
    scenarios = (
        SpreadScenarioPoint(
            spread_shock_bp=-5.0,
            cash_pnl_eur=25_000.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=0.0,
            cash_pnl_eur=1.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=5.0,
            cash_pnl_eur=-25_000.0,
        ),
    )

    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="zero-shock cash P&L",
    ):
        analyse_funding_adjusted_spread_breakeven(
            current_spread_bp=100.0,
            funding_overlay_eur=5_000.0,
            scenario_points=scenarios,
        )


def test_spread_breakeven_requires_scenarios_on_both_sides_of_zero() -> None:
    scenarios = (
        SpreadScenarioPoint(
            spread_shock_bp=5.0,
            cash_pnl_eur=-25_000.0,
        ),
        SpreadScenarioPoint(
            spread_shock_bp=10.0,
            cash_pnl_eur=-50_000.0,
        ),
    )

    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="both negative and positive",
    ):
        analyse_funding_adjusted_spread_breakeven(
            current_spread_bp=100.0,
            funding_overlay_eur=5_000.0,
            scenario_points=scenarios,
        )