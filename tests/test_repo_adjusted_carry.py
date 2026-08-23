from datetime import date

import pytest

from src.bond_analytics import FixedRateBond
from src.repo_adjusted_carry import (
    RepoAdjustedCarryValidationError,
    analyse_repo_adjusted_bond_carry,
)
from src.repo_analytics import RepoTradeInput


def make_test_bond() -> FixedRateBond:
    return FixedRateBond(
        isin="TESTBOND0001",
        issuer="Test Issuer",
        maturity_date=date(2030, 1, 1),
        annual_coupon_rate=0.03,
        coupon_frequency=2,
    )


def make_test_trade(
    *,
    repo_rate_percent: float = 1.50,
) -> RepoTradeInput:
    return RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.75,
        repo_rate_percent=repo_rate_percent,
        haircut_percent=2.0,
        purchase_date=date(2026, 4, 1),
        repurchase_date=date(2026, 5, 1),
        day_count_basis=360,
    )


def test_special_funding_improves_unchanged_yield_carry() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(repo_rate_percent=1.50),
        gc_repo_rate_percent=2.00,
    )

    assert analysis.specialness_bp == pytest.approx(50.0)
    assert (
        analysis.unchanged_yield_specific_pnl_eur
        > analysis.unchanged_yield_gc_pnl_eur
    )
    assert analysis.financing_advantage_vs_gc_eur > 0.0


def test_financing_advantage_equals_interest_cost_difference() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(),
        gc_repo_rate_percent=2.00,
    )

    expected = (
        analysis.gc_repo_interest_eur
        - analysis.specific_repo_interest_eur
    )

    assert analysis.financing_advantage_vs_gc_eur == pytest.approx(
        expected
    )

    assert (
        analysis.unchanged_yield_specific_pnl_eur
        - analysis.unchanged_yield_gc_pnl_eur
    ) == pytest.approx(expected)


def test_equal_specific_and_gc_rates_have_zero_financing_advantage() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(repo_rate_percent=2.00),
        gc_repo_rate_percent=2.00,
    )

    assert analysis.specialness_bp == pytest.approx(0.0)
    assert analysis.financing_advantage_vs_gc_eur == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert analysis.unchanged_yield_specific_pnl_eur == pytest.approx(
        analysis.unchanged_yield_gc_pnl_eur
    )


def test_repo_rate_shock_isolates_funding_sensitivity() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(repo_rate_percent=1.50),
        gc_repo_rate_percent=2.00,
        repo_rate_shocks_bp=(-25.0, 0.0, 25.0),
    )

    down, base, up = analysis.repo_rate_scenarios

    assert down.shocked_specific_repo_rate_percent == pytest.approx(1.25)
    assert base.shocked_specific_repo_rate_percent == pytest.approx(1.50)
    assert up.shocked_specific_repo_rate_percent == pytest.approx(1.75)

    assert (
        down.financing_adjusted_pnl_eur
        > base.financing_adjusted_pnl_eur
        > up.financing_adjusted_pnl_eur
    )

    assert down.specialness_vs_gc_bp == pytest.approx(75.0)
    assert base.specialness_vs_gc_bp == pytest.approx(50.0)
    assert up.specialness_vs_gc_bp == pytest.approx(25.0)


def test_repo_rate_shock_financing_advantage_matches_gc_difference() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(repo_rate_percent=1.50),
        gc_repo_rate_percent=2.00,
        repo_rate_shocks_bp=(0.0,),
    )

    scenario = analysis.repo_rate_scenarios[0]

    assert scenario.financing_advantage_vs_gc_eur == pytest.approx(
        analysis.financing_advantage_vs_gc_eur
    )

    assert (
        scenario.financing_adjusted_pnl_eur
        - analysis.unchanged_yield_gc_pnl_eur
    ) == pytest.approx(
        analysis.financing_advantage_vs_gc_eur
    )


def test_special_funding_changes_breakeven_yield_in_expected_direction() -> None:
    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=make_test_trade(repo_rate_percent=1.00),
        gc_repo_rate_percent=2.00,
    )

    assert analysis.specific_breakeven_exit_yield_percent is not None
    assert analysis.gc_breakeven_exit_yield_percent is not None
    assert analysis.breakeven_yield_advantage_bp is not None

    assert (
        analysis.specific_breakeven_exit_yield_percent
        > analysis.gc_breakeven_exit_yield_percent
    )

    assert analysis.breakeven_yield_advantage_bp > 0.0


def test_embedded_gc_on_trade_does_not_change_comparison() -> None:
    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.75,
        repo_rate_percent=1.50,
        haircut_percent=2.0,
        purchase_date=date(2026, 4, 1),
        repurchase_date=date(2026, 5, 1),
        day_count_basis=360,
        gc_repo_rate_percent=9.99,
    )

    analysis = analyse_repo_adjusted_bond_carry(
        bond=make_test_bond(),
        trade=trade,
        gc_repo_rate_percent=2.00,
        repo_rate_shocks_bp=(0.0,),
    )

    assert analysis.gc_repo_rate_percent == pytest.approx(2.00)
    assert analysis.specialness_bp == pytest.approx(50.0)


def test_duplicate_repo_rate_shocks_are_rejected() -> None:
    with pytest.raises(
        RepoAdjustedCarryValidationError,
        match="duplicate",
    ):
        analyse_repo_adjusted_bond_carry(
            bond=make_test_bond(),
            trade=make_test_trade(),
            gc_repo_rate_percent=2.00,
            repo_rate_shocks_bp=(0.0, 0.0),
        )


def test_empty_repo_rate_shocks_are_rejected() -> None:
    with pytest.raises(
        RepoAdjustedCarryValidationError,
        match="must not be empty",
    ):
        analyse_repo_adjusted_bond_carry(
            bond=make_test_bond(),
            trade=make_test_trade(),
            gc_repo_rate_percent=2.00,
            repo_rate_shocks_bp=(),
        )