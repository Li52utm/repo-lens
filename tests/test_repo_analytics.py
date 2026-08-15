from datetime import date

import pytest

from src.bond_analytics import FixedRateBond
from src.repo_analytics import (
    RepoTradeInput,
    RepoValidationError,
    analyse_discount_security_carry_to_maturity,
    analyse_financed_bond_carry,
    calculate_repo_trade,
    coupon_income_between_dates,
    required_collateral_market_value,
    required_face_value,
)


def test_repo_trade_without_haircut() -> None:
    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.0,
        repo_rate_percent=2.0,
        haircut_percent=0.0,
        purchase_date=date(2026, 8, 15),
        repurchase_date=date(2026, 8, 22),
        day_count_basis=360,
    )

    result = calculate_repo_trade(
        trade
    )

    assert result.repo_days == 7
    assert result.collateral_market_value_eur == pytest.approx(
        10_000_000.0
    )
    assert result.purchase_price_eur == pytest.approx(
        10_000_000.0
    )
    assert result.repo_interest_eur == pytest.approx(
        3_888.888888888889
    )
    assert result.repurchase_price_eur == pytest.approx(
        10_003_888.88888889
    )


def test_repo_trade_with_two_percent_haircut() -> None:
    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=99.0,
        accrued_interest_per_100=1.0,
        repo_rate_percent=2.5,
        haircut_percent=2.0,
        purchase_date=date(2026, 8, 15),
        repurchase_date=date(2026, 8, 16),
    )

    result = calculate_repo_trade(
        trade
    )

    assert result.dirty_price_per_100 == pytest.approx(
        100.0
    )
    assert result.collateral_market_value_eur == pytest.approx(
        10_000_000.0
    )
    assert result.haircut_amount_eur == pytest.approx(
        200_000.0
    )
    assert result.purchase_price_eur == pytest.approx(
        9_800_000.0
    )


def test_negative_repo_rate_is_supported() -> None:
    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.0,
        repo_rate_percent=-0.5,
        haircut_percent=0.0,
        purchase_date=date(2026, 8, 15),
        repurchase_date=date(2026, 8, 22),
    )

    result = calculate_repo_trade(
        trade
    )

    assert result.repo_interest_eur < 0.0
    assert result.repurchase_price_eur < result.purchase_price_eur


def test_specialness_and_financing_benefit() -> None:
    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.0,
        repo_rate_percent=1.5,
        haircut_percent=0.0,
        purchase_date=date(2026, 8, 15),
        repurchase_date=date(2026, 8, 22),
        gc_repo_rate_percent=2.0,
    )

    result = calculate_repo_trade(
        trade
    )

    assert result.specialness_bp == pytest.approx(
        50.0
    )
    assert result.financing_benefit_vs_gc_eur == pytest.approx(
        972.2222222222222
    )


def test_cash_to_collateral_sizing() -> None:
    market_value = required_collateral_market_value(
        target_cash_eur=10_000_000.0,
        haircut_percent=2.0,
    )

    face_value = required_face_value(
        target_cash_eur=10_000_000.0,
        dirty_price=100.0,
        haircut_percent=2.0,
    )

    assert market_value == pytest.approx(
        10_204_081.632653061
    )
    assert face_value == pytest.approx(
        10_204_081.632653061
    )


def test_invalid_dates_fail() -> None:
    with pytest.raises(
        RepoValidationError
    ):
        RepoTradeInput(
            face_value_eur=10_000_000.0,
            clean_price_per_100=100.0,
            accrued_interest_per_100=0.0,
            repo_rate_percent=2.0,
            haircut_percent=0.0,
            purchase_date=date(2026, 8, 15),
            repurchase_date=date(2026, 8, 15),
        )


def test_coupon_income_between_dates() -> None:
    bond = FixedRateBond(
        isin="TESTBOND0001",
        issuer="Test Issuer",
        maturity_date=date(2028, 1, 1),
        annual_coupon_rate=0.04,
        coupon_frequency=2,
    )

    coupon_income = coupon_income_between_dates(
        bond=bond,
        purchase_date=date(2026, 5, 1),
        repurchase_date=date(2026, 8, 1),
        position_face_value_eur=10_000_000.0,
    )

    assert coupon_income == pytest.approx(
        200_000.0
    )


def test_financed_bond_carry_has_unchanged_yield_scenario() -> None:
    bond = FixedRateBond(
        isin="TESTBOND0001",
        issuer="Test Issuer",
        maturity_date=date(2030, 1, 1),
        annual_coupon_rate=0.03,
        coupon_frequency=2,
    )

    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.75,
        repo_rate_percent=2.0,
        haircut_percent=0.0,
        purchase_date=date(2026, 4, 1),
        repurchase_date=date(2026, 5, 1),
    )

    analysis = analyse_financed_bond_carry(
        bond=bond,
        trade=trade,
        yield_shocks_bp=(
            -10.0,
            0.0,
            10.0,
        ),
    )

    unchanged = next(
        scenario
        for scenario in analysis.scenarios
        if scenario.yield_shock_bp == 0.0
    )

    assert analysis.start_yield_percent == pytest.approx(
        3.0,
        abs=0.05,
    )

    assert unchanged.repo_interest_eur > 0.0
    assert unchanged.financing_adjusted_pnl_per_eur_1m_face == pytest.approx(
        unchanged.financing_adjusted_pnl_eur
        / 10.0
    )


def test_financed_bond_carry_falls_when_yield_rises() -> None:
    bond = FixedRateBond(
        isin="TESTBOND0001",
        issuer="Test Issuer",
        maturity_date=date(2030, 1, 1),
        annual_coupon_rate=0.03,
        coupon_frequency=2,
    )

    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.75,
        repo_rate_percent=2.0,
        haircut_percent=0.0,
        purchase_date=date(2026, 4, 1),
        repurchase_date=date(2026, 5, 1),
    )

    analysis = analyse_financed_bond_carry(
        bond=bond,
        trade=trade,
        yield_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    yield_down = analysis.scenarios[0]
    yield_up = analysis.scenarios[1]

    assert (
        yield_down.financing_adjusted_pnl_eur
        > yield_up.financing_adjusted_pnl_eur
    )


def test_bond_carry_rejects_repo_through_maturity() -> None:
    bond = FixedRateBond(
        isin="TESTBOND0001",
        issuer="Test Issuer",
        maturity_date=date(2026, 8, 20),
        annual_coupon_rate=0.03,
        coupon_frequency=2,
    )

    trade = RepoTradeInput(
        face_value_eur=10_000_000.0,
        clean_price_per_100=100.0,
        accrued_interest_per_100=0.0,
        repo_rate_percent=2.0,
        haircut_percent=0.0,
        purchase_date=date(2026, 8, 15),
        repurchase_date=date(2026, 8, 20),
    )

    with pytest.raises(
        RepoValidationError,
        match="before maturity_date",
    ):
        analyse_financed_bond_carry(
            bond=bond,
            trade=trade,
        )


def test_discount_security_financing_to_maturity() -> None:
    analysis = analyse_discount_security_carry_to_maturity(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
        redemption_value_per_100=100.0,
        purchase_date=date(2026, 8, 15),
        maturity_date=date(2026, 11, 18),
        repo_rate_percent=1.60,
        haircut_percent=0.50,
        day_count_basis=360,
        gc_repo_rate_percent=2.15,
    )

    assert analysis.days_to_maturity == 95
    assert analysis.start_market_value_eur == pytest.approx(
        9_940_000.0
    )
    assert analysis.redemption_value_eur == pytest.approx(
        10_000_000.0
    )
    assert analysis.gross_pull_to_par_eur == pytest.approx(
        60_000.0
    )
    assert analysis.financing_cost_to_maturity_eur > 0.0
    assert (
        analysis.financing_adjusted_pull_to_par_eur
        < analysis.gross_pull_to_par_eur
    )
    assert analysis.breakeven_repo_rate_percent > 0.0
    assert analysis.financing_benefit_vs_gc_to_maturity_eur > 0.0


def test_discount_security_breakeven_repo_rate_zeroes_net_carry() -> None:
    base = analyse_discount_security_carry_to_maturity(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
        redemption_value_per_100=100.0,
        purchase_date=date(2026, 8, 15),
        maturity_date=date(2026, 11, 18),
        repo_rate_percent=1.60,
        haircut_percent=0.50,
        day_count_basis=360,
    )

    breakeven = analyse_discount_security_carry_to_maturity(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
        redemption_value_per_100=100.0,
        purchase_date=date(2026, 8, 15),
        maturity_date=date(2026, 11, 18),
        repo_rate_percent=base.breakeven_repo_rate_percent,
        haircut_percent=0.50,
        day_count_basis=360,
    )

    assert breakeven.financing_adjusted_pull_to_par_eur == pytest.approx(
        0.0,
        abs=1e-6,
    )