from datetime import date

import pytest

from src.repo_analytics import (
    RepoTradeInput,
    RepoValidationError,
    calculate_repo_trade,
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