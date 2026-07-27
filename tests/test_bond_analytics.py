from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.bond_analytics import (
    BondValidationError,
    FixedRateBond,
    accrued_interest,
    build_coupon_dates,
    calculate_bond_risk_metrics,
    clean_price_from_yield,
    dirty_price_from_yield,
    dv01_per_100,
    future_cash_flows,
    position_dv01,
    run_parallel_yield_scenarios,
    scenario_results_to_frame,
    yield_from_clean_price,
)


def create_annual_bond() -> FixedRateBond:
    """
    Create a deterministic annual-coupon test bond.
    """
    return FixedRateBond(
        isin="DE000TEST001",
        issuer="Federal Republic of Germany",
        maturity_date=date(
            2030,
            1,
            15,
        ),
        annual_coupon_rate=0.03,
        coupon_frequency=1,
        face_value=100.0,
        currency="EUR",
    )


def create_semiannual_bond() -> FixedRateBond:
    """
    Create a deterministic semi-annual test bond.
    """
    return FixedRateBond(
        isin="IT000TEST001",
        issuer="Republic of Italy",
        maturity_date=date(
            2032,
            6,
            1,
        ),
        annual_coupon_rate=0.04,
        coupon_frequency=2,
        face_value=100.0,
        currency="EUR",
    )


def test_invalid_coupon_frequency_is_rejected() -> None:
    with pytest.raises(
        BondValidationError,
        match="coupon_frequency",
    ):
        FixedRateBond(
            isin="DE000TEST002",
            issuer="Federal Republic of Germany",
            maturity_date=date(
                2030,
                1,
                15,
            ),
            annual_coupon_rate=0.03,
            coupon_frequency=3,
        )


def test_settlement_on_or_after_maturity_is_rejected() -> None:
    bond = create_annual_bond()

    with pytest.raises(
        BondValidationError,
        match="before maturity",
    ):
        dirty_price_from_yield(
            bond=bond,
            settlement_date=date(
                2030,
                1,
                15,
            ),
            yield_to_maturity=0.03,
        )


def test_coupon_schedule_contains_maturity() -> None:
    bond = create_annual_bond()

    schedule = build_coupon_dates(
        bond=bond,
        settlement_date=date(
            2026,
            1,
            15,
        ),
    )

    assert bond.maturity_date in schedule

    assert schedule == sorted(
        schedule
    )


def test_future_cash_flows_include_principal_at_maturity() -> None:
    bond = create_annual_bond()

    cash_flows = future_cash_flows(
        bond=bond,
        settlement_date=date(
            2026,
            1,
            15,
        ),
    )

    final_cash_flow = cash_flows[-1]

    assert (
        final_cash_flow.payment_date
        == bond.maturity_date
    )

    assert final_cash_flow.coupon_amount == pytest.approx(
        3.0
    )

    assert final_cash_flow.principal_amount == pytest.approx(
        100.0
    )

    assert final_cash_flow.total_amount == pytest.approx(
        103.0
    )


def test_accrued_interest_is_zero_on_coupon_date() -> None:
    bond = create_annual_bond()

    accrued = accrued_interest(
        bond=bond,
        settlement_date=date(
            2026,
            1,
            15,
        ),
    )

    assert accrued == pytest.approx(
        0.0
    )


def test_accrued_interest_is_positive_between_coupons() -> None:
    bond = create_annual_bond()

    accrued = accrued_interest(
        bond=bond,
        settlement_date=date(
            2026,
            7,
            15,
        ),
    )

    assert accrued > 0.0
    assert accrued < 3.0


def test_clean_plus_accrued_equals_dirty_price() -> None:
    bond = create_semiannual_bond()

    settlement_date = date(
        2026,
        7,
        17,
    )

    yield_to_maturity = 0.0375

    dirty_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    clean_price = clean_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    accrued = accrued_interest(
        bond=bond,
        settlement_date=settlement_date,
    )

    assert dirty_price == pytest.approx(
        clean_price + accrued,
        abs=1e-10,
    )


def test_coupon_rate_equal_to_yield_is_near_par_on_coupon_date() -> None:
    bond = create_annual_bond()

    clean_price = clean_price_from_yield(
        bond=bond,
        settlement_date=date(
            2026,
            1,
            15,
        ),
        yield_to_maturity=0.03,
    )

    assert clean_price == pytest.approx(
        100.0,
        abs=0.05,
    )


def test_higher_yield_produces_lower_price() -> None:
    bond = create_semiannual_bond()

    settlement_date = date(
        2026,
        7,
        17,
    )

    lower_yield_price = clean_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=0.03,
    )

    higher_yield_price = clean_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=0.05,
    )

    assert higher_yield_price < lower_yield_price


def test_yield_solver_recovers_original_yield() -> None:
    bond = create_semiannual_bond()

    settlement_date = date(
        2026,
        7,
        17,
    )

    original_yield = 0.0425

    clean_price = clean_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=original_yield,
    )

    solved_yield = yield_from_clean_price(
        bond=bond,
        settlement_date=settlement_date,
        clean_price=clean_price,
    )

    assert solved_yield == pytest.approx(
        original_yield,
        abs=1e-10,
    )


def test_risk_metrics_are_positive_and_consistent() -> None:
    bond = create_semiannual_bond()

    metrics = calculate_bond_risk_metrics(
        bond=bond,
        settlement_date=date(
            2026,
            7,
            17,
        ),
        yield_to_maturity=0.04,
    )

    assert metrics.clean_price > 0.0
    assert metrics.dirty_price > 0.0
    assert metrics.accrued_interest >= 0.0
    assert metrics.macaulay_duration > 0.0
    assert metrics.modified_duration > 0.0
    assert metrics.dv01_per_100 > 0.0
    assert metrics.convexity > 0.0

    assert (
        metrics.dirty_price
        == pytest.approx(
            metrics.clean_price
            + metrics.accrued_interest
        )
    )


def test_dv01_matches_central_price_difference() -> None:
    bond = create_annual_bond()

    settlement_date = date(
        2026,
        1,
        15,
    )

    yield_to_maturity = 0.03

    calculated_dv01 = dv01_per_100(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    price_down_one_bp = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=0.0299,
    )

    price_up_one_bp = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=0.0301,
    )

    expected_dv01 = (
        price_down_one_bp
        - price_up_one_bp
    ) / 2.0

    assert calculated_dv01 == pytest.approx(
        expected_dv01,
        abs=1e-12,
    )


def test_position_dv01_scales_by_notional() -> None:
    scaled_dv01 = position_dv01(
        dv01_per_100_value=0.075,
        position_notional=10_000_000.0,
    )

    assert scaled_dv01 == pytest.approx(
        7_500.0
    )


def test_parallel_scenarios_have_correct_price_direction() -> None:
    bond = create_semiannual_bond()

    scenarios = run_parallel_yield_scenarios(
        bond=bond,
        settlement_date=date(
            2026,
            7,
            17,
        ),
        yield_to_maturity=0.04,
        position_notional=10_000_000.0,
        yield_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    yield_fall = scenarios[0]
    yield_rise = scenarios[1]

    assert yield_fall.clean_price_change > 0.0
    assert yield_fall.position_pnl > 0.0

    assert yield_rise.clean_price_change < 0.0
    assert yield_rise.position_pnl < 0.0


def test_scenario_frame_contains_expected_columns() -> None:
    bond = create_annual_bond()

    scenarios = run_parallel_yield_scenarios(
        bond=bond,
        settlement_date=date(
            2026,
            1,
            15,
        ),
        yield_to_maturity=0.03,
        position_notional=5_000_000.0,
        yield_shocks_bp=(
            -5.0,
            5.0,
        ),
    )

    frame = scenario_results_to_frame(
        scenarios
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert list(
        frame.columns
    ) == [
        "yield_shock_bp",
        "shocked_yield",
        "shocked_clean_price",
        "clean_price_change",
        "position_pnl",
    ]

    assert len(
        frame
    ) == 2


def test_negative_position_notional_is_rejected() -> None:
    bond = create_annual_bond()

    with pytest.raises(
        BondValidationError,
        match="position_notional",
    ):
        run_parallel_yield_scenarios(
            bond=bond,
            settlement_date=date(
                2026,
                1,
                15,
            ),
            yield_to_maturity=0.03,
            position_notional=-1_000_000.0,
        )