from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import brentq


DEFAULT_FACE_VALUE: Final[float] = 100.0
DEFAULT_FREQUENCY: Final[int] = 1
DEFAULT_YIELD_SHOCKS_BP: Final[tuple[float, ...]] = (
    -25.0,
    -10.0,
    -5.0,
    -1.0,
    1.0,
    5.0,
    10.0,
    25.0,
)


class BondAnalyticsError(RuntimeError):
    """
    Base exception for RepoLens bond analytics.
    """


class BondValidationError(BondAnalyticsError):
    """
    Raised when a bond definition or market input is invalid.
    """


class YieldSolveError(BondAnalyticsError):
    """
    Raised when yield cannot be solved from the supplied price.
    """


@dataclass(frozen=True)
class FixedRateBond:
    """
    Define a plain fixed-rate sovereign bond.

    Coupon rates and yields are expressed as decimal values.

    Examples:
        0.025 means 2.50%.
        0.031 means 3.10%.
    """

    isin: str
    issuer: str
    maturity_date: date
    annual_coupon_rate: float
    coupon_frequency: int = DEFAULT_FREQUENCY
    face_value: float = DEFAULT_FACE_VALUE
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if not self.isin.strip():
            raise BondValidationError(
                "isin must not be empty."
            )

        if not self.issuer.strip():
            raise BondValidationError(
                "issuer must not be empty."
            )

        if self.annual_coupon_rate < 0.0:
            raise BondValidationError(
                "annual_coupon_rate must not be negative."
            )

        if self.coupon_frequency not in {
            1,
            2,
            4,
        }:
            raise BondValidationError(
                "coupon_frequency must be 1, 2 or 4."
            )

        if self.face_value <= 0.0:
            raise BondValidationError(
                "face_value must be positive."
            )

        if not self.currency.strip():
            raise BondValidationError(
                "currency must not be empty."
            )


@dataclass(frozen=True)
class BondCashFlow:
    """
    Represent one future bond cash flow.
    """

    payment_date: date
    coupon_amount: float
    principal_amount: float
    total_amount: float
    time_in_years: float


@dataclass(frozen=True)
class BondRiskMetrics:
    """
    Store price and interest-rate risk analytics.
    """

    settlement_date: date
    yield_to_maturity: float
    clean_price: float
    dirty_price: float
    accrued_interest: float
    macaulay_duration: float
    modified_duration: float
    dv01_per_100: float
    convexity: float


@dataclass(frozen=True)
class BondScenarioResult:
    """
    Store one parallel-yield scenario result.
    """

    yield_shock_bp: float
    shocked_yield: float
    shocked_clean_price: float
    clean_price_change: float
    position_pnl: float


def validate_settlement_date(
    bond: FixedRateBond,
    settlement_date: date,
) -> None:
    """
    Validate the settlement date against the bond maturity.
    """
    if settlement_date >= bond.maturity_date:
        raise BondValidationError(
            "settlement_date must be before maturity_date."
        )


def months_per_coupon(
    coupon_frequency: int,
) -> int:
    """
    Return the number of calendar months between coupons.
    """
    if coupon_frequency not in {
        1,
        2,
        4,
    }:
        raise BondValidationError(
            "coupon_frequency must be 1, 2 or 4."
        )

    return 12 // coupon_frequency


def build_coupon_dates(
    bond: FixedRateBond,
    settlement_date: date,
) -> list[date]:
    """
    Build all coupon dates surrounding and following settlement.

    Dates are generated backwards from maturity. This approach keeps
    the schedule anchored to the contractual maturity date.
    """
    validate_settlement_date(
        bond=bond,
        settlement_date=settlement_date,
    )

    coupon_dates: list[pd.Timestamp] = []

    current_date = pd.Timestamp(
        bond.maturity_date
    )

    month_step = months_per_coupon(
        bond.coupon_frequency
    )

    while current_date.date() > settlement_date:
        coupon_dates.append(
            current_date
        )

        current_date = (
            current_date
            - pd.DateOffset(
                months=month_step
            )
        )

    coupon_dates.append(
        current_date
    )

    return sorted(
        {
            timestamp.date()
            for timestamp in coupon_dates
        }
    )


def coupon_period_dates(
    bond: FixedRateBond,
    settlement_date: date,
) -> tuple[date, date]:
    """
    Return the previous and next contractual coupon dates.
    """
    schedule = build_coupon_dates(
        bond=bond,
        settlement_date=settlement_date,
    )

    previous_dates = [
        coupon_date
        for coupon_date in schedule
        if coupon_date <= settlement_date
    ]

    future_dates = [
        coupon_date
        for coupon_date in schedule
        if coupon_date > settlement_date
    ]

    if not previous_dates or not future_dates:
        raise BondValidationError(
            "Unable to determine the coupon period around settlement."
        )

    return (
        previous_dates[-1],
        future_dates[0],
    )


def year_fraction_actual_actual(
    start_date: date,
    end_date: date,
) -> float:
    """
    Calculate a simple Actual/Actual year fraction.

    The method divides actual calendar days by 365.25. It is transparent
    and appropriate for the first RepoLens analytical engine, but it is
    not presented as an exchange-specific settlement convention.
    """
    if end_date < start_date:
        raise BondValidationError(
            "end_date must not be before start_date."
        )

    return (
        end_date
        - start_date
    ).days / 365.25


def accrued_interest(
    bond: FixedRateBond,
    settlement_date: date,
) -> float:
    """
    Calculate accrued coupon interest per face-value amount.

    The accrued fraction uses actual days within the contractual coupon
    period.
    """
    previous_coupon, next_coupon = coupon_period_dates(
        bond=bond,
        settlement_date=settlement_date,
    )

    full_period_days = (
        next_coupon
        - previous_coupon
    ).days

    accrued_days = (
        settlement_date
        - previous_coupon
    ).days

    if full_period_days <= 0:
        raise BondValidationError(
            "Coupon period must contain a positive number of days."
        )

    coupon_payment = (
        bond.face_value
        * bond.annual_coupon_rate
        / bond.coupon_frequency
    )

    return (
        coupon_payment
        * accrued_days
        / full_period_days
    )


def future_cash_flows(
    bond: FixedRateBond,
    settlement_date: date,
) -> tuple[BondCashFlow, ...]:
    """
    Build all contractual cash flows strictly after settlement.
    """
    schedule = build_coupon_dates(
        bond=bond,
        settlement_date=settlement_date,
    )

    payment_dates = [
        coupon_date
        for coupon_date in schedule
        if coupon_date > settlement_date
    ]

    if not payment_dates:
        raise BondValidationError(
            "No future cash flows remain after settlement."
        )

    coupon_payment = (
        bond.face_value
        * bond.annual_coupon_rate
        / bond.coupon_frequency
    )

    cash_flows: list[BondCashFlow] = []

    for payment_date in payment_dates:
        principal_amount = (
            bond.face_value
            if payment_date == bond.maturity_date
            else 0.0
        )

        total_amount = (
            coupon_payment
            + principal_amount
        )

        cash_flows.append(
            BondCashFlow(
                payment_date=payment_date,
                coupon_amount=coupon_payment,
                principal_amount=principal_amount,
                total_amount=total_amount,
                time_in_years=year_fraction_actual_actual(
                    start_date=settlement_date,
                    end_date=payment_date,
                ),
            )
        )

    return tuple(
        cash_flows
    )


def validate_yield(
    yield_to_maturity: float,
    coupon_frequency: int,
) -> None:
    """
    Validate a nominal annual yield under periodic compounding.
    """
    periodic_rate = (
        yield_to_maturity
        / coupon_frequency
    )

    if periodic_rate <= -1.0:
        raise BondValidationError(
            "yield_to_maturity produces a non-positive "
            "periodic discount factor."
        )


def dirty_price_from_yield(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> float:
    """
    Calculate dirty price from nominal annual yield.

    Yield is compounded at the bond's coupon frequency.
    """
    validate_settlement_date(
        bond=bond,
        settlement_date=settlement_date,
    )

    validate_yield(
        yield_to_maturity=yield_to_maturity,
        coupon_frequency=bond.coupon_frequency,
    )

    periodic_rate = (
        yield_to_maturity
        / bond.coupon_frequency
    )

    dirty_price = 0.0

    for cash_flow in future_cash_flows(
        bond=bond,
        settlement_date=settlement_date,
    ):
        exponent = (
            cash_flow.time_in_years
            * bond.coupon_frequency
        )

        discount_factor = (
            1.0
            + periodic_rate
        ) ** exponent

        dirty_price += (
            cash_flow.total_amount
            / discount_factor
        )

    return float(
        dirty_price
    )


def clean_price_from_yield(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> float:
    """
    Calculate clean price from nominal annual yield.
    """
    dirty_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    return (
        dirty_price
        - accrued_interest(
            bond=bond,
            settlement_date=settlement_date,
        )
    )


def yield_from_clean_price(
    bond: FixedRateBond,
    settlement_date: date,
    clean_price: float,
    lower_bound: float = -0.95,
    upper_bound: float = 2.00,
) -> float:
    """
    Solve nominal annual yield from a clean price.

    Brent's method is used because it is robust and does not require an
    initial derivative estimate.
    """
    if clean_price <= 0.0:
        raise BondValidationError(
            "clean_price must be positive."
        )

    validate_settlement_date(
        bond=bond,
        settlement_date=settlement_date,
    )

    if lower_bound >= upper_bound:
        raise BondValidationError(
            "lower_bound must be below upper_bound."
        )

    minimum_valid_yield = (
        -float(
            bond.coupon_frequency
        )
        + 1e-8
    )

    effective_lower_bound = max(
        lower_bound,
        minimum_valid_yield,
    )

    def pricing_error(
        candidate_yield: float,
    ) -> float:
        return (
            clean_price_from_yield(
                bond=bond,
                settlement_date=settlement_date,
                yield_to_maturity=candidate_yield,
            )
            - clean_price
        )

    lower_error = pricing_error(
        effective_lower_bound
    )

    upper_error = pricing_error(
        upper_bound
    )

    if lower_error == 0.0:
        return effective_lower_bound

    if upper_error == 0.0:
        return upper_bound

    if lower_error * upper_error > 0.0:
        raise YieldSolveError(
            "Unable to bracket the yield solution. "
            "Check the supplied clean price and yield bounds."
        )

    solved_yield = brentq(
        pricing_error,
        effective_lower_bound,
        upper_bound,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=500,
    )

    return float(
        solved_yield
    )


def macaulay_duration(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> float:
    """
    Calculate Macaulay duration in years.
    """
    validate_yield(
        yield_to_maturity=yield_to_maturity,
        coupon_frequency=bond.coupon_frequency,
    )

    periodic_rate = (
        yield_to_maturity
        / bond.coupon_frequency
    )

    weighted_present_value = 0.0
    total_present_value = 0.0

    for cash_flow in future_cash_flows(
        bond=bond,
        settlement_date=settlement_date,
    ):
        exponent = (
            cash_flow.time_in_years
            * bond.coupon_frequency
        )

        present_value = (
            cash_flow.total_amount
            / (
                1.0
                + periodic_rate
            ) ** exponent
        )

        weighted_present_value += (
            cash_flow.time_in_years
            * present_value
        )

        total_present_value += present_value

    if total_present_value <= 0.0:
        raise BondAnalyticsError(
            "Bond present value must be positive."
        )

    return (
        weighted_present_value
        / total_present_value
    )


def modified_duration(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> float:
    """
    Calculate modified duration in years.
    """
    macaulay = macaulay_duration(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    periodic_rate = (
        yield_to_maturity
        / bond.coupon_frequency
    )

    return (
        macaulay
        / (
            1.0
            + periodic_rate
        )
    )


def convexity(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
    yield_step: float = 0.0001,
) -> float:
    """
    Estimate effective convexity using central price differences.

    The default yield step is one basis point.
    """
    if yield_step <= 0.0:
        raise BondValidationError(
            "yield_step must be positive."
        )

    base_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    lower_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=(
            yield_to_maturity
            - yield_step
        ),
    )

    upper_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=(
            yield_to_maturity
            + yield_step
        ),
    )

    return (
        lower_price
        + upper_price
        - 2.0 * base_price
    ) / (
        base_price
        * yield_step**2
    )


def dv01_per_100(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> float:
    """
    Calculate the positive price sensitivity to a one-basis-point fall.

    The output is currency value per 100 face value.
    """
    one_basis_point = 0.0001

    price_after_yield_fall = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=(
            yield_to_maturity
            - one_basis_point
        ),
    )

    price_after_yield_rise = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=(
            yield_to_maturity
            + one_basis_point
        ),
    )

    return (
        price_after_yield_fall
        - price_after_yield_rise
    ) / 2.0


def calculate_bond_risk_metrics(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
) -> BondRiskMetrics:
    """
    Calculate the core RepoLens bond valuation and risk metrics.
    """
    dirty_price = dirty_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    accrued = accrued_interest(
        bond=bond,
        settlement_date=settlement_date,
    )

    clean_price = (
        dirty_price
        - accrued
    )

    return BondRiskMetrics(
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
        clean_price=clean_price,
        dirty_price=dirty_price,
        accrued_interest=accrued,
        macaulay_duration=macaulay_duration(
            bond=bond,
            settlement_date=settlement_date,
            yield_to_maturity=yield_to_maturity,
        ),
        modified_duration=modified_duration(
            bond=bond,
            settlement_date=settlement_date,
            yield_to_maturity=yield_to_maturity,
        ),
        dv01_per_100=dv01_per_100(
            bond=bond,
            settlement_date=settlement_date,
            yield_to_maturity=yield_to_maturity,
        ),
        convexity=convexity(
            bond=bond,
            settlement_date=settlement_date,
            yield_to_maturity=yield_to_maturity,
        ),
    )


def position_dv01(
    dv01_per_100_value: float,
    position_notional: float,
) -> float:
    """
    Scale per-100 DV01 to a position notional.
    """
    if dv01_per_100_value < 0.0:
        raise BondValidationError(
            "dv01_per_100_value must not be negative."
        )

    if position_notional < 0.0:
        raise BondValidationError(
            "position_notional must not be negative."
        )

    return (
        dv01_per_100_value
        * position_notional
        / 100.0
    )


def run_parallel_yield_scenarios(
    bond: FixedRateBond,
    settlement_date: date,
    yield_to_maturity: float,
    position_notional: float,
    yield_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_YIELD_SHOCKS_BP,
) -> tuple[BondScenarioResult, ...]:
    """
    Reprice a bond under parallel yield shocks.

    Position P&L assumes position_notional is the face-value amount.
    A positive number represents a long position.
    """
    if position_notional < 0.0:
        raise BondValidationError(
            "position_notional must not be negative."
        )

    if not yield_shocks_bp:
        raise BondValidationError(
            "yield_shocks_bp must not be empty."
        )

    base_clean_price = clean_price_from_yield(
        bond=bond,
        settlement_date=settlement_date,
        yield_to_maturity=yield_to_maturity,
    )

    scenario_results: list[BondScenarioResult] = []

    for yield_shock_bp in yield_shocks_bp:
        shocked_yield = (
            yield_to_maturity
            + yield_shock_bp
            / 10_000.0
        )

        validate_yield(
            yield_to_maturity=shocked_yield,
            coupon_frequency=bond.coupon_frequency,
        )

        shocked_clean_price = clean_price_from_yield(
            bond=bond,
            settlement_date=settlement_date,
            yield_to_maturity=shocked_yield,
        )

        clean_price_change = (
            shocked_clean_price
            - base_clean_price
        )

        position_pnl = (
            clean_price_change
            * position_notional
            / bond.face_value
        )

        scenario_results.append(
            BondScenarioResult(
                yield_shock_bp=float(
                    yield_shock_bp
                ),
                shocked_yield=shocked_yield,
                shocked_clean_price=shocked_clean_price,
                clean_price_change=clean_price_change,
                position_pnl=position_pnl,
            )
        )

    return tuple(
        scenario_results
    )


def scenario_results_to_frame(
    scenario_results: tuple[
        BondScenarioResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Convert scenario results into a tabular output.
    """
    if not scenario_results:
        raise BondValidationError(
            "scenario_results must not be empty."
        )

    return pd.DataFrame(
        [
            {
                "yield_shock_bp": result.yield_shock_bp,
                "shocked_yield": result.shocked_yield,
                "shocked_clean_price": result.shocked_clean_price,
                "clean_price_change": result.clean_price_change,
                "position_pnl": result.position_pnl,
            }
            for result in scenario_results
        ]
    )