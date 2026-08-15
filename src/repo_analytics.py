from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

from src.bond_analytics import (
    BondValidationError,
    FixedRateBond,
    YieldSolveError,
    accrued_interest,
    dirty_price_from_yield,
    future_cash_flows,
    yield_from_clean_price,
)


DEFAULT_CARRY_SHOCKS_BP: Final[tuple[float, ...]] = (
    -25.0,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    25.0,
)


class RepoAnalyticsError(RuntimeError):
    """
    Base exception for RepoLens repo analytics.
    """


class RepoValidationError(RepoAnalyticsError):
    """
    Raised when repo trade inputs are invalid.
    """


@dataclass(frozen=True)
class RepoTradeInput:
    """
    Define a fixed-term classic repo from the collateral-provider view.

    Rates and haircuts are entered in percentage points.

    Examples:
        repo_rate_percent=2.50 means 2.50% per annum.
        haircut_percent=2.00 means a 2.00% haircut.
    """

    face_value_eur: float
    clean_price_per_100: float
    accrued_interest_per_100: float
    repo_rate_percent: float
    haircut_percent: float
    purchase_date: date
    repurchase_date: date
    day_count_basis: int = 360
    gc_repo_rate_percent: float | None = None
    interim_income_eur: float = 0.0

    def __post_init__(self) -> None:
        if self.face_value_eur <= 0.0:
            raise RepoValidationError(
                "face_value_eur must be positive."
            )

        if self.clean_price_per_100 < 0.0:
            raise RepoValidationError(
                "clean_price_per_100 must not be negative."
            )

        if self.accrued_interest_per_100 < 0.0:
            raise RepoValidationError(
                "accrued_interest_per_100 must not be negative."
            )

        if not -100.0 < self.haircut_percent < 100.0:
            raise RepoValidationError(
                "haircut_percent must be greater than -100% and less than 100%."
            )

        if self.repurchase_date <= self.purchase_date:
            raise RepoValidationError(
                "repurchase_date must be after purchase_date."
            )

        if self.day_count_basis not in {
            360,
            365,
        }:
            raise RepoValidationError(
                "day_count_basis must be 360 or 365."
            )

        if self.interim_income_eur < 0.0:
            raise RepoValidationError(
                "interim_income_eur must not be negative."
            )


@dataclass(frozen=True)
class RepoTradeResult:
    """
    Store calculated fixed-term repo economics.
    """

    repo_days: int
    dirty_price_per_100: float
    collateral_market_value_eur: float
    haircut_amount_eur: float
    purchase_price_eur: float
    repo_interest_eur: float
    repurchase_price_eur: float
    manufactured_payment_eur: float
    total_cash_returned_to_collateral_provider_eur: float
    gc_repo_rate_percent: float | None
    specialness_bp: float | None
    financing_benefit_vs_gc_eur: float | None


@dataclass(frozen=True)
class BondCarryScenario:
    """
    Store one financed bond horizon scenario.

    The collateral provider remains economically exposed to the bond.
    Scenario P&L therefore combines bond price change, coupon income and
    the repo financing cost over the same horizon.
    """

    yield_shock_bp: float
    exit_yield_percent: float
    exit_dirty_price_per_100: float
    coupon_income_eur: float
    gross_bond_pnl_eur: float
    repo_interest_eur: float
    financing_adjusted_pnl_eur: float
    financing_adjusted_pnl_per_eur_1m_face: float


@dataclass(frozen=True)
class BondCarryAnalysis:
    """
    Store the complete financing-adjusted carry analysis for a coupon bond.
    """

    start_yield_percent: float
    start_dirty_price_per_100: float
    start_market_value_eur: float
    repo_interest_eur: float
    coupon_income_eur: float
    breakeven_exit_yield_percent: float | None
    breakeven_yield_move_bp: float | None
    scenarios: tuple[
        BondCarryScenario,
        ...,
    ]


@dataclass(frozen=True)
class DiscountSecurityCarryAnalysis:
    """
    Store financing-to-maturity economics for a zero-coupon security.

    The entered repo rate is assumed to remain unchanged through maturity.
    This is a scenario, not an executable term-repo quote.
    """

    days_to_maturity: int
    start_price_per_100: float
    start_market_value_eur: float
    redemption_value_eur: float
    cash_advanced_eur: float
    gross_pull_to_par_eur: float
    financing_cost_to_maturity_eur: float
    financing_adjusted_pull_to_par_eur: float
    financing_adjusted_pull_to_par_per_eur_1m_face: float
    breakeven_repo_rate_percent: float
    financing_adjusted_annualised_return_percent: float
    gc_financing_cost_to_maturity_eur: float | None
    financing_benefit_vs_gc_to_maturity_eur: float | None


def dirty_price_per_100(
    clean_price_per_100: float,
    accrued_interest_per_100: float,
) -> float:
    """
    Return full/dirty price per 100 of face value.
    """
    if clean_price_per_100 < 0.0:
        raise RepoValidationError(
            "clean_price_per_100 must not be negative."
        )

    if accrued_interest_per_100 < 0.0:
        raise RepoValidationError(
            "accrued_interest_per_100 must not be negative."
        )

    return (
        clean_price_per_100
        + accrued_interest_per_100
    )


def collateral_market_value(
    face_value_eur: float,
    dirty_price: float,
) -> float:
    """
    Convert face value and dirty price into collateral market value.
    """
    if face_value_eur <= 0.0:
        raise RepoValidationError(
            "face_value_eur must be positive."
        )

    if dirty_price < 0.0:
        raise RepoValidationError(
            "dirty_price must not be negative."
        )

    return (
        face_value_eur
        * dirty_price
        / 100.0
    )


def purchase_price_from_haircut(
    collateral_market_value_eur: float,
    haircut_percent: float,
) -> float:
    """
    Calculate cash advanced after applying a haircut to collateral value.
    """
    if collateral_market_value_eur < 0.0:
        raise RepoValidationError(
            "collateral_market_value_eur must not be negative."
        )

    if not -100.0 < haircut_percent < 100.0:
        raise RepoValidationError(
            "haircut_percent must be greater than -100% and less than 100%."
        )

    return (
        collateral_market_value_eur
        * (
            1.0
            - haircut_percent
            / 100.0
        )
    )


def repo_interest(
    purchase_price_eur: float,
    repo_rate_percent: float,
    repo_days: int,
    day_count_basis: int = 360,
) -> float:
    """
    Calculate simple repo interest for a fixed-term transaction.
    """
    if purchase_price_eur < 0.0:
        raise RepoValidationError(
            "purchase_price_eur must not be negative."
        )

    if repo_days <= 0:
        raise RepoValidationError(
            "repo_days must be positive."
        )

    if day_count_basis not in {
        360,
        365,
    }:
        raise RepoValidationError(
            "day_count_basis must be 360 or 365."
        )

    return (
        purchase_price_eur
        * (
            repo_rate_percent
            / 100.0
        )
        * repo_days
        / day_count_basis
    )


def required_collateral_market_value(
    target_cash_eur: float,
    haircut_percent: float,
) -> float:
    """
    Calculate collateral market value required for a target cash amount.
    """
    if target_cash_eur <= 0.0:
        raise RepoValidationError(
            "target_cash_eur must be positive."
        )

    haircut_multiplier = (
        1.0
        - haircut_percent
        / 100.0
    )

    if haircut_multiplier <= 0.0:
        raise RepoValidationError(
            "haircut_percent leaves no positive purchase-price multiplier."
        )

    return (
        target_cash_eur
        / haircut_multiplier
    )


def required_face_value(
    target_cash_eur: float,
    dirty_price: float,
    haircut_percent: float,
) -> float:
    """
    Calculate face value required to raise target cash.
    """
    if dirty_price <= 0.0:
        raise RepoValidationError(
            "dirty_price must be positive."
        )

    required_market_value = (
        required_collateral_market_value(
            target_cash_eur=target_cash_eur,
            haircut_percent=haircut_percent,
        )
    )

    return (
        required_market_value
        * 100.0
        / dirty_price
    )


def calculate_repo_trade(
    trade: RepoTradeInput,
) -> RepoTradeResult:
    """
    Calculate fixed-term classic repo economics.

    The purchase price is the collateral market value after haircut.
    Repo interest is simple interest on that purchase price.
    In a classic repurchase transaction, interim collateral income is
    represented as an equivalent manufactured payment back to the
    collateral provider rather than a reduction in repurchase price.
    """
    full_price = dirty_price_per_100(
        clean_price_per_100=trade.clean_price_per_100,
        accrued_interest_per_100=trade.accrued_interest_per_100,
    )

    market_value = collateral_market_value(
        face_value_eur=trade.face_value_eur,
        dirty_price=full_price,
    )

    purchase_price = purchase_price_from_haircut(
        collateral_market_value_eur=market_value,
        haircut_percent=trade.haircut_percent,
    )

    haircut_amount = (
        market_value
        - purchase_price
    )

    days = (
        trade.repurchase_date
        - trade.purchase_date
    ).days

    interest = repo_interest(
        purchase_price_eur=purchase_price,
        repo_rate_percent=trade.repo_rate_percent,
        repo_days=days,
        day_count_basis=trade.day_count_basis,
    )

    repurchase_price = (
        purchase_price
        + interest
    )

    specialness_bp: float | None = None
    financing_benefit_vs_gc_eur: float | None = None

    if trade.gc_repo_rate_percent is not None:
        specialness_bp = (
            trade.gc_repo_rate_percent
            - trade.repo_rate_percent
        ) * 100.0

        gc_interest = repo_interest(
            purchase_price_eur=purchase_price,
            repo_rate_percent=trade.gc_repo_rate_percent,
            repo_days=days,
            day_count_basis=trade.day_count_basis,
        )

        financing_benefit_vs_gc_eur = (
            gc_interest
            - interest
        )

    return RepoTradeResult(
        repo_days=days,
        dirty_price_per_100=full_price,
        collateral_market_value_eur=market_value,
        haircut_amount_eur=haircut_amount,
        purchase_price_eur=purchase_price,
        repo_interest_eur=interest,
        repurchase_price_eur=repurchase_price,
        manufactured_payment_eur=trade.interim_income_eur,
        total_cash_returned_to_collateral_provider_eur=(
            trade.interim_income_eur
        ),
        gc_repo_rate_percent=trade.gc_repo_rate_percent,
        specialness_bp=specialness_bp,
        financing_benefit_vs_gc_eur=financing_benefit_vs_gc_eur,
    )


def coupon_income_between_dates(
    bond: FixedRateBond,
    purchase_date: date,
    repurchase_date: date,
    position_face_value_eur: float,
) -> float:
    """
    Calculate coupon income economically attributable to the bond holder
    during the repo horizon.

    Principal is excluded because financed carry currently requires the
    repo to terminate before collateral maturity.
    """
    if repurchase_date >= bond.maturity_date:
        raise RepoValidationError(
            "Bond carry analysis requires repurchase_date before maturity_date."
        )

    if position_face_value_eur <= 0.0:
        raise RepoValidationError(
            "position_face_value_eur must be positive."
        )

    try:
        cash_flows = future_cash_flows(
            bond=bond,
            settlement_date=purchase_date,
        )
    except BondValidationError as error:
        raise RepoValidationError(
            str(
                error
            )
        ) from error

    coupon_per_100 = sum(
        cash_flow.coupon_amount
        for cash_flow in cash_flows
        if (
            purchase_date
            < cash_flow.payment_date
            <= repurchase_date
        )
    )

    return (
        coupon_per_100
        * position_face_value_eur
        / bond.face_value
    )


def bond_carry_scenario(
    bond: FixedRateBond,
    trade: RepoTradeInput,
    repo_result: RepoTradeResult,
    start_yield_decimal: float,
    yield_shock_bp: float,
    coupon_income_eur: float,
) -> BondCarryScenario:
    """
    Reprice a financed coupon bond at repo maturity under one yield shock.
    """
    if trade.repurchase_date >= bond.maturity_date:
        raise RepoValidationError(
            "Bond carry analysis requires repurchase_date before maturity_date."
        )

    exit_yield_decimal = (
        start_yield_decimal
        + yield_shock_bp
        / 10_000.0
    )

    try:
        exit_dirty_price = dirty_price_from_yield(
            bond=bond,
            settlement_date=trade.repurchase_date,
            yield_to_maturity=exit_yield_decimal,
        )
    except BondValidationError as error:
        raise RepoValidationError(
            str(
                error
            )
        ) from error

    exit_market_value = collateral_market_value(
        face_value_eur=trade.face_value_eur,
        dirty_price=exit_dirty_price,
    )

    gross_bond_pnl = (
        exit_market_value
        + coupon_income_eur
        - repo_result.collateral_market_value_eur
    )

    financing_adjusted_pnl = (
        gross_bond_pnl
        - repo_result.repo_interest_eur
    )

    pnl_per_eur_1m = (
        financing_adjusted_pnl
        * 1_000_000.0
        / trade.face_value_eur
    )

    return BondCarryScenario(
        yield_shock_bp=yield_shock_bp,
        exit_yield_percent=(
            exit_yield_decimal
            * 100.0
        ),
        exit_dirty_price_per_100=exit_dirty_price,
        coupon_income_eur=coupon_income_eur,
        gross_bond_pnl_eur=gross_bond_pnl,
        repo_interest_eur=repo_result.repo_interest_eur,
        financing_adjusted_pnl_eur=financing_adjusted_pnl,
        financing_adjusted_pnl_per_eur_1m_face=pnl_per_eur_1m,
    )


def breakeven_exit_yield(
    bond: FixedRateBond,
    trade: RepoTradeInput,
    repo_result: RepoTradeResult,
    coupon_income_eur: float,
) -> float | None:
    """
    Solve the exit yield at which financing-adjusted horizon P&L is zero.

    The required exit dirty value equals starting collateral market value
    plus repo interest less coupon income received over the horizon.
    """
    required_exit_market_value = (
        repo_result.collateral_market_value_eur
        + repo_result.repo_interest_eur
        - coupon_income_eur
    )

    required_exit_dirty_price = (
        required_exit_market_value
        * 100.0
        / trade.face_value_eur
    )

    try:
        exit_accrued = accrued_interest(
            bond=bond,
            settlement_date=trade.repurchase_date,
        )
    except BondValidationError:
        return None

    required_exit_clean_price = (
        required_exit_dirty_price
        - exit_accrued
    )

    if required_exit_clean_price <= 0.0:
        return None

    try:
        return yield_from_clean_price(
            bond=bond,
            settlement_date=trade.repurchase_date,
            clean_price=required_exit_clean_price,
        )
    except (
        BondValidationError,
        YieldSolveError,
    ):
        return None


def analyse_financed_bond_carry(
    bond: FixedRateBond,
    trade: RepoTradeInput,
    yield_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_CARRY_SHOCKS_BP,
) -> BondCarryAnalysis:
    """
    Analyse coupon-bond carry over the repo horizon.

    Starting yield is solved from the desk clean price. At repo maturity,
    the bond is repriced at the starting yield plus each requested shock.
    Coupon income during the horizon is included and repo interest is
    deducted to produce financing-adjusted P&L.
    """
    if not yield_shocks_bp:
        raise RepoValidationError(
            "yield_shocks_bp must not be empty."
        )

    if trade.repurchase_date >= bond.maturity_date:
        raise RepoValidationError(
            "Bond carry analysis requires repurchase_date before maturity_date."
        )

    try:
        start_yield = yield_from_clean_price(
            bond=bond,
            settlement_date=trade.purchase_date,
            clean_price=trade.clean_price_per_100,
        )
    except (
        BondValidationError,
        YieldSolveError,
    ) as error:
        raise RepoValidationError(
            "Unable to solve starting bond yield from the supplied clean price."
        ) from error

    repo_result = calculate_repo_trade(
        trade
    )

    coupon_income = coupon_income_between_dates(
        bond=bond,
        purchase_date=trade.purchase_date,
        repurchase_date=trade.repurchase_date,
        position_face_value_eur=trade.face_value_eur,
    )

    scenarios = tuple(
        bond_carry_scenario(
            bond=bond,
            trade=trade,
            repo_result=repo_result,
            start_yield_decimal=start_yield,
            yield_shock_bp=shock,
            coupon_income_eur=coupon_income,
        )
        for shock in yield_shocks_bp
    )

    breakeven_yield = breakeven_exit_yield(
        bond=bond,
        trade=trade,
        repo_result=repo_result,
        coupon_income_eur=coupon_income,
    )

    breakeven_move_bp: float | None = None

    if breakeven_yield is not None:
        breakeven_move_bp = (
            breakeven_yield
            - start_yield
        ) * 10_000.0

    return BondCarryAnalysis(
        start_yield_percent=(
            start_yield
            * 100.0
        ),
        start_dirty_price_per_100=repo_result.dirty_price_per_100,
        start_market_value_eur=repo_result.collateral_market_value_eur,
        repo_interest_eur=repo_result.repo_interest_eur,
        coupon_income_eur=coupon_income,
        breakeven_exit_yield_percent=(
            breakeven_yield
            * 100.0
            if breakeven_yield is not None
            else None
        ),
        breakeven_yield_move_bp=breakeven_move_bp,
        scenarios=scenarios,
    )


def analyse_discount_security_carry_to_maturity(
    *,
    face_value_eur: float,
    price_per_100: float,
    redemption_value_per_100: float,
    purchase_date: date,
    maturity_date: date,
    repo_rate_percent: float,
    haircut_percent: float,
    day_count_basis: int = 360,
    gc_repo_rate_percent: float | None = None,
) -> DiscountSecurityCarryAnalysis:
    """
    Analyse a zero-coupon security from purchase through redemption.

    Financing cost assumes the entered repo rate can be maintained or
    rolled unchanged until maturity. Haircut opportunity cost, transaction
    costs, margin changes and future repo-rate changes are excluded.
    """
    if face_value_eur <= 0.0:
        raise RepoValidationError(
            "face_value_eur must be positive."
        )

    if price_per_100 <= 0.0:
        raise RepoValidationError(
            "price_per_100 must be positive."
        )

    if redemption_value_per_100 <= 0.0:
        raise RepoValidationError(
            "redemption_value_per_100 must be positive."
        )

    if maturity_date <= purchase_date:
        raise RepoValidationError(
            "maturity_date must be after purchase_date."
        )

    if day_count_basis not in {360, 365}:
        raise RepoValidationError(
            "day_count_basis must be 360 or 365."
        )

    days_to_maturity = (
        maturity_date
        - purchase_date
    ).days

    start_market_value = collateral_market_value(
        face_value_eur=face_value_eur,
        dirty_price=price_per_100,
    )

    cash_advanced = purchase_price_from_haircut(
        collateral_market_value_eur=start_market_value,
        haircut_percent=haircut_percent,
    )

    if cash_advanced <= 0.0:
        raise RepoValidationError(
            "Haircut leaves no positive cash advanced."
        )

    redemption_value = (
        face_value_eur
        * redemption_value_per_100
        / 100.0
    )

    gross_pull_to_par = (
        redemption_value
        - start_market_value
    )

    financing_cost = repo_interest(
        purchase_price_eur=cash_advanced,
        repo_rate_percent=repo_rate_percent,
        repo_days=days_to_maturity,
        day_count_basis=day_count_basis,
    )

    net_pull_to_par = (
        gross_pull_to_par
        - financing_cost
    )

    pnl_per_eur_1m_face = (
        net_pull_to_par
        * 1_000_000.0
        / face_value_eur
    )

    breakeven_repo_rate_percent = (
        gross_pull_to_par
        / cash_advanced
        * day_count_basis
        / days_to_maturity
        * 100.0
    )

    annualised_net_return_percent = (
        net_pull_to_par
        / start_market_value
        * day_count_basis
        / days_to_maturity
        * 100.0
    )

    gc_financing_cost: float | None = None
    financing_benefit_vs_gc: float | None = None

    if gc_repo_rate_percent is not None:
        gc_financing_cost = repo_interest(
            purchase_price_eur=cash_advanced,
            repo_rate_percent=gc_repo_rate_percent,
            repo_days=days_to_maturity,
            day_count_basis=day_count_basis,
        )

        financing_benefit_vs_gc = (
            gc_financing_cost
            - financing_cost
        )

    return DiscountSecurityCarryAnalysis(
        days_to_maturity=days_to_maturity,
        start_price_per_100=price_per_100,
        start_market_value_eur=start_market_value,
        redemption_value_eur=redemption_value,
        cash_advanced_eur=cash_advanced,
        gross_pull_to_par_eur=gross_pull_to_par,
        financing_cost_to_maturity_eur=financing_cost,
        financing_adjusted_pull_to_par_eur=net_pull_to_par,
        financing_adjusted_pull_to_par_per_eur_1m_face=(
            pnl_per_eur_1m_face
        ),
        breakeven_repo_rate_percent=breakeven_repo_rate_percent,
        financing_adjusted_annualised_return_percent=(
            annualised_net_return_percent
        ),
        gc_financing_cost_to_maturity_eur=gc_financing_cost,
        financing_benefit_vs_gc_to_maturity_eur=(
            financing_benefit_vs_gc
        ),
    )