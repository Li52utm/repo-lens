from __future__ import annotations

from dataclasses import dataclass
from datetime import date


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