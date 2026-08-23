from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.repo_analytics import (
    collateral_market_value,
    purchase_price_from_haircut,
    repo_interest,
)
from src.sovereign_relative_value import PositionDirection


class RepoAdjustedRelativeValueError(RuntimeError):
    """Base exception for RepoLens repo-adjusted relative-value analytics."""


class RepoAdjustedRelativeValueValidationError(
    RepoAdjustedRelativeValueError
):
    """Raised when a repo-adjusted RV input is internally inconsistent."""


@dataclass(frozen=True)
class RepoFundingLegInput:
    """
    Define the repo-funding economics for one sovereign RV leg.

    Rates and haircuts are percentage points. Prices are per 100 face value.

    Direction determines how collateral specialness affects the trade:
    - LONG: cheaper specific funding versus GC is beneficial.
    - SHORT: obtaining special collateral is economically costly versus GC,
      so the same financing edge is applied with the opposite sign.
    """

    isin: str
    direction: PositionDirection
    face_value_eur: float
    dirty_price_per_100: float
    haircut_percent: float
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    repo_days: int
    day_count_basis: int = 360

    def __post_init__(self) -> None:
        isin = self.isin.strip().upper()

        if len(isin) != 12 or not isin.isalnum():
            raise RepoAdjustedRelativeValueValidationError(
                "isin must contain exactly 12 alphanumeric characters."
            )

        if self.face_value_eur <= 0.0:
            raise RepoAdjustedRelativeValueValidationError(
                "face_value_eur must be positive."
            )

        if not isfinite(self.dirty_price_per_100):
            raise RepoAdjustedRelativeValueValidationError(
                "dirty_price_per_100 must be finite."
            )

        if self.dirty_price_per_100 <= 0.0:
            raise RepoAdjustedRelativeValueValidationError(
                "dirty_price_per_100 must be positive."
            )

        if not isfinite(self.haircut_percent):
            raise RepoAdjustedRelativeValueValidationError(
                "haircut_percent must be finite."
            )

        if not -100.0 < self.haircut_percent < 100.0:
            raise RepoAdjustedRelativeValueValidationError(
                "haircut_percent must be greater than -100% and less than 100%."
            )

        for field_name, rate in (
            (
                "specific_repo_rate_percent",
                self.specific_repo_rate_percent,
            ),
            (
                "gc_repo_rate_percent",
                self.gc_repo_rate_percent,
            ),
        ):
            if not isfinite(rate):
                raise RepoAdjustedRelativeValueValidationError(
                    f"{field_name} must be finite."
                )

            if rate <= -100.0:
                raise RepoAdjustedRelativeValueValidationError(
                    f"{field_name} must be greater than -100%."
                )

        if self.repo_days <= 0:
            raise RepoAdjustedRelativeValueValidationError(
                "repo_days must be positive."
            )

        if self.day_count_basis not in {
            360,
            365,
        }:
            raise RepoAdjustedRelativeValueValidationError(
                "day_count_basis must be 360 or 365."
            )


@dataclass(frozen=True)
class RepoFundingLegResult:
    """Store the funding overlay for one long or short RV leg."""

    isin: str
    direction: PositionDirection
    face_value_eur: float
    dirty_price_per_100: float
    collateral_market_value_eur: float
    cash_advanced_eur: float
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float
    specific_repo_interest_eur: float
    gc_repo_interest_eur: float
    unsigned_financing_edge_vs_gc_eur: float
    signed_financing_impact_vs_gc_eur: float
    signed_financing_impact_per_eur_1m_face: float


@dataclass(frozen=True)
class RepoAdjustedRelativeValueAnalysis:
    """
    Store a two-leg repo funding overlay for an existing sovereign RV trade.

    This object does not replace cash-bond spread or duration analytics. It
    isolates how each leg's specific repo rate changes the economics relative
    to funding/borrowing the same trade at its matched GC reference.
    """

    anchor: RepoFundingLegResult
    hedge: RepoFundingLegResult
    repo_days: int
    day_count_basis: int
    net_signed_financing_impact_vs_gc_eur: float
    gross_absolute_financing_impact_vs_gc_eur: float
    net_signed_financing_impact_per_eur_1m_anchor_face: float
    anchor_minus_hedge_specialness_bp: float


def _direction_sign(
    direction: PositionDirection,
) -> float:
    if direction == PositionDirection.LONG:
        return 1.0

    if direction == PositionDirection.SHORT:
        return -1.0

    raise RepoAdjustedRelativeValueValidationError(
        "direction must be LONG or SHORT."
    )


def analyse_repo_funding_leg(
    leg: RepoFundingLegInput,
) -> RepoFundingLegResult:
    """
    Calculate the matched specific-versus-GC funding overlay for one RV leg.

    Positive unsigned financing edge means the specific repo rate is below GC.
    The signed trade impact is positive for a long collateral position and
    negative for a short collateral position.
    """
    try:
        market_value = collateral_market_value(
            face_value_eur=leg.face_value_eur,
            dirty_price=leg.dirty_price_per_100,
        )

        cash_advanced = purchase_price_from_haircut(
            collateral_market_value_eur=market_value,
            haircut_percent=leg.haircut_percent,
        )

        specific_interest = repo_interest(
            purchase_price_eur=cash_advanced,
            repo_rate_percent=leg.specific_repo_rate_percent,
            repo_days=leg.repo_days,
            day_count_basis=leg.day_count_basis,
        )

        gc_interest = repo_interest(
            purchase_price_eur=cash_advanced,
            repo_rate_percent=leg.gc_repo_rate_percent,
            repo_days=leg.repo_days,
            day_count_basis=leg.day_count_basis,
        )

    except Exception as error:
        raise RepoAdjustedRelativeValueValidationError(
            str(error)
        ) from error

    specialness_bp = (
        leg.gc_repo_rate_percent
        - leg.specific_repo_rate_percent
    ) * 100.0

    unsigned_edge = (
        gc_interest
        - specific_interest
    )

    signed_impact = (
        unsigned_edge
        * _direction_sign(
            leg.direction
        )
    )

    signed_impact_per_eur_1m = (
        signed_impact
        * 1_000_000.0
        / leg.face_value_eur
    )

    return RepoFundingLegResult(
        isin=leg.isin.strip().upper(),
        direction=leg.direction,
        face_value_eur=leg.face_value_eur,
        dirty_price_per_100=leg.dirty_price_per_100,
        collateral_market_value_eur=market_value,
        cash_advanced_eur=cash_advanced,
        specific_repo_rate_percent=leg.specific_repo_rate_percent,
        gc_repo_rate_percent=leg.gc_repo_rate_percent,
        specialness_bp=specialness_bp,
        specific_repo_interest_eur=specific_interest,
        gc_repo_interest_eur=gc_interest,
        unsigned_financing_edge_vs_gc_eur=unsigned_edge,
        signed_financing_impact_vs_gc_eur=signed_impact,
        signed_financing_impact_per_eur_1m_face=(
            signed_impact_per_eur_1m
        ),
    )


def analyse_repo_adjusted_relative_value(
    *,
    anchor: RepoFundingLegInput,
    hedge: RepoFundingLegInput,
) -> RepoAdjustedRelativeValueAnalysis:
    """
    Build the repo-funding overlay for a two-leg sovereign RV position.

    Both legs must use the same contractual repo horizon and money-market
    day-count basis. RepoLens rejects unmatched horizons rather than silently
    comparing different funding periods.

    The result is deliberately an overlay. Cash spread, DV01-neutral sizing,
    convexity and yield-scenario P&L remain the responsibility of the sovereign
    relative-value engine.
    """
    if anchor.isin.strip().upper() == hedge.isin.strip().upper():
        raise RepoAdjustedRelativeValueValidationError(
            "Anchor and hedge must be different instruments."
        )

    if anchor.repo_days != hedge.repo_days:
        raise RepoAdjustedRelativeValueValidationError(
            "Anchor and hedge repo_days must match."
        )

    if anchor.day_count_basis != hedge.day_count_basis:
        raise RepoAdjustedRelativeValueValidationError(
            "Anchor and hedge day_count_basis must match."
        )

    if anchor.direction == hedge.direction:
        raise RepoAdjustedRelativeValueValidationError(
            "Anchor and hedge directions must be opposite."
        )

    anchor_result = analyse_repo_funding_leg(
        anchor
    )

    hedge_result = analyse_repo_funding_leg(
        hedge
    )

    net_signed_impact = (
        anchor_result.signed_financing_impact_vs_gc_eur
        + hedge_result.signed_financing_impact_vs_gc_eur
    )

    gross_absolute_impact = (
        abs(
            anchor_result.signed_financing_impact_vs_gc_eur
        )
        + abs(
            hedge_result.signed_financing_impact_vs_gc_eur
        )
    )

    net_per_eur_1m_anchor = (
        net_signed_impact
        * 1_000_000.0
        / anchor.face_value_eur
    )

    return RepoAdjustedRelativeValueAnalysis(
        anchor=anchor_result,
        hedge=hedge_result,
        repo_days=anchor.repo_days,
        day_count_basis=anchor.day_count_basis,
        net_signed_financing_impact_vs_gc_eur=(
            net_signed_impact
        ),
        gross_absolute_financing_impact_vs_gc_eur=(
            gross_absolute_impact
        ),
        net_signed_financing_impact_per_eur_1m_anchor_face=(
            net_per_eur_1m_anchor
        ),
        anchor_minus_hedge_specialness_bp=(
            anchor_result.specialness_bp
            - hedge_result.specialness_bp
        ),
    )