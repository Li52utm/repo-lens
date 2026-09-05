from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

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



@dataclass(frozen=True)
class SpreadScenarioPoint:
    """
    Store one full-repricing cash-RV spread scenario.

    spread_shock_bp is the change in the anchor-minus-hedge yield spread.
    cash_pnl_eur is the two-leg cash-bond P&L before the repo funding overlay.
    """

    spread_shock_bp: float
    cash_pnl_eur: float

    def __post_init__(self) -> None:
        if not isfinite(self.spread_shock_bp):
            raise RepoAdjustedRelativeValueValidationError(
                "spread_shock_bp must be finite."
            )

        if not isfinite(self.cash_pnl_eur):
            raise RepoAdjustedRelativeValueValidationError(
                "cash_pnl_eur must be finite."
            )


@dataclass(frozen=True)
class FundingAdjustedSpreadBreakeven:
    """
    Describe how repo funding shifts the spread breakeven of an RV trade.

    The cash-RV P&L curve is supplied as full-repricing scenario points.
    RepoLens adds the constant specific-versus-GC funding overlay and locates
    the nearest zero crossing in the economically relevant direction.

    The root is linearly interpolated between the two full-repricing scenario
    points that bracket zero. This is deliberately transparent and avoids
    presenting a false exactness beyond the supplied scenario grid.
    """

    current_spread_bp: float
    funding_overlay_eur: float
    local_cash_pnl_per_spread_bp_eur: float
    first_order_equivalent_spread_move_bp: float
    breakeven_spread_shock_bp: float | None
    breakeven_spread_level_bp: float | None
    breakeven_move_magnitude_bp: float | None
    direction: str
    interpretation: str
    lower_bracket_shock_bp: float | None
    upper_bracket_shock_bp: float | None
    interpolation_width_bp: float | None
    within_scenario_range: bool


def _validated_spread_scenario_points(
    scenario_points: Sequence[SpreadScenarioPoint],
) -> tuple[SpreadScenarioPoint, ...]:
    """
    Validate and sort a spread-scenario curve.

    A zero-shock point with zero cash P&L is inserted when omitted because
    the current position is the cash-RV valuation reference point.
    """
    if not scenario_points:
        raise RepoAdjustedRelativeValueValidationError(
            "scenario_points must not be empty."
        )

    points_by_shock: dict[float, SpreadScenarioPoint] = {}

    for point in scenario_points:
        shock = float(
            point.spread_shock_bp
        )

        if shock in points_by_shock:
            raise RepoAdjustedRelativeValueValidationError(
                "scenario_points must not contain duplicate spread shocks."
            )

        points_by_shock[
            shock
        ] = point

    if 0.0 in points_by_shock:
        zero_point = points_by_shock[
            0.0
        ]

        if abs(
            zero_point.cash_pnl_eur
        ) > 1e-8:
            raise RepoAdjustedRelativeValueValidationError(
                "The zero-shock cash P&L must be zero."
            )
    else:
        points_by_shock[
            0.0
        ] = SpreadScenarioPoint(
            spread_shock_bp=0.0,
            cash_pnl_eur=0.0,
        )

    ordered = tuple(
        points_by_shock[
            shock
        ]
        for shock in sorted(
            points_by_shock
        )
    )

    has_negative = any(
        point.spread_shock_bp < 0.0
        for point in ordered
    )

    has_positive = any(
        point.spread_shock_bp > 0.0
        for point in ordered
    )

    if not (
        has_negative
        and has_positive
    ):
        raise RepoAdjustedRelativeValueValidationError(
            "scenario_points must include both negative and positive spread shocks."
        )

    return ordered


def _local_spread_pnl_slope(
    points: Sequence[SpreadScenarioPoint],
) -> float:
    """
    Estimate local cash-RV P&L per spread bp around the current spread.

    The slope uses the nearest full-repricing point on either side of zero.
    """
    negative_points = [
        point
        for point in points
        if point.spread_shock_bp < 0.0
    ]

    positive_points = [
        point
        for point in points
        if point.spread_shock_bp > 0.0
    ]

    nearest_negative = max(
        negative_points,
        key=lambda point: point.spread_shock_bp,
    )

    nearest_positive = min(
        positive_points,
        key=lambda point: point.spread_shock_bp,
    )

    shock_width = (
        nearest_positive.spread_shock_bp
        - nearest_negative.spread_shock_bp
    )

    if shock_width <= 0.0:
        raise RepoAdjustedRelativeValueValidationError(
            "Unable to determine a valid local spread-scenario slope."
        )

    slope = (
        nearest_positive.cash_pnl_eur
        - nearest_negative.cash_pnl_eur
    ) / shock_width

    if abs(
        slope
    ) <= 1e-12:
        raise RepoAdjustedRelativeValueValidationError(
            "The local spread P&L slope is zero; a spread breakeven cannot be inferred."
        )

    return float(
        slope
    )


def analyse_funding_adjusted_spread_breakeven(
    *,
    current_spread_bp: float,
    funding_overlay_eur: float,
    scenario_points: Sequence[SpreadScenarioPoint],
) -> FundingAdjustedSpreadBreakeven:
    """
    Translate the repo funding overlay into a cash-spread breakeven.

    For a positive funding overlay, the result answers:

        How far can the cash spread move in the adverse direction before
        the specific-versus-GC funding benefit is consumed?

    For a negative funding overlay, the result instead reports the favourable
    spread move required to offset the funding drag.

    Cash P&L comes from the supplied full-repricing spread scenarios. The
    breakeven root is linearly interpolated only between scenario points that
    bracket zero after adding the funding overlay.
    """
    if not isfinite(
        current_spread_bp
    ):
        raise RepoAdjustedRelativeValueValidationError(
            "current_spread_bp must be finite."
        )

    if not isfinite(
        funding_overlay_eur
    ):
        raise RepoAdjustedRelativeValueValidationError(
            "funding_overlay_eur must be finite."
        )

    points = _validated_spread_scenario_points(
        scenario_points
    )

    local_slope = _local_spread_pnl_slope(
        points
    )

    first_order_move = (
        -funding_overlay_eur
        / local_slope
    )

    if abs(
        funding_overlay_eur
    ) <= 1e-8:
        return FundingAdjustedSpreadBreakeven(
            current_spread_bp=float(
                current_spread_bp
            ),
            funding_overlay_eur=0.0,
            local_cash_pnl_per_spread_bp_eur=(
                local_slope
            ),
            first_order_equivalent_spread_move_bp=0.0,
            breakeven_spread_shock_bp=0.0,
            breakeven_spread_level_bp=float(
                current_spread_bp
            ),
            breakeven_move_magnitude_bp=0.0,
            direction="NONE",
            interpretation=(
                "No repo funding edge or drag versus GC is present."
            ),
            lower_bracket_shock_bp=0.0,
            upper_bracket_shock_bp=0.0,
            interpolation_width_bp=0.0,
            within_scenario_range=True,
        )

    target_direction = (
        -1.0
        if first_order_move < 0.0
        else 1.0
    )

    if funding_overlay_eur > 0.0:
        interpretation = (
            "ADVERSE_MOVE_CAPACITY"
        )
    else:
        interpretation = (
            "FAVOURABLE_MOVE_REQUIRED"
        )

    direction = (
        "NARROWING"
        if target_direction < 0.0
        else "WIDENING"
    )

    zero_point = next(
        point
        for point in points
        if point.spread_shock_bp == 0.0
    )

    directional_points = [
        zero_point,
        *sorted(
            [
                point
                for point in points
                if (
                    point.spread_shock_bp
                    * target_direction
                    > 0.0
                )
            ],
            key=lambda point: abs(
                point.spread_shock_bp
            ),
        ),
    ]

    previous_point = directional_points[
        0
    ]

    previous_adjusted_pnl = (
        previous_point.cash_pnl_eur
        + funding_overlay_eur
    )

    for current_point in directional_points[
        1:
    ]:
        current_adjusted_pnl = (
            current_point.cash_pnl_eur
            + funding_overlay_eur
        )

        if abs(
            current_adjusted_pnl
        ) <= 1e-8:
            root_shock = float(
                current_point.spread_shock_bp
            )

            lower_shock = min(
                previous_point.spread_shock_bp,
                current_point.spread_shock_bp,
            )

            upper_shock = max(
                previous_point.spread_shock_bp,
                current_point.spread_shock_bp,
            )

            return FundingAdjustedSpreadBreakeven(
                current_spread_bp=float(
                    current_spread_bp
                ),
                funding_overlay_eur=float(
                    funding_overlay_eur
                ),
                local_cash_pnl_per_spread_bp_eur=(
                    local_slope
                ),
                first_order_equivalent_spread_move_bp=(
                    first_order_move
                ),
                breakeven_spread_shock_bp=root_shock,
                breakeven_spread_level_bp=(
                    float(
                        current_spread_bp
                    )
                    + root_shock
                ),
                breakeven_move_magnitude_bp=abs(
                    root_shock
                ),
                direction=direction,
                interpretation=interpretation,
                lower_bracket_shock_bp=(
                    float(
                        lower_shock
                    )
                ),
                upper_bracket_shock_bp=(
                    float(
                        upper_shock
                    )
                ),
                interpolation_width_bp=abs(
                    current_point.spread_shock_bp
                    - previous_point.spread_shock_bp
                ),
                within_scenario_range=True,
            )

        if (
            previous_adjusted_pnl
            * current_adjusted_pnl
            < 0.0
        ):
            pnl_change = (
                current_adjusted_pnl
                - previous_adjusted_pnl
            )

            if abs(
                pnl_change
            ) <= 1e-12:
                raise RepoAdjustedRelativeValueValidationError(
                    "Unable to interpolate the spread breakeven from a flat P&L bracket."
                )

            interpolation_fraction = (
                -previous_adjusted_pnl
                / pnl_change
            )

            root_shock = (
                previous_point.spread_shock_bp
                + interpolation_fraction
                * (
                    current_point.spread_shock_bp
                    - previous_point.spread_shock_bp
                )
            )

            lower_shock = min(
                previous_point.spread_shock_bp,
                current_point.spread_shock_bp,
            )

            upper_shock = max(
                previous_point.spread_shock_bp,
                current_point.spread_shock_bp,
            )

            return FundingAdjustedSpreadBreakeven(
                current_spread_bp=float(
                    current_spread_bp
                ),
                funding_overlay_eur=float(
                    funding_overlay_eur
                ),
                local_cash_pnl_per_spread_bp_eur=(
                    local_slope
                ),
                first_order_equivalent_spread_move_bp=(
                    first_order_move
                ),
                breakeven_spread_shock_bp=float(
                    root_shock
                ),
                breakeven_spread_level_bp=(
                    float(
                        current_spread_bp
                    )
                    + float(
                        root_shock
                    )
                ),
                breakeven_move_magnitude_bp=abs(
                    float(
                        root_shock
                    )
                ),
                direction=direction,
                interpretation=interpretation,
                lower_bracket_shock_bp=float(
                    lower_shock
                ),
                upper_bracket_shock_bp=float(
                    upper_shock
                ),
                interpolation_width_bp=abs(
                    current_point.spread_shock_bp
                    - previous_point.spread_shock_bp
                ),
                within_scenario_range=True,
            )

        previous_point = current_point
        previous_adjusted_pnl = (
            current_adjusted_pnl
        )

    return FundingAdjustedSpreadBreakeven(
        current_spread_bp=float(
            current_spread_bp
        ),
        funding_overlay_eur=float(
            funding_overlay_eur
        ),
        local_cash_pnl_per_spread_bp_eur=(
            local_slope
        ),
        first_order_equivalent_spread_move_bp=(
            first_order_move
        ),
        breakeven_spread_shock_bp=None,
        breakeven_spread_level_bp=None,
        breakeven_move_magnitude_bp=None,
        direction=direction,
        interpretation=interpretation,
        lower_bracket_shock_bp=None,
        upper_bracket_shock_bp=None,
        interpolation_width_bp=None,
        within_scenario_range=False,
    )

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