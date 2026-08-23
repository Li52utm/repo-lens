from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from src.bond_analytics import FixedRateBond
from src.repo_analytics import (
    BondCarryAnalysis,
    RepoTradeInput,
    RepoValidationError,
    analyse_financed_bond_carry,
)


DEFAULT_REPO_RATE_SHOCKS_BP: Final[tuple[float, ...]] = (
    -50.0,
    -25.0,
    -10.0,
    0.0,
    10.0,
    25.0,
    50.0,
)


class RepoAdjustedCarryError(RuntimeError):
    """Base exception for RepoLens repo-adjusted carry analytics."""


class RepoAdjustedCarryValidationError(RepoAdjustedCarryError):
    """Raised when repo-adjusted carry inputs are inconsistent."""


@dataclass(frozen=True)
class RepoRateCarryScenario:
    """One unchanged-yield carry scenario under a shocked specific repo rate."""

    repo_rate_shock_bp: float
    shocked_specific_repo_rate_percent: float
    specialness_vs_gc_bp: float
    financing_adjusted_pnl_eur: float
    financing_adjusted_pnl_per_eur_1m_face: float
    financing_advantage_vs_gc_eur: float
    financing_advantage_vs_gc_per_eur_1m_face: float


@dataclass(frozen=True)
class RepoAdjustedCarryAnalysis:
    """Compare coupon-bond carry under specific funding and GC funding."""

    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float
    start_yield_percent: float
    repo_days: int
    start_market_value_eur: float
    coupon_income_eur: float
    specific_repo_interest_eur: float
    gc_repo_interest_eur: float
    financing_advantage_vs_gc_eur: float
    financing_advantage_vs_gc_per_eur_1m_face: float
    unchanged_yield_specific_pnl_eur: float
    unchanged_yield_gc_pnl_eur: float
    unchanged_yield_specific_pnl_per_eur_1m_face: float
    unchanged_yield_gc_pnl_per_eur_1m_face: float
    specific_breakeven_exit_yield_percent: float | None
    gc_breakeven_exit_yield_percent: float | None
    specific_breakeven_yield_move_bp: float | None
    gc_breakeven_yield_move_bp: float | None
    breakeven_yield_advantage_bp: float | None
    repo_rate_scenarios: tuple[RepoRateCarryScenario, ...]


def _zero_yield_scenario(analysis: BondCarryAnalysis):
    matches = tuple(
        scenario
        for scenario in analysis.scenarios
        if abs(scenario.yield_shock_bp) < 1e-12
    )

    if len(matches) != 1:
        raise RepoAdjustedCarryValidationError(
            "Carry analysis must contain exactly one 0 bp yield scenario."
        )

    return matches[0]


def _per_eur_1m(
    *,
    amount_eur: float,
    face_value_eur: float,
) -> float:
    if face_value_eur <= 0.0:
        raise RepoAdjustedCarryValidationError(
            "face_value_eur must be positive."
        )

    return amount_eur * 1_000_000.0 / face_value_eur


def _gc_trade(
    *,
    specific_trade: RepoTradeInput,
    gc_repo_rate_percent: float,
) -> RepoTradeInput:
    return replace(
        specific_trade,
        repo_rate_percent=gc_repo_rate_percent,
        gc_repo_rate_percent=None,
    )


def _specific_trade_without_embedded_gc(
    *,
    trade: RepoTradeInput,
) -> RepoTradeInput:
    return replace(
        trade,
        gc_repo_rate_percent=None,
    )


def analyse_repo_adjusted_bond_carry(
    *,
    bond: FixedRateBond,
    trade: RepoTradeInput,
    gc_repo_rate_percent: float,
    repo_rate_shocks_bp: tuple[float, ...] = DEFAULT_REPO_RATE_SHOCKS_BP,
) -> RepoAdjustedCarryAnalysis:
    """
    Compare coupon-bond carry under specific repo and a GC funding reference.

    The base comparison changes only the financing rate. Bond terms, face
    value, price, accrued interest, haircut, dates and day-count convention
    remain identical.

    Repo-rate shocks hold bond yield unchanged and vary only the specific repo
    rate, isolating funding sensitivity from cash-bond yield risk.

    Transaction costs, fail charges, variation margin, haircut opportunity
    cost, settlement-calendar effects and future repo roll paths are excluded.
    """
    if not repo_rate_shocks_bp:
        raise RepoAdjustedCarryValidationError(
            "repo_rate_shocks_bp must not be empty."
        )

    if len(set(repo_rate_shocks_bp)) != len(repo_rate_shocks_bp):
        raise RepoAdjustedCarryValidationError(
            "repo_rate_shocks_bp must not contain duplicate shocks."
        )

    specific_trade = _specific_trade_without_embedded_gc(
        trade=trade
    )

    gc_trade = _gc_trade(
        specific_trade=specific_trade,
        gc_repo_rate_percent=float(gc_repo_rate_percent),
    )

    try:
        specific_analysis = analyse_financed_bond_carry(
            bond=bond,
            trade=specific_trade,
            yield_shocks_bp=(0.0,),
        )

        gc_analysis = analyse_financed_bond_carry(
            bond=bond,
            trade=gc_trade,
            yield_shocks_bp=(0.0,),
        )

    except RepoValidationError as error:
        raise RepoAdjustedCarryValidationError(
            str(error)
        ) from error

    specific_zero = _zero_yield_scenario(
        specific_analysis
    )
    gc_zero = _zero_yield_scenario(
        gc_analysis
    )

    repo_days = (
        specific_trade.repurchase_date
        - specific_trade.purchase_date
    ).days

    specialness_bp = (
        float(gc_repo_rate_percent)
        - specific_trade.repo_rate_percent
    ) * 100.0

    financing_advantage = (
        gc_analysis.repo_interest_eur
        - specific_analysis.repo_interest_eur
    )

    financing_advantage_per_eur_1m = _per_eur_1m(
        amount_eur=financing_advantage,
        face_value_eur=specific_trade.face_value_eur,
    )

    breakeven_yield_advantage_bp: float | None = None

    if (
        specific_analysis.breakeven_exit_yield_percent is not None
        and gc_analysis.breakeven_exit_yield_percent is not None
    ):
        breakeven_yield_advantage_bp = (
            specific_analysis.breakeven_exit_yield_percent
            - gc_analysis.breakeven_exit_yield_percent
        ) * 100.0

    repo_scenarios: list[RepoRateCarryScenario] = []

    for shock_bp in repo_rate_shocks_bp:
        shocked_rate = (
            specific_trade.repo_rate_percent
            + shock_bp / 100.0
        )

        shocked_trade = replace(
            specific_trade,
            repo_rate_percent=shocked_rate,
        )

        try:
            shocked_analysis = analyse_financed_bond_carry(
                bond=bond,
                trade=shocked_trade,
                yield_shocks_bp=(0.0,),
            )
        except RepoValidationError as error:
            raise RepoAdjustedCarryValidationError(
                "Unable to calculate repo-rate carry shock."
            ) from error

        shocked_zero = _zero_yield_scenario(
            shocked_analysis
        )

        shocked_advantage = (
            gc_analysis.repo_interest_eur
            - shocked_analysis.repo_interest_eur
        )

        repo_scenarios.append(
            RepoRateCarryScenario(
                repo_rate_shock_bp=float(shock_bp),
                shocked_specific_repo_rate_percent=shocked_rate,
                specialness_vs_gc_bp=(
                    float(gc_repo_rate_percent)
                    - shocked_rate
                ) * 100.0,
                financing_adjusted_pnl_eur=(
                    shocked_zero.financing_adjusted_pnl_eur
                ),
                financing_adjusted_pnl_per_eur_1m_face=(
                    shocked_zero.financing_adjusted_pnl_per_eur_1m_face
                ),
                financing_advantage_vs_gc_eur=shocked_advantage,
                financing_advantage_vs_gc_per_eur_1m_face=(
                    _per_eur_1m(
                        amount_eur=shocked_advantage,
                        face_value_eur=specific_trade.face_value_eur,
                    )
                ),
            )
        )

    return RepoAdjustedCarryAnalysis(
        specific_repo_rate_percent=specific_trade.repo_rate_percent,
        gc_repo_rate_percent=float(gc_repo_rate_percent),
        specialness_bp=specialness_bp,
        start_yield_percent=specific_analysis.start_yield_percent,
        repo_days=repo_days,
        start_market_value_eur=specific_analysis.start_market_value_eur,
        coupon_income_eur=specific_analysis.coupon_income_eur,
        specific_repo_interest_eur=specific_analysis.repo_interest_eur,
        gc_repo_interest_eur=gc_analysis.repo_interest_eur,
        financing_advantage_vs_gc_eur=financing_advantage,
        financing_advantage_vs_gc_per_eur_1m_face=(
            financing_advantage_per_eur_1m
        ),
        unchanged_yield_specific_pnl_eur=(
            specific_zero.financing_adjusted_pnl_eur
        ),
        unchanged_yield_gc_pnl_eur=(
            gc_zero.financing_adjusted_pnl_eur
        ),
        unchanged_yield_specific_pnl_per_eur_1m_face=(
            specific_zero.financing_adjusted_pnl_per_eur_1m_face
        ),
        unchanged_yield_gc_pnl_per_eur_1m_face=(
            gc_zero.financing_adjusted_pnl_per_eur_1m_face
        ),
        specific_breakeven_exit_yield_percent=(
            specific_analysis.breakeven_exit_yield_percent
        ),
        gc_breakeven_exit_yield_percent=(
            gc_analysis.breakeven_exit_yield_percent
        ),
        specific_breakeven_yield_move_bp=(
            specific_analysis.breakeven_yield_move_bp
        ),
        gc_breakeven_yield_move_bp=(
            gc_analysis.breakeven_yield_move_bp
        ),
        breakeven_yield_advantage_bp=breakeven_yield_advantage_bp,
        repo_rate_scenarios=tuple(repo_scenarios),
    )