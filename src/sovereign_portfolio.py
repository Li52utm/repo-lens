from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
)
from src.sovereign_relative_value import PositionDirection
from src.sovereign_snapshot import (
    SovereignSnapshotResult,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    snapshot_scenarios,
)


DEFAULT_PARALLEL_SHOCKS_BP: Final[tuple[float, ...]] = (
    -25.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    25.0,
)

DEFAULT_ITALY_GERMANY_SPREAD_SHOCKS_BP: Final[
    tuple[float, ...]
] = (
    -25.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    25.0,
)

DEFAULT_CONCENTRATION_WARNING_THRESHOLD: Final[float] = 0.50


class SovereignPortfolioError(RuntimeError):
    """
    Base exception for RepoLens sovereign portfolio analytics.
    """


class SovereignPortfolioValidationError(
    SovereignPortfolioError
):
    """
    Raised when sovereign portfolio inputs fail validation.
    """


@dataclass(frozen=True)
class SovereignPortfolioPosition:
    """
    Define one long or short sovereign bond position.

    notional_eur is the absolute face value. Position direction is stored
    separately so negative notionals are not accepted.
    """

    position_id: str
    instrument: SovereignInstrument
    direction: PositionDirection
    notional_eur: float
    yield_input: SovereignYieldInput | None = None

    def __post_init__(self) -> None:
        if not self.position_id.strip():
            raise SovereignPortfolioValidationError(
                "position_id must not be empty."
            )

        if not np.isfinite(
            self.notional_eur
        ):
            raise SovereignPortfolioValidationError(
                "notional_eur must be finite."
            )

        if self.notional_eur <= 0.0:
            raise SovereignPortfolioValidationError(
                "notional_eur must be positive."
            )

        if (
            self.yield_input is not None
            and self.yield_input.isin.strip().upper()
            != self.instrument.isin
        ):
            raise SovereignPortfolioValidationError(
                "yield_input ISIN must match the position instrument."
            )


@dataclass(frozen=True)
class SovereignPortfolioPositionResult:
    """
    Store calculated valuation and risk for one portfolio position.
    """

    position_id: str
    instrument: SovereignInstrument
    direction: PositionDirection
    notional_eur: float
    signed_notional_eur: float
    snapshot: SovereignSnapshotResult
    market_value_eur: float
    signed_market_value_eur: float
    absolute_dv01_eur: float
    signed_dv01_eur: float


@dataclass(frozen=True)
class SovereignPortfolioSummary:
    """
    Store headline portfolio risk measures.
    """

    position_count: int
    gross_notional_eur: float
    net_notional_eur: float
    gross_market_value_eur: float
    net_market_value_eur: float
    gross_dv01_eur: float
    net_dv01_eur: float
    largest_risk_position_id: str
    largest_risk_position_dv01_eur: float
    largest_risk_share: float
    concentration_warning: bool
    concentration_warning_threshold: float
    settlement_date: date


def validate_positions(
    positions: tuple[
        SovereignPortfolioPosition,
        ...,
    ],
) -> None:
    """
    Validate portfolio-level position requirements.
    """
    if not positions:
        raise SovereignPortfolioValidationError(
            "positions must not be empty."
        )

    position_ids = [
        position.position_id.strip()
        for position in positions
    ]

    if len(
        position_ids
    ) != len(
        set(
            position_ids
        )
    ):
        raise SovereignPortfolioValidationError(
            "position_id values must be unique."
        )


def validate_concentration_threshold(
    threshold: float,
) -> None:
    """
    Validate the portfolio concentration threshold.
    """
    if not np.isfinite(
        threshold
    ):
        raise SovereignPortfolioValidationError(
            "concentration threshold must be finite."
        )

    if not 0.0 < threshold <= 1.0:
        raise SovereignPortfolioValidationError(
            "concentration threshold must be greater than 0 and "
            "less than or equal to 1."
        )


def build_position_result(
    position: SovereignPortfolioPosition,
    german_curve: pd.DataFrame,
    settlement_date: date,
) -> SovereignPortfolioPositionResult:
    """
    Value one portfolio position and calculate signed exposure.
    """
    try:
        snapshot = build_instrument_snapshot(
            instrument=position.instrument,
            german_curve=german_curve,
            settlement_date=settlement_date,
            position_notional_eur=position.notional_eur,
            explicit_yield_input=position.yield_input,
        )
    except SovereignSnapshotValidationError as error:
        raise SovereignPortfolioValidationError(
            f"Position {position.position_id}: {error}"
        ) from error

    signed_notional_eur = (
        position.direction.sign
        * position.notional_eur
    )

    market_value_eur = (
        snapshot.dirty_price
        / 100.0
        * position.notional_eur
    )

    signed_market_value_eur = (
        position.direction.sign
        * market_value_eur
    )

    absolute_dv01_eur = (
        snapshot.position_dv01_eur
    )

    signed_dv01_eur = (
        position.direction.sign
        * absolute_dv01_eur
    )

    return SovereignPortfolioPositionResult(
        position_id=position.position_id.strip(),
        instrument=position.instrument,
        direction=position.direction,
        notional_eur=position.notional_eur,
        signed_notional_eur=signed_notional_eur,
        snapshot=snapshot,
        market_value_eur=market_value_eur,
        signed_market_value_eur=signed_market_value_eur,
        absolute_dv01_eur=absolute_dv01_eur,
        signed_dv01_eur=signed_dv01_eur,
    )


def build_portfolio_results(
    positions: tuple[
        SovereignPortfolioPosition,
        ...,
    ],
    german_curve: pd.DataFrame,
    settlement_date: date,
) -> tuple[
    SovereignPortfolioPositionResult,
    ...,
]:
    """
    Calculate valuation and risk for every portfolio position.
    """
    validate_positions(
        positions
    )

    return tuple(
        build_position_result(
            position=position,
            german_curve=german_curve,
            settlement_date=settlement_date,
        )
        for position in positions
    )


def summarise_portfolio(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
    settlement_date: date,
    concentration_warning_threshold: float = (
        DEFAULT_CONCENTRATION_WARNING_THRESHOLD
    ),
) -> SovereignPortfolioSummary:
    """
    Calculate portfolio headline risk measures.
    """
    if not results:
        raise SovereignPortfolioValidationError(
            "results must not be empty."
        )

    validate_concentration_threshold(
        concentration_warning_threshold
    )

    gross_notional_eur = sum(
        result.notional_eur
        for result in results
    )

    net_notional_eur = sum(
        result.signed_notional_eur
        for result in results
    )

    gross_market_value_eur = sum(
        result.market_value_eur
        for result in results
    )

    net_market_value_eur = sum(
        result.signed_market_value_eur
        for result in results
    )

    gross_dv01_eur = sum(
        result.absolute_dv01_eur
        for result in results
    )

    net_dv01_eur = sum(
        result.signed_dv01_eur
        for result in results
    )

    largest_result = max(
        results,
        key=lambda result: result.absolute_dv01_eur,
    )

    if gross_dv01_eur > 0.0:
        largest_risk_share = (
            largest_result.absolute_dv01_eur
            / gross_dv01_eur
        )
    else:
        largest_risk_share = 0.0

    return SovereignPortfolioSummary(
        position_count=len(
            results
        ),
        gross_notional_eur=gross_notional_eur,
        net_notional_eur=net_notional_eur,
        gross_market_value_eur=gross_market_value_eur,
        net_market_value_eur=net_market_value_eur,
        gross_dv01_eur=gross_dv01_eur,
        net_dv01_eur=net_dv01_eur,
        largest_risk_position_id=(
            largest_result.position_id
        ),
        largest_risk_position_dv01_eur=(
            largest_result.absolute_dv01_eur
        ),
        largest_risk_share=largest_risk_share,
        concentration_warning=(
            largest_risk_share
            >= concentration_warning_threshold
        ),
        concentration_warning_threshold=(
            concentration_warning_threshold
        ),
        settlement_date=settlement_date,
    )


def build_portfolio(
    positions: tuple[
        SovereignPortfolioPosition,
        ...,
    ],
    german_curve: pd.DataFrame,
    settlement_date: date,
    concentration_warning_threshold: float = (
        DEFAULT_CONCENTRATION_WARNING_THRESHOLD
    ),
) -> tuple[
    tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
    SovereignPortfolioSummary,
]:
    """
    Build complete position results and portfolio summary.
    """
    results = build_portfolio_results(
        positions=positions,
        german_curve=german_curve,
        settlement_date=settlement_date,
    )

    summary = summarise_portfolio(
        results=results,
        settlement_date=settlement_date,
        concentration_warning_threshold=(
            concentration_warning_threshold
        ),
    )

    return (
        results,
        summary,
    )


def positions_to_frame(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Convert portfolio positions into a dashboard-friendly table.
    """
    if not results:
        raise SovereignPortfolioValidationError(
            "results must not be empty."
        )

    rows = [
        {
            "position_id": result.position_id,
            "isin": result.instrument.isin,
            "display_name": result.instrument.display_name,
            "country": result.instrument.country.value,
            "security_type": (
                result.instrument.security_type.value
            ),
            "benchmark_tenor_years": (
                result.instrument.benchmark_tenor_years
            ),
            "direction": result.direction.value,
            "notional_eur": result.notional_eur,
            "signed_notional_eur": (
                result.signed_notional_eur
            ),
            "yield_percent": (
                result.snapshot.yield_percent
            ),
            "dirty_price": result.snapshot.dirty_price,
            "market_value_eur": result.market_value_eur,
            "signed_market_value_eur": (
                result.signed_market_value_eur
            ),
            "absolute_dv01_eur": (
                result.absolute_dv01_eur
            ),
            "signed_dv01_eur": result.signed_dv01_eur,
            "data_status": (
                result.snapshot.data_status.value
            ),
            "source_name": result.snapshot.source_name,
        }
        for result in results
    ]

    return pd.DataFrame(
        rows
    )


def aggregate_dv01_by_country(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Aggregate signed and gross DV01 by sovereign country.
    """
    positions = positions_to_frame(
        results
    )

    grouped = (
        positions
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            gross_dv01_eur=(
                "absolute_dv01_eur",
                "sum",
            ),
            net_dv01_eur=(
                "signed_dv01_eur",
                "sum",
            ),
            gross_notional_eur=(
                "notional_eur",
                "sum",
            ),
            net_notional_eur=(
                "signed_notional_eur",
                "sum",
            ),
            position_count=(
                "position_id",
                "count",
            ),
        )
        .sort_values(
            "country"
        )
        .reset_index(
            drop=True
        )
    )

    return grouped


def aggregate_dv01_by_tenor(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Aggregate signed and gross DV01 by benchmark tenor.
    """
    positions = positions_to_frame(
        results
    )

    grouped = (
        positions
        .groupby(
            "benchmark_tenor_years",
            as_index=False,
        )
        .agg(
            gross_dv01_eur=(
                "absolute_dv01_eur",
                "sum",
            ),
            net_dv01_eur=(
                "signed_dv01_eur",
                "sum",
            ),
            gross_notional_eur=(
                "notional_eur",
                "sum",
            ),
            net_notional_eur=(
                "signed_notional_eur",
                "sum",
            ),
            position_count=(
                "position_id",
                "count",
            ),
        )
        .sort_values(
            "benchmark_tenor_years"
        )
        .reset_index(
            drop=True
        )
    )

    return grouped


def calculate_position_scenario_pnl(
    result: SovereignPortfolioPositionResult,
    yield_shock_bp: float,
) -> float:
    """
    Calculate signed full-repricing P&L for one position.
    """
    if not np.isfinite(
        yield_shock_bp
    ):
        raise SovereignPortfolioValidationError(
            "yield_shock_bp must be finite."
        )

    scenarios = snapshot_scenarios(
        instrument=result.instrument,
        settlement_date=result.snapshot.settlement_date,
        yield_percent=result.snapshot.yield_percent,
        position_notional_eur=result.notional_eur,
        yield_shocks_bp=(
            yield_shock_bp,
        ),
    )

    unsigned_pnl_eur = float(
        scenarios.iloc[0][
            "position_pnl_eur"
        ]
    )

    return (
        result.direction.sign
        * unsigned_pnl_eur
    )


def build_parallel_portfolio_scenarios(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
    yield_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_PARALLEL_SHOCKS_BP,
) -> pd.DataFrame:
    """
    Reprice the full portfolio under parallel yield shocks.
    """
    if not results:
        raise SovereignPortfolioValidationError(
            "results must not be empty."
        )

    if not yield_shocks_bp:
        raise SovereignPortfolioValidationError(
            "yield_shocks_bp must not be empty."
        )

    rows: list[
        dict[str, float]
    ] = []

    for yield_shock_bp in yield_shocks_bp:
        if not np.isfinite(
            yield_shock_bp
        ):
            raise SovereignPortfolioValidationError(
                "Yield shocks must be finite."
            )

        position_pnls = [
            calculate_position_scenario_pnl(
                result=result,
                yield_shock_bp=yield_shock_bp,
            )
            for result in results
        ]

        rows.append(
            {
                "yield_shock_bp": yield_shock_bp,
                "portfolio_pnl_eur": sum(
                    position_pnls
                ),
                "largest_gain_eur": max(
                    position_pnls
                ),
                "largest_loss_eur": min(
                    position_pnls
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_italy_germany_spread_scenarios(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
    spread_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_ITALY_GERMANY_SPREAD_SHOCKS_BP,
) -> pd.DataFrame:
    """
    Reprice the portfolio under Italy-versus-Germany spread shocks.

    The requested spread shock is divided equally:

        Italian yield shock = spread shock / 2
        German yield shock = -spread shock / 2

    A positive shock therefore widens Italian yields relative to German
    yields while keeping the cross-country average yield approximately
    unchanged.
    """
    if not results:
        raise SovereignPortfolioValidationError(
            "results must not be empty."
        )

    if not spread_shocks_bp:
        raise SovereignPortfolioValidationError(
            "spread_shocks_bp must not be empty."
        )

    rows: list[
        dict[str, float]
    ] = []

    for spread_shock_bp in spread_shocks_bp:
        if not np.isfinite(
            spread_shock_bp
        ):
            raise SovereignPortfolioValidationError(
                "Spread shocks must be finite."
            )

        italy_yield_shock_bp = (
            spread_shock_bp
            / 2.0
        )

        germany_yield_shock_bp = (
            -spread_shock_bp
            / 2.0
        )

        italy_pnl_eur = 0.0
        germany_pnl_eur = 0.0

        for result in results:
            if (
                result.instrument.country
                == SovereignCountry.ITALY
            ):
                position_pnl_eur = (
                    calculate_position_scenario_pnl(
                        result=result,
                        yield_shock_bp=(
                            italy_yield_shock_bp
                        ),
                    )
                )

                italy_pnl_eur += position_pnl_eur

            elif (
                result.instrument.country
                == SovereignCountry.GERMANY
            ):
                position_pnl_eur = (
                    calculate_position_scenario_pnl(
                        result=result,
                        yield_shock_bp=(
                            germany_yield_shock_bp
                        ),
                    )
                )

                germany_pnl_eur += position_pnl_eur

            else:
                raise SovereignPortfolioValidationError(
                    "Italy-Germany spread scenarios support only "
                    "Italian and German instruments."
                )

        rows.append(
            {
                "spread_shock_bp": spread_shock_bp,
                "italy_yield_shock_bp": (
                    italy_yield_shock_bp
                ),
                "germany_yield_shock_bp": (
                    germany_yield_shock_bp
                ),
                "italy_pnl_eur": italy_pnl_eur,
                "germany_pnl_eur": germany_pnl_eur,
                "portfolio_pnl_eur": (
                    italy_pnl_eur
                    + germany_pnl_eur
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def risk_contribution_frame(
    results: tuple[
        SovereignPortfolioPositionResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Calculate each position's share of gross portfolio DV01.
    """
    positions = positions_to_frame(
        results
    )

    gross_dv01_eur = float(
        positions[
            "absolute_dv01_eur"
        ].sum()
    )

    if gross_dv01_eur <= 0.0:
        positions[
            "gross_dv01_share"
        ] = 0.0
    else:
        positions[
            "gross_dv01_share"
        ] = (
            positions[
                "absolute_dv01_eur"
            ]
            / gross_dv01_eur
        )

    return (
        positions[
            [
                "position_id",
                "isin",
                "display_name",
                "country",
                "benchmark_tenor_years",
                "direction",
                "absolute_dv01_eur",
                "signed_dv01_eur",
                "gross_dv01_share",
            ]
        ]
        .sort_values(
            "absolute_dv01_eur",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )