from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from src.sovereign_instruments import SovereignInstrument
from src.sovereign_snapshot import (
    SovereignSnapshotResult,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    snapshot_scenarios,
)


DEFAULT_ANCHOR_NOTIONAL_EUR: Final[float] = 10_000_000.0

DEFAULT_SPREAD_SHOCKS_BP: Final[tuple[float, ...]] = (
    -25.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    25.0,
)

DEFAULT_PARALLEL_SHOCKS_BP: Final[tuple[float, ...]] = (
    -25.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    25.0,
)


class RelativeValueError(RuntimeError):
    """
    Base exception for RepoLens sovereign relative-value analytics.
    """


class RelativeValueValidationError(RelativeValueError):
    """
    Raised when relative-value inputs fail validation.
    """


class PositionDirection(StrEnum):
    """
    Supported sovereign position directions.
    """

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> float:
        """
        Return the numerical position sign.
        """
        if self == PositionDirection.LONG:
            return 1.0

        return -1.0


@dataclass(frozen=True)
class RelativeValueLeg:
    """
    Define one leg of a sovereign relative-value position.
    """

    instrument: SovereignInstrument
    direction: PositionDirection
    yield_input: SovereignYieldInput | None = None


@dataclass(frozen=True)
class RelativeValuePosition:
    """
    Store one DV01-neutral two-leg sovereign position.
    """

    anchor_isin: str
    hedge_isin: str
    anchor_direction: PositionDirection
    hedge_direction: PositionDirection
    anchor_notional_eur: float
    hedge_notional_eur: float
    hedge_notional_per_anchor_euro: float
    anchor_yield_percent: float
    hedge_yield_percent: float
    spread_bp: float
    anchor_position_dv01_eur: float
    hedge_position_dv01_eur: float
    signed_anchor_dv01_eur: float
    signed_hedge_dv01_eur: float
    net_dv01_eur: float
    gross_dv01_eur: float
    dv01_hedge_error_eur: float
    settlement_date: date


def validate_positive_notional(
    notional_eur: float,
) -> None:
    """
    Validate a strictly positive position notional.
    """
    if not np.isfinite(
        notional_eur
    ):
        raise RelativeValueValidationError(
            "anchor_notional_eur must be finite."
        )

    if notional_eur <= 0.0:
        raise RelativeValueValidationError(
            "anchor_notional_eur must be positive."
        )


def validate_relative_value_legs(
    anchor_leg: RelativeValueLeg,
    hedge_leg: RelativeValueLeg,
) -> None:
    """
    Validate the two instruments and position directions.
    """
    if (
        anchor_leg.instrument.isin
        == hedge_leg.instrument.isin
    ):
        raise RelativeValueValidationError(
            "Anchor and hedge instruments must be different."
        )

    if (
        anchor_leg.direction
        == hedge_leg.direction
    ):
        raise RelativeValueValidationError(
            "Anchor and hedge legs must have opposite directions."
        )

    if (
        anchor_leg.instrument.currency
        != hedge_leg.instrument.currency
    ):
        raise RelativeValueValidationError(
            "Anchor and hedge instruments must use the same currency."
        )


def build_leg_snapshot(
    leg: RelativeValueLeg,
    german_curve: pd.DataFrame,
    settlement_date: date,
    position_notional_eur: float,
) -> SovereignSnapshotResult:
    """
    Build the valuation snapshot for one relative-value leg.
    """
    try:
        return build_instrument_snapshot(
            instrument=leg.instrument,
            german_curve=german_curve,
            settlement_date=settlement_date,
            position_notional_eur=position_notional_eur,
            explicit_yield_input=leg.yield_input,
        )
    except SovereignSnapshotValidationError as error:
        raise RelativeValueValidationError(
            str(
                error
            )
        ) from error


def calculate_dv01_neutral_hedge_notional(
    anchor_dv01_per_eur_1m: float,
    hedge_dv01_per_eur_1m: float,
    anchor_notional_eur: float,
) -> float:
    """
    Calculate hedge face value required to match anchor DV01.
    """
    validate_positive_notional(
        anchor_notional_eur
    )

    for value, name in (
        (
            anchor_dv01_per_eur_1m,
            "anchor_dv01_per_eur_1m",
        ),
        (
            hedge_dv01_per_eur_1m,
            "hedge_dv01_per_eur_1m",
        ),
    ):
        if not np.isfinite(
            value
        ):
            raise RelativeValueValidationError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise RelativeValueValidationError(
                f"{name} must be positive."
            )

    anchor_dv01 = (
        anchor_dv01_per_eur_1m
        * anchor_notional_eur
        / 1_000_000.0
    )

    return (
        anchor_dv01
        / hedge_dv01_per_eur_1m
        * 1_000_000.0
    )


def build_dv01_neutral_position(
    anchor_leg: RelativeValueLeg,
    hedge_leg: RelativeValueLeg,
    german_curve: pd.DataFrame,
    settlement_date: date,
    anchor_notional_eur: float = DEFAULT_ANCHOR_NOTIONAL_EUR,
) -> RelativeValuePosition:
    """
    Construct a two-leg sovereign trade with matched absolute DV01.

    The anchor notional is supplied by the user. RepoLens calculates
    the hedge notional required to offset the anchor leg's first-order
    interest-rate risk.
    """
    validate_positive_notional(
        anchor_notional_eur
    )

    validate_relative_value_legs(
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
    )

    anchor_snapshot = build_leg_snapshot(
        leg=anchor_leg,
        german_curve=german_curve,
        settlement_date=settlement_date,
        position_notional_eur=anchor_notional_eur,
    )

    hedge_unit_snapshot = build_leg_snapshot(
        leg=hedge_leg,
        german_curve=german_curve,
        settlement_date=settlement_date,
        position_notional_eur=1_000_000.0,
    )

    hedge_notional_eur = (
        calculate_dv01_neutral_hedge_notional(
            anchor_dv01_per_eur_1m=(
                anchor_snapshot.dv01_per_eur_1m
            ),
            hedge_dv01_per_eur_1m=(
                hedge_unit_snapshot.dv01_per_eur_1m
            ),
            anchor_notional_eur=anchor_notional_eur,
        )
    )

    hedge_snapshot = build_leg_snapshot(
        leg=hedge_leg,
        german_curve=german_curve,
        settlement_date=settlement_date,
        position_notional_eur=hedge_notional_eur,
    )

    signed_anchor_dv01_eur = (
        anchor_leg.direction.sign
        * anchor_snapshot.position_dv01_eur
    )

    signed_hedge_dv01_eur = (
        hedge_leg.direction.sign
        * hedge_snapshot.position_dv01_eur
    )

    net_dv01_eur = (
        signed_anchor_dv01_eur
        + signed_hedge_dv01_eur
    )

    gross_dv01_eur = (
        abs(
            signed_anchor_dv01_eur
        )
        + abs(
            signed_hedge_dv01_eur
        )
    )

    spread_bp = (
        anchor_snapshot.yield_percent
        - hedge_snapshot.yield_percent
    ) * 100.0

    return RelativeValuePosition(
        anchor_isin=anchor_leg.instrument.isin,
        hedge_isin=hedge_leg.instrument.isin,
        anchor_direction=anchor_leg.direction,
        hedge_direction=hedge_leg.direction,
        anchor_notional_eur=anchor_notional_eur,
        hedge_notional_eur=hedge_notional_eur,
        hedge_notional_per_anchor_euro=(
            hedge_notional_eur
            / anchor_notional_eur
        ),
        anchor_yield_percent=(
            anchor_snapshot.yield_percent
        ),
        hedge_yield_percent=(
            hedge_snapshot.yield_percent
        ),
        spread_bp=spread_bp,
        anchor_position_dv01_eur=(
            anchor_snapshot.position_dv01_eur
        ),
        hedge_position_dv01_eur=(
            hedge_snapshot.position_dv01_eur
        ),
        signed_anchor_dv01_eur=(
            signed_anchor_dv01_eur
        ),
        signed_hedge_dv01_eur=(
            signed_hedge_dv01_eur
        ),
        net_dv01_eur=net_dv01_eur,
        gross_dv01_eur=gross_dv01_eur,
        dv01_hedge_error_eur=abs(
            net_dv01_eur
        ),
        settlement_date=settlement_date,
    )


def scenario_leg_pnl(
    leg: RelativeValueLeg,
    settlement_date: date,
    starting_yield_percent: float,
    position_notional_eur: float,
    yield_shock_bp: float,
) -> float:
    """
    Calculate signed full-repricing P&L for one position leg.
    """
    scenarios = snapshot_scenarios(
        instrument=leg.instrument,
        settlement_date=settlement_date,
        yield_percent=starting_yield_percent,
        position_notional_eur=position_notional_eur,
        yield_shocks_bp=(
            yield_shock_bp,
        ),
    )

    unsigned_pnl = float(
        scenarios.iloc[0][
            "position_pnl_eur"
        ]
    )

    return (
        leg.direction.sign
        * unsigned_pnl
    )


def build_spread_scenarios(
    position: RelativeValuePosition,
    anchor_leg: RelativeValueLeg,
    hedge_leg: RelativeValueLeg,
    spread_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_SPREAD_SHOCKS_BP,
) -> pd.DataFrame:
    """
    Reprice a relative-value trade under spread shocks.

    A spread shock is split equally between the two legs:

        anchor yield shock = spread shock / 2
        hedge yield shock = -spread shock / 2

    This changes the anchor-minus-hedge spread by the requested amount
    while keeping the average yield approximately unchanged.
    """
    if not spread_shocks_bp:
        raise RelativeValueValidationError(
            "spread_shocks_bp must not be empty."
        )

    rows: list[
        dict[str, object]
    ] = []

    for spread_shock_bp in spread_shocks_bp:
        if not np.isfinite(
            spread_shock_bp
        ):
            raise RelativeValueValidationError(
                "Spread shocks must be finite."
            )

        anchor_yield_shock_bp = (
            spread_shock_bp
            / 2.0
        )

        hedge_yield_shock_bp = (
            -spread_shock_bp
            / 2.0
        )

        anchor_pnl_eur = scenario_leg_pnl(
            leg=anchor_leg,
            settlement_date=position.settlement_date,
            starting_yield_percent=(
                position.anchor_yield_percent
            ),
            position_notional_eur=(
                position.anchor_notional_eur
            ),
            yield_shock_bp=anchor_yield_shock_bp,
        )

        hedge_pnl_eur = scenario_leg_pnl(
            leg=hedge_leg,
            settlement_date=position.settlement_date,
            starting_yield_percent=(
                position.hedge_yield_percent
            ),
            position_notional_eur=(
                position.hedge_notional_eur
            ),
            yield_shock_bp=hedge_yield_shock_bp,
        )

        rows.append(
            {
                "spread_shock_bp": spread_shock_bp,
                "anchor_yield_shock_bp": (
                    anchor_yield_shock_bp
                ),
                "hedge_yield_shock_bp": (
                    hedge_yield_shock_bp
                ),
                "shocked_spread_bp": (
                    position.spread_bp
                    + spread_shock_bp
                ),
                "anchor_pnl_eur": anchor_pnl_eur,
                "hedge_pnl_eur": hedge_pnl_eur,
                "total_pnl_eur": (
                    anchor_pnl_eur
                    + hedge_pnl_eur
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_parallel_scenarios(
    position: RelativeValuePosition,
    anchor_leg: RelativeValueLeg,
    hedge_leg: RelativeValueLeg,
    parallel_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_PARALLEL_SHOCKS_BP,
) -> pd.DataFrame:
    """
    Reprice both legs under equal parallel yield shocks.
    """
    if not parallel_shocks_bp:
        raise RelativeValueValidationError(
            "parallel_shocks_bp must not be empty."
        )

    rows: list[
        dict[str, object]
    ] = []

    for parallel_shock_bp in parallel_shocks_bp:
        if not np.isfinite(
            parallel_shock_bp
        ):
            raise RelativeValueValidationError(
                "Parallel shocks must be finite."
            )

        anchor_pnl_eur = scenario_leg_pnl(
            leg=anchor_leg,
            settlement_date=position.settlement_date,
            starting_yield_percent=(
                position.anchor_yield_percent
            ),
            position_notional_eur=(
                position.anchor_notional_eur
            ),
            yield_shock_bp=parallel_shock_bp,
        )

        hedge_pnl_eur = scenario_leg_pnl(
            leg=hedge_leg,
            settlement_date=position.settlement_date,
            starting_yield_percent=(
                position.hedge_yield_percent
            ),
            position_notional_eur=(
                position.hedge_notional_eur
            ),
            yield_shock_bp=parallel_shock_bp,
        )

        rows.append(
            {
                "parallel_shock_bp": parallel_shock_bp,
                "anchor_pnl_eur": anchor_pnl_eur,
                "hedge_pnl_eur": hedge_pnl_eur,
                "total_pnl_eur": (
                    anchor_pnl_eur
                    + hedge_pnl_eur
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def position_to_frame(
    position: RelativeValuePosition,
) -> pd.DataFrame:
    """
    Convert a relative-value position into a two-row leg table.
    """
    return pd.DataFrame(
        [
            {
                "leg": "Anchor",
                "isin": position.anchor_isin,
                "direction": (
                    position.anchor_direction.value
                ),
                "notional_eur": (
                    position.anchor_notional_eur
                ),
                "yield_percent": (
                    position.anchor_yield_percent
                ),
                "position_dv01_eur": (
                    position.anchor_position_dv01_eur
                ),
                "signed_dv01_eur": (
                    position.signed_anchor_dv01_eur
                ),
            },
            {
                "leg": "Hedge",
                "isin": position.hedge_isin,
                "direction": (
                    position.hedge_direction.value
                ),
                "notional_eur": (
                    position.hedge_notional_eur
                ),
                "yield_percent": (
                    position.hedge_yield_percent
                ),
                "position_dv01_eur": (
                    position.hedge_position_dv01_eur
                ),
                "signed_dv01_eur": (
                    position.signed_hedge_dv01_eur
                ),
            },
        ]
    )