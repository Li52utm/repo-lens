from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, Mapping

import numpy as np
import pandas as pd

from src.sovereign_instruments import (
    SovereignCountry,
    instrument_for_country_tenor,
)
from src.sovereign_relative_value import (
    PositionDirection,
    RelativeValueLeg,
    RelativeValuePosition,
    RelativeValueValidationError,
    build_dv01_neutral_position,
    build_spread_scenarios,
)
from src.sovereign_snapshot import SovereignYieldInput


SUPPORTED_TENORS: Final[tuple[int, ...]] = (
    2,
    5,
    10,
    30,
)

DEFAULT_ANCHOR_NOTIONAL_EUR: Final[float] = 10_000_000.0

DEFAULT_ADVERSE_SPREAD_SHOCK_BP: Final[float] = 10.0


class SovereignOpportunityError(RuntimeError):
    """
    Base exception for RepoLens sovereign opportunity analytics.
    """


class SovereignOpportunityValidationError(
    SovereignOpportunityError
):
    """
    Raised when sovereign opportunity inputs fail validation.
    """


class OpportunityDirection(StrEnum):
    """
    Direction of the suggested Italy-versus-Germany trade.
    """

    LONG_ITALY_SHORT_GERMANY = (
        "LONG_ITALY_SHORT_GERMANY"
    )

    SHORT_ITALY_LONG_GERMANY = (
        "SHORT_ITALY_LONG_GERMANY"
    )

    NO_TRADE = "NO_TRADE"


class OpportunityConviction(StrEnum):
    """
    Simple current-versus-target opportunity classification.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class SovereignOpportunityInput:
    """
    Define current and target market inputs for one tenor.

    Yields and spreads are expressed in percentage points and basis
    points respectively.

    Example:
        italian_yield_percent = 3.85
        target_spread_bp = 90.0
    """

    tenor_years: int
    italian_yield_percent: float
    target_spread_bp: float
    source_name: str = "Desk input"

    def __post_init__(self) -> None:
        if self.tenor_years not in SUPPORTED_TENORS:
            raise SovereignOpportunityValidationError(
                "tenor_years must be one of "
                f"{SUPPORTED_TENORS}."
            )

        if not np.isfinite(
            self.italian_yield_percent
        ):
            raise SovereignOpportunityValidationError(
                "italian_yield_percent must be finite."
            )

        if (
            self.italian_yield_percent
            <= -100.0
        ):
            raise SovereignOpportunityValidationError(
                "italian_yield_percent must be greater "
                "than -100%."
            )

        if (
            self.italian_yield_percent
            > 100.0
        ):
            raise SovereignOpportunityValidationError(
                "italian_yield_percent is implausibly high."
            )

        if not np.isfinite(
            self.target_spread_bp
        ):
            raise SovereignOpportunityValidationError(
                "target_spread_bp must be finite."
            )

        if not self.source_name.strip():
            raise SovereignOpportunityValidationError(
                "source_name must not be empty."
            )


@dataclass(frozen=True)
class SovereignOpportunityResult:
    """
    Store one ranked Italy-versus-Germany opportunity.
    """

    tenor_years: int
    italy_isin: str
    germany_isin: str
    italy_display_name: str
    germany_display_name: str
    italian_yield_percent: float
    german_yield_percent: float
    current_spread_bp: float
    target_spread_bp: float
    target_spread_change_bp: float
    absolute_dislocation_bp: float
    trade_direction: OpportunityDirection
    conviction: OpportunityConviction
    anchor_notional_eur: float
    hedge_notional_eur: float
    hedge_ratio: float
    gross_dv01_eur: float
    net_dv01_eur: float
    target_pnl_eur: float
    adverse_spread_shock_bp: float
    adverse_pnl_eur: float
    reward_to_risk: float
    observation_date: date
    italian_source_name: str
    german_source_name: str


def validate_positive_value(
    value: float,
    field_name: str,
) -> None:
    """
    Validate a finite, strictly positive numerical value.
    """
    if not np.isfinite(
        value
    ):
        raise SovereignOpportunityValidationError(
            f"{field_name} must be finite."
        )

    if value <= 0.0:
        raise SovereignOpportunityValidationError(
            f"{field_name} must be positive."
        )


def latest_german_curve(
    german_curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and return the latest German observation for each tenor.
    """
    required_columns = {
        "observation_date",
        "country_code",
        "tenor_years",
        "yield_percent",
        "source_name",
        "data_status",
    }

    missing_columns = (
        required_columns
        - set(
            german_curve.columns
        )
    )

    if missing_columns:
        raise SovereignOpportunityValidationError(
            "German curve is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    prepared = german_curve.copy()

    prepared["observation_date"] = pd.to_datetime(
        prepared["observation_date"],
        errors="coerce",
    )

    prepared["tenor_years"] = pd.to_numeric(
        prepared["tenor_years"],
        errors="coerce",
    )

    prepared["yield_percent"] = pd.to_numeric(
        prepared["yield_percent"],
        errors="coerce",
    )

    prepared["country_code"] = (
        prepared["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    prepared = prepared.loc[
        prepared["country_code"].eq(
            "DE"
        )
    ]

    prepared = prepared.dropna(
        subset=[
            "observation_date",
            "tenor_years",
            "yield_percent",
            "source_name",
            "data_status",
        ]
    )

    if prepared.empty:
        raise SovereignOpportunityValidationError(
            "German curve contains no valid observations."
        )

    prepared["tenor_years"] = (
        prepared["tenor_years"]
        .astype(int)
    )

    latest = (
        prepared
        .sort_values(
            [
                "tenor_years",
                "observation_date",
            ]
        )
        .groupby(
            "tenor_years",
            as_index=False,
        )
        .tail(1)
        .sort_values(
            "tenor_years"
        )
        .reset_index(
            drop=True
        )
    )

    missing_tenors = (
        set(
            SUPPORTED_TENORS
        )
        - set(
            latest[
                "tenor_years"
            ]
        )
    )

    if missing_tenors:
        raise SovereignOpportunityValidationError(
            "German curve is missing supported tenors: "
            f"{sorted(missing_tenors)}."
        )

    return latest


def german_observation_for_tenor(
    prepared_curve: pd.DataFrame,
    tenor_years: int,
) -> pd.Series:
    """
    Return one latest German benchmark observation.
    """
    matches = prepared_curve.loc[
        prepared_curve[
            "tenor_years"
        ].eq(
            tenor_years
        )
    ]

    if len(
        matches
    ) != 1:
        raise SovereignOpportunityValidationError(
            "Expected exactly one German benchmark "
            f"observation for {tenor_years}Y."
        )

    return matches.iloc[0]


def determine_trade_direction(
    target_spread_change_bp: float,
    no_trade_threshold_bp: float,
) -> OpportunityDirection:
    """
    Determine the trade required to move toward target fair value.
    """
    validate_positive_value(
        no_trade_threshold_bp,
        "no_trade_threshold_bp",
    )

    if (
        abs(
            target_spread_change_bp
        )
        < no_trade_threshold_bp
    ):
        return OpportunityDirection.NO_TRADE

    if target_spread_change_bp < 0.0:
        return (
            OpportunityDirection
            .LONG_ITALY_SHORT_GERMANY
        )

    return (
        OpportunityDirection
        .SHORT_ITALY_LONG_GERMANY
    )


def classify_conviction(
    absolute_dislocation_bp: float,
    no_trade_threshold_bp: float,
) -> OpportunityConviction:
    """
    Classify current spread dislocation magnitude.

    This is a transparent current-versus-target classification, not a
    historical rich/cheap signal.
    """
    validate_positive_value(
        no_trade_threshold_bp,
        "no_trade_threshold_bp",
    )

    if (
        absolute_dislocation_bp
        < no_trade_threshold_bp
    ):
        return OpportunityConviction.NEUTRAL

    if absolute_dislocation_bp >= 25.0:
        return OpportunityConviction.HIGH

    if absolute_dislocation_bp >= 10.0:
        return OpportunityConviction.MEDIUM

    return OpportunityConviction.LOW


def scenario_pnl_for_spread_change(
    position: RelativeValuePosition,
    anchor_leg: RelativeValueLeg,
    hedge_leg: RelativeValueLeg,
    spread_change_bp: float,
) -> float:
    """
    Calculate full-repricing trade P&L for one spread change.
    """
    scenarios = build_spread_scenarios(
        position=position,
        anchor_leg=anchor_leg,
        hedge_leg=hedge_leg,
        spread_shocks_bp=(
            spread_change_bp,
        ),
    )

    return float(
        scenarios.iloc[0][
            "total_pnl_eur"
        ]
    )


def build_opportunity(
    opportunity_input: SovereignOpportunityInput,
    german_curve: pd.DataFrame,
    settlement_date: date,
    anchor_notional_eur: float = (
        DEFAULT_ANCHOR_NOTIONAL_EUR
    ),
    adverse_spread_shock_bp: float = (
        DEFAULT_ADVERSE_SPREAD_SHOCK_BP
    ),
    no_trade_threshold_bp: float = 2.0,
) -> SovereignOpportunityResult:
    """
    Build one DV01-neutral Italy-versus-Germany opportunity.
    """
    validate_positive_value(
        anchor_notional_eur,
        "anchor_notional_eur",
    )

    validate_positive_value(
        adverse_spread_shock_bp,
        "adverse_spread_shock_bp",
    )

    prepared_curve = latest_german_curve(
        german_curve
    )

    german_observation = (
        german_observation_for_tenor(
            prepared_curve=prepared_curve,
            tenor_years=(
                opportunity_input.tenor_years
            ),
        )
    )

    german_yield_percent = float(
        german_observation[
            "yield_percent"
        ]
    )

    current_spread_bp = (
        opportunity_input.italian_yield_percent
        - german_yield_percent
    ) * 100.0

    target_spread_change_bp = (
        opportunity_input.target_spread_bp
        - current_spread_bp
    )

    absolute_dislocation_bp = abs(
        target_spread_change_bp
    )

    trade_direction = determine_trade_direction(
        target_spread_change_bp=(
            target_spread_change_bp
        ),
        no_trade_threshold_bp=(
            no_trade_threshold_bp
        ),
    )

    conviction = classify_conviction(
        absolute_dislocation_bp=(
            absolute_dislocation_bp
        ),
        no_trade_threshold_bp=(
            no_trade_threshold_bp
        ),
    )

    italy_instrument = instrument_for_country_tenor(
        country=SovereignCountry.ITALY,
        benchmark_tenor_years=(
            opportunity_input.tenor_years
        ),
    )

    germany_instrument = (
        instrument_for_country_tenor(
            country=SovereignCountry.GERMANY,
            benchmark_tenor_years=(
                opportunity_input.tenor_years
            ),
        )
    )

    italian_yield_input = SovereignYieldInput(
        isin=italy_instrument.isin,
        yield_percent=(
            opportunity_input
            .italian_yield_percent
        ),
        observation_date=settlement_date,
        source_name=(
            opportunity_input.source_name
        ),
    )

    if (
        trade_direction
        == OpportunityDirection
        .SHORT_ITALY_LONG_GERMANY
    ):
        anchor_direction = PositionDirection.SHORT
        hedge_direction = PositionDirection.LONG
    else:
        anchor_direction = PositionDirection.LONG
        hedge_direction = PositionDirection.SHORT

    anchor_leg = RelativeValueLeg(
        instrument=italy_instrument,
        direction=anchor_direction,
        yield_input=italian_yield_input,
    )

    hedge_leg = RelativeValueLeg(
        instrument=germany_instrument,
        direction=hedge_direction,
    )

    try:
        position = build_dv01_neutral_position(
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            german_curve=prepared_curve,
            settlement_date=settlement_date,
            anchor_notional_eur=(
                anchor_notional_eur
            ),
        )
    except RelativeValueValidationError as error:
        raise SovereignOpportunityValidationError(
            str(
                error
            )
        ) from error

    if (
        trade_direction
        == OpportunityDirection.NO_TRADE
    ):
        target_pnl_eur = 0.0
        adverse_pnl_eur = 0.0
        reward_to_risk = 0.0
    else:
        target_pnl_eur = (
            scenario_pnl_for_spread_change(
                position=position,
                anchor_leg=anchor_leg,
                hedge_leg=hedge_leg,
                spread_change_bp=(
                    target_spread_change_bp
                ),
            )
        )

        adverse_shock = (
            adverse_spread_shock_bp
            if trade_direction
            == OpportunityDirection
            .LONG_ITALY_SHORT_GERMANY
            else -adverse_spread_shock_bp
        )

        adverse_pnl_eur = (
            scenario_pnl_for_spread_change(
                position=position,
                anchor_leg=anchor_leg,
                hedge_leg=hedge_leg,
                spread_change_bp=adverse_shock,
            )
        )

        adverse_loss = abs(
            min(
                adverse_pnl_eur,
                0.0,
            )
        )

        if adverse_loss > 0.0:
            reward_to_risk = (
                max(
                    target_pnl_eur,
                    0.0,
                )
                / adverse_loss
            )
        else:
            reward_to_risk = float(
                "inf"
            )

    return SovereignOpportunityResult(
        tenor_years=(
            opportunity_input.tenor_years
        ),
        italy_isin=italy_instrument.isin,
        germany_isin=germany_instrument.isin,
        italy_display_name=(
            italy_instrument.display_name
        ),
        germany_display_name=(
            germany_instrument.display_name
        ),
        italian_yield_percent=(
            opportunity_input
            .italian_yield_percent
        ),
        german_yield_percent=(
            german_yield_percent
        ),
        current_spread_bp=current_spread_bp,
        target_spread_bp=(
            opportunity_input.target_spread_bp
        ),
        target_spread_change_bp=(
            target_spread_change_bp
        ),
        absolute_dislocation_bp=(
            absolute_dislocation_bp
        ),
        trade_direction=trade_direction,
        conviction=conviction,
        anchor_notional_eur=(
            position.anchor_notional_eur
        ),
        hedge_notional_eur=(
            position.hedge_notional_eur
        ),
        hedge_ratio=(
            position
            .hedge_notional_per_anchor_euro
        ),
        gross_dv01_eur=(
            position.gross_dv01_eur
        ),
        net_dv01_eur=(
            position.net_dv01_eur
        ),
        target_pnl_eur=target_pnl_eur,
        adverse_spread_shock_bp=(
            adverse_spread_shock_bp
        ),
        adverse_pnl_eur=adverse_pnl_eur,
        reward_to_risk=reward_to_risk,
        observation_date=settlement_date,
        italian_source_name=(
            opportunity_input.source_name
        ),
        german_source_name=str(
            german_observation[
                "source_name"
            ]
        ),
    )


def validate_opportunity_inputs(
    opportunity_inputs: tuple[
        SovereignOpportunityInput,
        ...,
    ],
) -> None:
    """
    Validate complete tenor coverage and uniqueness.
    """
    if not opportunity_inputs:
        raise SovereignOpportunityValidationError(
            "opportunity_inputs must not be empty."
        )

    tenors = [
        opportunity_input.tenor_years
        for opportunity_input
        in opportunity_inputs
    ]

    if len(
        tenors
    ) != len(
        set(
            tenors
        )
    ):
        raise SovereignOpportunityValidationError(
            "Opportunity tenors must be unique."
        )

    if set(
        tenors
    ) != set(
        SUPPORTED_TENORS
    ):
        raise SovereignOpportunityValidationError(
            "Opportunity inputs must cover exactly "
            f"{SUPPORTED_TENORS}."
        )


def build_opportunity_set(
    opportunity_inputs: tuple[
        SovereignOpportunityInput,
        ...,
    ],
    german_curve: pd.DataFrame,
    settlement_date: date,
    anchor_notional_eur: float = (
        DEFAULT_ANCHOR_NOTIONAL_EUR
    ),
    adverse_spread_shock_bp: float = (
        DEFAULT_ADVERSE_SPREAD_SHOCK_BP
    ),
    no_trade_threshold_bp: float = 2.0,
) -> tuple[
    SovereignOpportunityResult,
    ...,
]:
    """
    Build and rank the complete 2Y, 5Y, 10Y and 30Y opportunity set.
    """
    validate_opportunity_inputs(
        opportunity_inputs
    )

    results = tuple(
        build_opportunity(
            opportunity_input=opportunity_input,
            german_curve=german_curve,
            settlement_date=settlement_date,
            anchor_notional_eur=(
                anchor_notional_eur
            ),
            adverse_spread_shock_bp=(
                adverse_spread_shock_bp
            ),
            no_trade_threshold_bp=(
                no_trade_threshold_bp
            ),
        )
        for opportunity_input
        in opportunity_inputs
    )

    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.trade_direction
                == OpportunityDirection.NO_TRADE,
                -result.reward_to_risk,
                -result.absolute_dislocation_bp,
            ),
        )
    )


def opportunities_to_frame(
    opportunities: tuple[
        SovereignOpportunityResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Convert opportunity results into a dashboard table.
    """
    if not opportunities:
        raise SovereignOpportunityValidationError(
            "opportunities must not be empty."
        )

    return pd.DataFrame(
        [
            {
                "rank": rank,
                "tenor_years": opportunity.tenor_years,
                "italy_isin": opportunity.italy_isin,
                "germany_isin": opportunity.germany_isin,
                "italian_yield_percent": (
                    opportunity
                    .italian_yield_percent
                ),
                "german_yield_percent": (
                    opportunity
                    .german_yield_percent
                ),
                "current_spread_bp": (
                    opportunity.current_spread_bp
                ),
                "target_spread_bp": (
                    opportunity.target_spread_bp
                ),
                "target_spread_change_bp": (
                    opportunity
                    .target_spread_change_bp
                ),
                "absolute_dislocation_bp": (
                    opportunity
                    .absolute_dislocation_bp
                ),
                "trade_direction": (
                    opportunity
                    .trade_direction
                    .value
                ),
                "conviction": (
                    opportunity
                    .conviction
                    .value
                ),
                "anchor_notional_eur": (
                    opportunity
                    .anchor_notional_eur
                ),
                "hedge_notional_eur": (
                    opportunity
                    .hedge_notional_eur
                ),
                "hedge_ratio": (
                    opportunity.hedge_ratio
                ),
                "gross_dv01_eur": (
                    opportunity.gross_dv01_eur
                ),
                "net_dv01_eur": (
                    opportunity.net_dv01_eur
                ),
                "target_pnl_eur": (
                    opportunity.target_pnl_eur
                ),
                "adverse_pnl_eur": (
                    opportunity.adverse_pnl_eur
                ),
                "reward_to_risk": (
                    opportunity.reward_to_risk
                ),
                "observation_date": (
                    opportunity.observation_date
                ),
                "italian_source_name": (
                    opportunity
                    .italian_source_name
                ),
                "german_source_name": (
                    opportunity
                    .german_source_name
                ),
            }
            for rank, opportunity
            in enumerate(
                opportunities,
                start=1,
            )
        ]
    )


def curve_slope_frame(
    opportunity_inputs: tuple[
        SovereignOpportunityInput,
        ...,
    ],
    german_curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate current Germany and Italy curve slopes.
    """
    validate_opportunity_inputs(
        opportunity_inputs
    )

    prepared_curve = latest_german_curve(
        german_curve
    )

    italian_yields: Mapping[
        int,
        float,
    ] = {
        opportunity_input.tenor_years: (
            opportunity_input
            .italian_yield_percent
        )
        for opportunity_input
        in opportunity_inputs
    }

    german_yields: Mapping[
        int,
        float,
    ] = {
        tenor: float(
            german_observation_for_tenor(
                prepared_curve=prepared_curve,
                tenor_years=tenor,
            )[
                "yield_percent"
            ]
        )
        for tenor in SUPPORTED_TENORS
    }

    slope_pairs = (
        (
            "2s5s",
            2,
            5,
        ),
        (
            "2s10s",
            2,
            10,
        ),
        (
            "5s10s",
            5,
            10,
        ),
        (
            "10s30s",
            10,
            30,
        ),
    )

    return pd.DataFrame(
        [
            {
                "curve_segment": label,
                "short_tenor_years": short_tenor,
                "long_tenor_years": long_tenor,
                "germany_slope_bp": (
                    german_yields[
                        long_tenor
                    ]
                    - german_yields[
                        short_tenor
                    ]
                ) * 100.0,
                "italy_slope_bp": (
                    italian_yields[
                        long_tenor
                    ]
                    - italian_yields[
                        short_tenor
                    ]
                ) * 100.0,
                "italy_minus_germany_slope_bp": (
                    (
                        italian_yields[
                            long_tenor
                        ]
                        - italian_yields[
                            short_tenor
                        ]
                    )
                    - (
                        german_yields[
                            long_tenor
                        ]
                        - german_yields[
                            short_tenor
                        ]
                    )
                ) * 100.0,
            }
            for (
                label,
                short_tenor,
                long_tenor,
            )
            in slope_pairs
        ]
    )