from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from src.bond_analytics import (
    BondRiskMetrics,
    calculate_bond_risk_metrics,
    position_dv01,
    run_parallel_yield_scenarios,
)
from src.sovereign_instruments import (
    SOVEREIGN_INSTRUMENTS,
    SovereignCountry,
    SovereignInstrument,
)


DEFAULT_POSITION_NOTIONAL_EUR: Final[float] = 10_000_000.0

DEFAULT_SCENARIO_SHOCKS_BP: Final[
    tuple[
        float,
        ...,
    ]
] = (
    -25.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    25.0,
)


class SovereignSnapshotError(RuntimeError):
    """
    Base exception for RepoLens sovereign snapshots.
    """


class SovereignSnapshotValidationError(
    SovereignSnapshotError
):
    """
    Raised when snapshot inputs fail validation.
    """


class SnapshotDataStatus(StrEnum):
    """
    Describe the origin of the yield used in a bond snapshot.
    """

    OFFICIAL_DAILY = "OFFICIAL_DAILY"
    DESK_INPUT = "DESK_INPUT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SovereignYieldInput:
    """
    Define one explicit instrument-level yield observation.

    yield_percent is expressed in percentage points.

    Example:
        3.85 means 3.85%.
    """

    isin: str
    yield_percent: float
    observation_date: date
    source_name: str = "Desk input"
    data_status: SnapshotDataStatus = (
        SnapshotDataStatus.DESK_INPUT
    )

    def __post_init__(self) -> None:
        if len(
            self.isin.strip()
        ) != 12:
            raise SovereignSnapshotValidationError(
                "isin must contain exactly 12 characters."
            )

        if not np.isfinite(
            self.yield_percent
        ):
            raise SovereignSnapshotValidationError(
                "yield_percent must be finite."
            )

        if self.yield_percent <= -100.0:
            raise SovereignSnapshotValidationError(
                "yield_percent must be greater than -100%."
            )

        if self.yield_percent > 100.0:
            raise SovereignSnapshotValidationError(
                "yield_percent is implausibly high."
            )

        if not self.source_name.strip():
            raise SovereignSnapshotValidationError(
                "source_name must not be empty."
            )


@dataclass(frozen=True)
class SovereignSnapshotResult:
    """
    Store one complete instrument-level valuation snapshot.

    german_benchmark_yield_percent and spread_to_germany_bp may contain
    NaN when no exact permitted German benchmark observation exists for
    the instrument's maturity sector.
    """

    isin: str
    display_name: str
    country: str
    security_type: str
    benchmark_tenor_years: int
    observation_date: date | None
    settlement_date: date
    source_name: str
    data_status: SnapshotDataStatus
    market_data_available: bool
    yield_percent: float
    german_benchmark_yield_percent: float
    spread_to_germany_bp: float
    clean_price: float
    dirty_price: float
    accrued_interest: float
    modified_duration: float
    macaulay_duration: float
    convexity: float
    dv01_per_100: float
    dv01_per_eur_1m: float
    position_notional_eur: float
    position_dv01_eur: float


def validate_position_notional(
    position_notional_eur: float,
) -> None:
    """
    Validate the position face-value amount.
    """
    if not np.isfinite(
        position_notional_eur
    ):
        raise SovereignSnapshotValidationError(
            "position_notional_eur must be finite."
        )

    if position_notional_eur < 0.0:
        raise SovereignSnapshotValidationError(
            "position_notional_eur must not be negative."
        )


def prepare_german_benchmark_curve(
    benchmark_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the latest official German yield for every available tenor.

    RepoLens does not interpolate missing tenors in this function.
    The returned frame therefore contains only observations supplied by
    the approved German benchmark dataset.
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
            benchmark_data.columns
        )
    )

    if missing_columns:
        raise SovereignSnapshotValidationError(
            "German benchmark data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    prepared = benchmark_data.copy()

    prepared[
        "observation_date"
    ] = pd.to_datetime(
        prepared[
            "observation_date"
        ],
        errors="coerce",
    )

    prepared[
        "tenor_years"
    ] = pd.to_numeric(
        prepared[
            "tenor_years"
        ],
        errors="coerce",
    )

    prepared[
        "yield_percent"
    ] = pd.to_numeric(
        prepared[
            "yield_percent"
        ],
        errors="coerce",
    )

    prepared[
        "country_code"
    ] = (
        prepared[
            "country_code"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    prepared = prepared.loc[
        prepared[
            "country_code"
        ].eq(
            "DE"
        )
    ]

    prepared = prepared.dropna(
        subset=[
            "observation_date",
            "country_code",
            "tenor_years",
            "yield_percent",
            "source_name",
            "data_status",
        ]
    )

    if prepared.empty:
        raise SovereignSnapshotValidationError(
            "German benchmark data contains no valid observations."
        )

    prepared[
        "tenor_years"
    ] = (
        prepared[
            "tenor_years"
        ]
        .astype(
            int
        )
    )

    invalid_tenors = (
        prepared[
            "tenor_years"
        ]
        .le(
            0
        )
    )

    if invalid_tenors.any():
        raise SovereignSnapshotValidationError(
            "German benchmark tenors must be positive."
        )

    invalid_yields = (
        prepared[
            "yield_percent"
        ].le(
            -100.0
        )
        | prepared[
            "yield_percent"
        ].gt(
            100.0
        )
    )

    if invalid_yields.any():
        raise SovereignSnapshotValidationError(
            "German benchmark data contains implausible yields."
        )

    latest_curve = (
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
        .tail(
            1
        )
        .sort_values(
            "tenor_years"
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_tenors = (
        latest_curve[
            "tenor_years"
        ]
        .duplicated()
        .any()
    )

    if duplicate_tenors:
        raise SovereignSnapshotValidationError(
            "German benchmark curve contains duplicate latest tenors."
        )

    return latest_curve[
        [
            "observation_date",
            "country_code",
            "tenor_years",
            "yield_percent",
            "source_name",
            "data_status",
        ]
    ]


def optional_german_benchmark_for_tenor(
    prepared_curve: pd.DataFrame,
    tenor_years: int,
) -> pd.Series | None:
    """
    Return an exact German benchmark observation when available.

    Missing maturity sectors return None. RepoLens does not interpolate
    or manufacture a German benchmark yield.
    """
    if tenor_years <= 0:
        raise SovereignSnapshotValidationError(
            "tenor_years must be positive."
        )

    matches = prepared_curve.loc[
        prepared_curve[
            "tenor_years"
        ].eq(
            tenor_years
        )
    ]

    if matches.empty:
        return None

    if len(
        matches
    ) != 1:
        raise SovereignSnapshotValidationError(
            "More than one German benchmark observation exists for "
            f"{tenor_years}Y."
        )

    return matches.iloc[
        0
    ]


def german_benchmark_for_tenor(
    prepared_curve: pd.DataFrame,
    tenor_years: int,
) -> pd.Series:
    """
    Return the latest exact German benchmark observation for one tenor.

    This strict helper is retained for callers that explicitly require
    an exact benchmark.
    """
    benchmark = optional_german_benchmark_for_tenor(
        prepared_curve=prepared_curve,
        tenor_years=tenor_years,
    )

    if benchmark is None:
        raise SovereignSnapshotValidationError(
            "No German benchmark yield is available for "
            f"{tenor_years}Y."
        )

    return benchmark


def validate_valuation_date(
    instrument: SovereignInstrument,
    settlement_date: date,
) -> None:
    """
    Ensure valuation occurs during the instrument's life.
    """
    if settlement_date < instrument.issue_date:
        raise SovereignSnapshotValidationError(
            "settlement_date must not be before the instrument issue date."
        )

    if settlement_date >= instrument.maturity_date:
        raise SovereignSnapshotValidationError(
            "settlement_date must be before the instrument maturity date."
        )


def metrics_to_snapshot(
    instrument: SovereignInstrument,
    metrics: BondRiskMetrics,
    observation_date: date,
    source_name: str,
    data_status: SnapshotDataStatus,
    german_benchmark_yield_percent: float,
    position_notional_eur: float,
) -> SovereignSnapshotResult:
    """
    Convert bond analytics into the standard snapshot contract.

    A missing exact German benchmark is represented by NaN rather than
    an interpolated or proxy observation.
    """
    yield_percent = (
        metrics.yield_to_maturity
        * 100.0
    )

    german_benchmark_available = bool(
        np.isfinite(
            german_benchmark_yield_percent
        )
    )

    if not german_benchmark_available:
        spread_to_germany_bp = float(
            "nan"
        )
    elif (
        instrument.country
        == SovereignCountry.GERMANY
    ):
        spread_to_germany_bp = 0.0
    else:
        spread_to_germany_bp = (
            yield_percent
            - german_benchmark_yield_percent
        ) * 100.0

    position_dv01_eur = position_dv01(
        dv01_per_100_value=(
            metrics.dv01_per_100
        ),
        position_notional=(
            position_notional_eur
        ),
    )

    dv01_per_eur_1m = position_dv01(
        dv01_per_100_value=(
            metrics.dv01_per_100
        ),
        position_notional=(
            1_000_000.0
        ),
    )

    return SovereignSnapshotResult(
        isin=instrument.isin,
        display_name=(
            instrument.display_name
        ),
        country=(
            instrument.country.value
        ),
        security_type=(
            instrument.security_type.value
        ),
        benchmark_tenor_years=(
            instrument.benchmark_tenor_years
        ),
        observation_date=observation_date,
        settlement_date=(
            metrics.settlement_date
        ),
        source_name=source_name,
        data_status=data_status,
        market_data_available=True,
        yield_percent=yield_percent,
        german_benchmark_yield_percent=(
            german_benchmark_yield_percent
        ),
        spread_to_germany_bp=(
            spread_to_germany_bp
        ),
        clean_price=(
            metrics.clean_price
        ),
        dirty_price=(
            metrics.dirty_price
        ),
        accrued_interest=(
            metrics.accrued_interest
        ),
        modified_duration=(
            metrics.modified_duration
        ),
        macaulay_duration=(
            metrics.macaulay_duration
        ),
        convexity=(
            metrics.convexity
        ),
        dv01_per_100=(
            metrics.dv01_per_100
        ),
        dv01_per_eur_1m=(
            dv01_per_eur_1m
        ),
        position_notional_eur=(
            position_notional_eur
        ),
        position_dv01_eur=(
            position_dv01_eur
        ),
    )


def build_instrument_snapshot(
    instrument: SovereignInstrument,
    german_curve: pd.DataFrame,
    settlement_date: date,
    position_notional_eur: float = (
        DEFAULT_POSITION_NOTIONAL_EUR
    ),
    explicit_yield_input: (
        SovereignYieldInput
        | None
    ) = None,
) -> SovereignSnapshotResult:
    """
    Build one complete sovereign bond snapshot.

    German instruments use an exact matching official benchmark yield
    when one exists and no explicit input is supplied.

    Any instrument may instead use an explicit desk-input yield.

    Non-German instruments always require an explicit instrument-level
    yield.

    When no exact German benchmark exists for the instrument's assigned
    maturity sector, RepoLens still values an instrument from an
    explicit yield but reports the German benchmark and sovereign spread
    as unavailable rather than interpolating them.
    """
    validate_position_notional(
        position_notional_eur
    )

    validate_valuation_date(
        instrument=instrument,
        settlement_date=settlement_date,
    )

    prepared_curve = prepare_german_benchmark_curve(
        german_curve
    )

    german_benchmark = (
        optional_german_benchmark_for_tenor(
            prepared_curve=prepared_curve,
            tenor_years=(
                instrument
                .benchmark_tenor_years
            ),
        )
    )

    if german_benchmark is None:
        german_yield_percent = float(
            "nan"
        )
    else:
        german_yield_percent = float(
            german_benchmark[
                "yield_percent"
            ]
        )

    if explicit_yield_input is not None:
        if (
            explicit_yield_input
            .isin
            .strip()
            .upper()
            != instrument.isin
        ):
            raise SovereignSnapshotValidationError(
                "Explicit yield input ISIN does not match the instrument."
            )

        yield_percent = (
            explicit_yield_input
            .yield_percent
        )

        observation_date = (
            explicit_yield_input
            .observation_date
        )

        source_name = (
            explicit_yield_input
            .source_name
        )

        data_status = (
            explicit_yield_input
            .data_status
        )

    elif (
        instrument.country
        == SovereignCountry.GERMANY
        and german_benchmark is not None
    ):
        yield_percent = (
            german_yield_percent
        )

        observation_date = pd.Timestamp(
            german_benchmark[
                "observation_date"
            ]
        ).date()

        source_name = str(
            german_benchmark[
                "source_name"
            ]
        )

        data_status = (
            SnapshotDataStatus
            .OFFICIAL_DAILY
        )

    elif (
        instrument.country
        == SovereignCountry.GERMANY
    ):
        raise SovereignSnapshotValidationError(
            "An explicit instrument-level yield is required for "
            f"{instrument.display_name} because no exact permitted "
            f"German {instrument.benchmark_tenor_years}Y benchmark "
            "observation is available. RepoLens will not interpolate "
            "or manufacture one."
        )

    else:
        raise SovereignSnapshotValidationError(
            "An explicit instrument-level yield is required for "
            f"{instrument.display_name}. RepoLens will not infer "
            "an Italian yield from the German curve."
        )

    if observation_date > settlement_date:
        raise SovereignSnapshotValidationError(
            "Yield observation_date must not be after settlement_date."
        )

    metrics = calculate_bond_risk_metrics(
        bond=(
            instrument
            .to_fixed_rate_bond()
        ),
        settlement_date=(
            settlement_date
        ),
        yield_to_maturity=(
            yield_percent
            / 100.0
        ),
    )

    return metrics_to_snapshot(
        instrument=instrument,
        metrics=metrics,
        observation_date=(
            observation_date
        ),
        source_name=source_name,
        data_status=data_status,
        german_benchmark_yield_percent=(
            german_yield_percent
        ),
        position_notional_eur=(
            position_notional_eur
        ),
    )


def unavailable_snapshot(
    instrument: SovereignInstrument,
    settlement_date: date,
    german_benchmark_yield_percent: float,
    position_notional_eur: float,
) -> SovereignSnapshotResult:
    """
    Create a transparent unavailable row without invented analytics.
    """
    missing = float(
        "nan"
    )

    return SovereignSnapshotResult(
        isin=instrument.isin,
        display_name=(
            instrument.display_name
        ),
        country=(
            instrument.country.value
        ),
        security_type=(
            instrument.security_type.value
        ),
        benchmark_tenor_years=(
            instrument.benchmark_tenor_years
        ),
        observation_date=None,
        settlement_date=settlement_date,
        source_name=(
            "No permitted instrument-level observation"
        ),
        data_status=(
            SnapshotDataStatus.UNAVAILABLE
        ),
        market_data_available=False,
        yield_percent=missing,
        german_benchmark_yield_percent=(
            german_benchmark_yield_percent
        ),
        spread_to_germany_bp=missing,
        clean_price=missing,
        dirty_price=missing,
        accrued_interest=missing,
        modified_duration=missing,
        macaulay_duration=missing,
        convexity=missing,
        dv01_per_100=missing,
        dv01_per_eur_1m=missing,
        position_notional_eur=(
            position_notional_eur
        ),
        position_dv01_eur=missing,
    )


def validate_explicit_yield_inputs(
    explicit_yield_inputs: tuple[
        SovereignYieldInput,
        ...,
    ],
    instruments: tuple[
        SovereignInstrument,
        ...,
    ],
) -> dict[
    str,
    SovereignYieldInput,
]:
    """
    Validate explicit yields against the actual instrument collection.

    This deliberately avoids looking instruments up in the legacy
    eight-bond registry so expanded catalogue instruments can be used
    safely.
    """
    instrument_isins = {
        instrument.isin
        for instrument in instruments
    }

    input_by_isin: dict[
        str,
        SovereignYieldInput,
    ] = {}

    for yield_input in explicit_yield_inputs:
        normalised_isin = (
            yield_input
            .isin
            .strip()
            .upper()
        )

        if (
            normalised_isin
            in input_by_isin
        ):
            raise SovereignSnapshotValidationError(
                "Duplicate explicit yield input for "
                f"{normalised_isin}."
            )

        if (
            normalised_isin
            not in instrument_isins
        ):
            raise SovereignSnapshotValidationError(
                "Explicit yield input references an instrument "
                "that is not present in the supplied sovereign "
                f"instrument collection: {normalised_isin}."
            )

        input_by_isin[
            normalised_isin
        ] = yield_input

    return input_by_isin


def build_registry_snapshot(
    german_curve: pd.DataFrame,
    settlement_date: date,
    position_notional_eur: float = (
        DEFAULT_POSITION_NOTIONAL_EUR
    ),
    explicit_yield_inputs: tuple[
        SovereignYieldInput,
        ...,
    ] = (),
    instruments: tuple[
        SovereignInstrument,
        ...,
    ] = SOVEREIGN_INSTRUMENTS,
) -> pd.DataFrame:
    """
    Build a snapshot for a sovereign instrument collection.

    Missing instrument-level market data is represented as UNAVAILABLE.

    Missing German maturity-sector observations are also represented
    transparently. RepoLens does not use an interpolated or synthetic
    German yield as a substitute for an unavailable official tenor.
    """
    validate_position_notional(
        position_notional_eur
    )

    if not instruments:
        raise SovereignSnapshotValidationError(
            "instruments must not be empty."
        )

    input_by_isin = (
        validate_explicit_yield_inputs(
            explicit_yield_inputs=(
                explicit_yield_inputs
            ),
            instruments=instruments,
        )
    )

    prepared_curve = prepare_german_benchmark_curve(
        german_curve
    )

    snapshots: list[
        SovereignSnapshotResult
    ] = []

    for instrument in instruments:
        benchmark = (
            optional_german_benchmark_for_tenor(
                prepared_curve=prepared_curve,
                tenor_years=(
                    instrument
                    .benchmark_tenor_years
                ),
            )
        )

        if benchmark is None:
            german_yield_percent = float(
                "nan"
            )
        else:
            german_yield_percent = float(
                benchmark[
                    "yield_percent"
                ]
            )

        explicit_input = (
            input_by_isin.get(
                instrument.isin
            )
        )

        has_usable_market_input = (
            explicit_input is not None
            or (
                instrument.country
                == SovereignCountry.GERMANY
                and benchmark is not None
            )
        )

        if not has_usable_market_input:
            snapshots.append(
                unavailable_snapshot(
                    instrument=instrument,
                    settlement_date=(
                        settlement_date
                    ),
                    german_benchmark_yield_percent=(
                        german_yield_percent
                    ),
                    position_notional_eur=(
                        position_notional_eur
                    ),
                )
            )

            continue

        snapshots.append(
            build_instrument_snapshot(
                instrument=instrument,
                german_curve=prepared_curve,
                settlement_date=(
                    settlement_date
                ),
                position_notional_eur=(
                    position_notional_eur
                ),
                explicit_yield_input=(
                    explicit_input
                ),
            )
        )

    return snapshots_to_frame(
        tuple(
            snapshots
        )
    )


def snapshot_scenarios(
    instrument: SovereignInstrument,
    settlement_date: date,
    yield_percent: float,
    position_notional_eur: float,
    yield_shocks_bp: tuple[
        float,
        ...,
    ] = DEFAULT_SCENARIO_SHOCKS_BP,
) -> pd.DataFrame:
    """
    Calculate position P&L under parallel yield shocks.
    """
    validate_position_notional(
        position_notional_eur
    )

    validate_valuation_date(
        instrument=instrument,
        settlement_date=settlement_date,
    )

    scenario_results = (
        run_parallel_yield_scenarios(
            bond=(
                instrument
                .to_fixed_rate_bond()
            ),
            settlement_date=(
                settlement_date
            ),
            yield_to_maturity=(
                yield_percent
                / 100.0
            ),
            position_notional=(
                position_notional_eur
            ),
            yield_shocks_bp=(
                yield_shocks_bp
            ),
        )
    )

    return pd.DataFrame(
        [
            {
                "isin": (
                    instrument.isin
                ),
                "yield_shock_bp": (
                    result.yield_shock_bp
                ),
                "shocked_yield_percent": (
                    result.shocked_yield
                    * 100.0
                ),
                "shocked_clean_price": (
                    result
                    .shocked_clean_price
                ),
                "clean_price_change": (
                    result
                    .clean_price_change
                ),
                "position_pnl_eur": (
                    result.position_pnl
                ),
            }
            for result
            in scenario_results
        ]
    )


def snapshots_to_frame(
    snapshots: tuple[
        SovereignSnapshotResult,
        ...,
    ],
) -> pd.DataFrame:
    """
    Convert sovereign snapshot results into a table.
    """
    if not snapshots:
        raise SovereignSnapshotValidationError(
            "snapshots must not be empty."
        )

    rows = [
        {
            "isin": snapshot.isin,
            "display_name": (
                snapshot.display_name
            ),
            "country": (
                snapshot.country
            ),
            "security_type": (
                snapshot.security_type
            ),
            "benchmark_tenor_years": (
                snapshot
                .benchmark_tenor_years
            ),
            "observation_date": (
                snapshot.observation_date
            ),
            "settlement_date": (
                snapshot.settlement_date
            ),
            "source_name": (
                snapshot.source_name
            ),
            "data_status": (
                snapshot
                .data_status
                .value
            ),
            "market_data_available": (
                snapshot
                .market_data_available
            ),
            "yield_percent": (
                snapshot.yield_percent
            ),
            "german_benchmark_yield_percent": (
                snapshot
                .german_benchmark_yield_percent
            ),
            "spread_to_germany_bp": (
                snapshot
                .spread_to_germany_bp
            ),
            "clean_price": (
                snapshot.clean_price
            ),
            "dirty_price": (
                snapshot.dirty_price
            ),
            "accrued_interest": (
                snapshot.accrued_interest
            ),
            "modified_duration": (
                snapshot.modified_duration
            ),
            "macaulay_duration": (
                snapshot.macaulay_duration
            ),
            "convexity": (
                snapshot.convexity
            ),
            "dv01_per_100": (
                snapshot.dv01_per_100
            ),
            "dv01_per_eur_1m": (
                snapshot.dv01_per_eur_1m
            ),
            "position_notional_eur": (
                snapshot
                .position_notional_eur
            ),
            "position_dv01_eur": (
                snapshot.position_dv01_eur
            ),
        }
        for snapshot
        in snapshots
    ]

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "country",
                "benchmark_tenor_years",
                "isin",
            ]
        )
        .reset_index(
            drop=True
        )
    )