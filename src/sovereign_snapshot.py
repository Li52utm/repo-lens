from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Iterable

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

SOVEREIGN_HISTORY_PATH: Final[Path] = Path(
    "data/market/sovereign_history.csv"
)

ITALY_EXACT_MARKET_PATH: Final[Path] = Path(
    "data/market/italy_nominal_btp_market.csv"
)

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
    Describe the origin of data used in a sovereign snapshot.

    Existing values are retained for backwards compatibility. The additional
    values let the same snapshot module carry exact public/reference,
    broker/desk and RepoLens-derived observations without inventing a second
    market-snapshot abstraction elsewhere in the codebase.
    """

    OFFICIAL_DAILY = "OFFICIAL_DAILY"
    OFFICIAL_REFERENCE = "OFFICIAL_REFERENCE"
    PUBLIC_REFERENCE = "PUBLIC_REFERENCE"
    DELAYED_PUBLIC_REFERENCE = "DELAYED_PUBLIC_REFERENCE"
    DESK_INPUT = "DESK_INPUT"
    BROKER_INPUT = "BROKER_INPUT"
    REPOLENS_DERIVED = "REPOLENS_DERIVED"
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

# ---------------------------------------------------------------------------
# Exact sovereign market observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactSovereignMarketObservation:
    """
    One sourced exact-ISIN market observation.

    This is deliberately separate from SovereignSnapshotResult, which is a
    valuation/risk result. Keeping both concepts in this same module avoids
    creating a second competing "sovereign snapshot" file while preserving the
    existing analytics API.
    """

    isin: str
    observation_date: date
    price_per_100: float | None
    yield_percent: float | None
    source_name: str
    data_status: SnapshotDataStatus
    observation_time: datetime | None = None

    @property
    def price_available(self) -> bool:
        return self.price_per_100 is not None

    @property
    def yield_available(self) -> bool:
        return self.yield_percent is not None

    @property
    def completeness_score(self) -> int:
        return int(
            self.price_available
        ) + int(
            self.yield_available
        )


def _snapshot_text(
    row: dict[str, str],
    *keys: str,
) -> str:
    for key in keys:
        value = (
            row.get(
                key
            )
            or ""
        ).strip()

        if value:
            return value

    return ""


def _snapshot_float_or_none(
    value: str,
) -> float | None:
    text = (
        value
        .replace(
            "\xa0",
            "",
        )
        .replace(
            "%",
            "",
        )
        .replace(
            " ",
            "",
        )
        .strip()
    )

    if not text:
        return None

    if (
        "," in text
        and "." in text
    ):
        if text.rfind(
            ","
        ) > text.rfind(
            "."
        ):
            text = (
                text
                .replace(
                    ".",
                    "",
                )
                .replace(
                    ",",
                    ".",
                )
            )
        else:
            text = text.replace(
                ",",
                "",
            )

    elif "," in text:
        text = text.replace(
            ",",
            ".",
        )

    try:
        value_float = float(
            text
        )
    except ValueError:
        return None

    if not np.isfinite(
        value_float
    ):
        return None

    return value_float


def _snapshot_date_or_none(
    value: str,
) -> date | None:
    text = value.strip()

    if not text:
        return None

    try:
        return date.fromisoformat(
            text[
                :10
            ]
        )
    except ValueError:
        pass

    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            pass

    return None


def _snapshot_datetime_or_none(
    value: str,
) -> datetime | None:
    text = value.strip()

    if not text:
        return None

    try:
        return datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None


def _normalise_snapshot_status(
    value: str,
) -> SnapshotDataStatus:
    normalised = (
        value
        .strip()
        .upper()
        .replace(
            " ",
            "_",
        )
    )

    aliases = {
        "OFFICIAL": SnapshotDataStatus.OFFICIAL_DAILY,
        "OFFICIAL_DAILY": SnapshotDataStatus.OFFICIAL_DAILY,
        "OFFICIAL_REFERENCE": SnapshotDataStatus.OFFICIAL_REFERENCE,
        "PUBLIC_REFERENCE": SnapshotDataStatus.PUBLIC_REFERENCE,
        "DELAYED_PUBLIC_REFERENCE": (
            SnapshotDataStatus.DELAYED_PUBLIC_REFERENCE
        ),
        "DESK_INPUT": SnapshotDataStatus.DESK_INPUT,
        "BROKER_INPUT": SnapshotDataStatus.BROKER_INPUT,
        "REPOLENS_DERIVED": SnapshotDataStatus.REPOLENS_DERIVED,
        "UNAVAILABLE": SnapshotDataStatus.UNAVAILABLE,
    }

    return aliases.get(
        normalised,
        SnapshotDataStatus.UNAVAILABLE,
    )


def _snapshot_status_priority(
    status: SnapshotDataStatus,
) -> int:
    """
    Tie-break only.

    Recency remains the primary rule. We do not allow an older "better" source
    to overwrite a newer exact observation.
    """
    priorities = {
        SnapshotDataStatus.OFFICIAL_DAILY: 60,
        SnapshotDataStatus.OFFICIAL_REFERENCE: 55,
        SnapshotDataStatus.BROKER_INPUT: 50,
        SnapshotDataStatus.DESK_INPUT: 50,
        SnapshotDataStatus.PUBLIC_REFERENCE: 40,
        SnapshotDataStatus.DELAYED_PUBLIC_REFERENCE: 35,
        SnapshotDataStatus.REPOLENS_DERIVED: 20,
        SnapshotDataStatus.UNAVAILABLE: 0,
    }

    return priorities[
        status
    ]


def _exact_observation_sort_key(
    observation: ExactSovereignMarketObservation,
) -> tuple[
    date,
    int,
    int,
    datetime,
]:
    return (
        observation.observation_date,
        observation.completeness_score,
        _snapshot_status_priority(
            observation.data_status
        ),
        (
            observation.observation_time
            if observation.observation_time is not None
            else datetime.min
        ),
    )


def load_persisted_exact_market_observations(
    path: Path = SOVEREIGN_HISTORY_PATH,
) -> tuple[
    ExactSovereignMarketObservation,
    ...,
]:
    """
    Load exact-ISIN observations already persisted by RepoLens.

    Benchmark/index yields are intentionally excluded because they are not
    exact observations for an individual cash bond.
    """
    if not path.exists():
        return ()

    observations: list[
        ExactSovereignMarketObservation
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            isin = _snapshot_text(
                row,
                "isin",
            ).upper()

            observation_date = _snapshot_date_or_none(
                _snapshot_text(
                    row,
                    "observation_date",
                    "as_of_date",
                )
            )

            price = _snapshot_float_or_none(
                _snapshot_text(
                    row,
                    "price_per_100",
                    "clean_price",
                    "price",
                )
            )

            yield_percent = _snapshot_float_or_none(
                _snapshot_text(
                    row,
                    "yield_percent",
                    "ytm_percent",
                    "yield",
                )
            )

            if (
                len(
                    isin
                )
                != 12
                or observation_date is None
                or (
                    price is None
                    and yield_percent is None
                )
            ):
                continue

            source_name = _snapshot_text(
                row,
                "source_name",
            ) or "Persisted exact sovereign observation"

            status = _normalise_snapshot_status(
                _snapshot_text(
                    row,
                    "data_status",
                )
            )

            observations.append(
                ExactSovereignMarketObservation(
                    isin=isin,
                    observation_date=observation_date,
                    price_per_100=price,
                    yield_percent=yield_percent,
                    source_name=source_name,
                    data_status=status,
                    observation_time=(
                        _snapshot_datetime_or_none(
                            _snapshot_text(
                                row,
                                "observation_timestamp",
                                "as_of_timestamp",
                                "timestamp",
                            )
                        )
                    ),
                )
            )

    return tuple(
        observations
    )


def load_italy_exact_market_observations(
    path: Path = ITALY_EXACT_MARKET_PATH,
) -> tuple[
    ExactSovereignMarketObservation,
    ...,
]:
    """
    Load exact-ISIN Borsa observations from the expanded Italy market file.

    Newer files can contain:
    - last_price from the Borsa list page;
    - reference_price from the instrument detail page;
    - yield_percent from the instrument detail page;
    - observation_date from the instrument detail page.

    Backwards compatibility:
    - older price-only files remain valid;
    - if no observation_date is present, file mtime is used as a fallback;
    - no yield is manufactured from price in this loader.

    Source/status remain explicit so RepoLens can distinguish sourced reference
    observations from later REPOLENS_DERIVED analytics.
    """
    if not path.exists():
        return ()

    file_date = date.fromtimestamp(
        path.stat().st_mtime
    )

    observations: list[
        ExactSovereignMarketObservation
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            isin = _snapshot_text(
                row,
                "isin",
            ).upper()

            if len(
                isin
            ) != 12:
                continue

            reference_price = _snapshot_float_or_none(
                _snapshot_text(
                    row,
                    "reference_price",
                )
            )

            last_price = _snapshot_float_or_none(
                _snapshot_text(
                    row,
                    "last_price",
                    "price_per_100",
                    "price",
                )
            )

            price = (
                reference_price
                if reference_price is not None
                else last_price
            )

            yield_percent = _snapshot_float_or_none(
                _snapshot_text(
                    row,
                    "yield_percent",
                    "ytm_percent",
                    "yield",
                )
            )

            if (
                price is None
                and yield_percent is None
            ):
                continue

            observation_date = _snapshot_date_or_none(
                _snapshot_text(
                    row,
                    "observation_date",
                    "reference_date",
                    "as_of_date",
                )
            )

            if observation_date is None:
                observation_date = file_date

            source_name = _snapshot_text(
                row,
                "source_name",
            ) or "Borsa Italiana MOT"

            status = _normalise_snapshot_status(
                _snapshot_text(
                    row,
                    "data_status",
                )
                or "PUBLIC_REFERENCE"
            )

            observations.append(
                ExactSovereignMarketObservation(
                    isin=isin,
                    observation_date=observation_date,
                    price_per_100=price,
                    yield_percent=yield_percent,
                    source_name=source_name,
                    data_status=status,
                )
            )

    return tuple(
        observations
    )


def latest_exact_market_observations_by_isin(
    observations: Iterable[
        ExactSovereignMarketObservation
    ],
) -> dict[
    str,
    ExactSovereignMarketObservation,
]:
    """
    Select one deterministic latest exact observation per ISIN.
    """
    best: dict[
        str,
        ExactSovereignMarketObservation,
    ] = {}

    for observation in observations:
        existing = best.get(
            observation.isin
        )

        if (
            existing is None
            or _exact_observation_sort_key(
                observation
            )
            > _exact_observation_sort_key(
                existing
            )
        ):
            best[
                observation.isin
            ] = observation

    return best


def load_latest_exact_market_observations(
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
) -> dict[
    str,
    ExactSovereignMarketObservation,
]:
    """
    Unified exact-observation loader used by snapshots and downstream RV work.
    """
    observations = (
        *load_persisted_exact_market_observations(
            history_path
        ),
        *load_italy_exact_market_observations(
            italy_market_path
        ),
    )

    return latest_exact_market_observations_by_isin(
        observations
    )


def persisted_yield_inputs_for_instruments(
    instruments: tuple[
        SovereignInstrument,
        ...,
    ],
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
) -> tuple[
    SovereignYieldInput,
    ...,
]:
    """
    Convert exact persisted yields into the EXISTING snapshot input contract.

    This is the bridge from the data-ingestion work into the original
    valuation/risk snapshot engine. Only exact yields are admitted.
    Price-only rows remain price-only and are not reverse-engineered here.
    """
    latest = load_latest_exact_market_observations(
        history_path=history_path,
        italy_market_path=italy_market_path,
    )

    yield_inputs: list[
        SovereignYieldInput
    ] = []

    for instrument in instruments:
        observation = latest.get(
            instrument.isin
        )

        if (
            observation is None
            or observation.yield_percent is None
        ):
            continue

        yield_inputs.append(
            SovereignYieldInput(
                isin=instrument.isin,
                yield_percent=observation.yield_percent,
                observation_date=observation.observation_date,
                source_name=observation.source_name,
                data_status=observation.data_status,
            )
        )

    return tuple(
        yield_inputs
    )


def build_registry_snapshot_from_persisted_market(
    german_curve: pd.DataFrame,
    settlement_date: date,
    position_notional_eur: float = DEFAULT_POSITION_NOTIONAL_EUR,
    instruments: tuple[
        SovereignInstrument,
        ...,
    ] = SOVEREIGN_INSTRUMENTS,
    explicit_yield_inputs: tuple[
        SovereignYieldInput,
        ...,
    ] = (),
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
) -> pd.DataFrame:
    """
    Build the existing valuation/risk snapshot using exact persisted market
    yields automatically.

    Explicit desk/broker inputs supplied by the caller override a persisted
    input for the same ISIN. This preserves the old workflow while removing
    unnecessary manual entry where exact sourced yields already exist.
    """
    persisted_inputs = {
        item.isin: item
        for item in persisted_yield_inputs_for_instruments(
            instruments,
            history_path=history_path,
            italy_market_path=italy_market_path,
        )
    }

    explicit_inputs = {
        item.isin: item
        for item in explicit_yield_inputs
    }

    persisted_inputs.update(
        explicit_inputs
    )

    return build_registry_snapshot(
        german_curve=german_curve,
        settlement_date=settlement_date,
        position_notional_eur=position_notional_eur,
        explicit_yield_inputs=tuple(
            persisted_inputs.values()
        ),
        instruments=instruments,
    )


def sovereign_market_coverage_frame(
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """
    Desk-facing coverage table for the FULL dynamic reference universe.

    This function imports the already-existing sovereign_desk_universe module
    lazily so the original snapshot analytics remain independent and backwards
    compatible.

    Reference coverage and executable/observed market coverage are deliberately
    kept separate.
    """
    from src.sovereign_desk_universe import (
        load_desk_sovereign_universe,
    )

    if as_of_date is None:
        as_of_date = date.today()

    instruments = load_desk_sovereign_universe()

    latest = load_latest_exact_market_observations(
        history_path=history_path,
        italy_market_path=italy_market_path,
    )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for instrument in instruments:
        observation = latest.get(
            instrument.isin
        )

        if observation is None:
            stale_days = np.nan
        else:
            stale_days = max(
                (
                    as_of_date
                    - observation.observation_date
                ).days,
                0,
            )

        rows.append(
            {
                "country": instrument.country.value,
                "display_name": instrument.display_name,
                "isin": instrument.isin,
                "instrument_type": instrument.instrument_type,
                "currency": instrument.currency,
                "maturity_date": instrument.maturity_date,
                "maturity_bucket_years": (
                    instrument.benchmark_tenor_years
                ),
                "reference_source": instrument.source_name,
                "reference_status": instrument.data_status,
                "market_observation_available": (
                    observation is not None
                ),
                "price_per_100": (
                    observation.price_per_100
                    if observation is not None
                    else np.nan
                ),
                "yield_percent": (
                    observation.yield_percent
                    if observation is not None
                    else np.nan
                ),
                "observation_date": (
                    observation.observation_date
                    if observation is not None
                    else None
                ),
                "market_source": (
                    observation.source_name
                    if observation is not None
                    else "UNAVAILABLE"
                ),
                "market_status": (
                    observation.data_status.value
                    if observation is not None
                    else SnapshotDataStatus.UNAVAILABLE.value
                ),
                "price_available": bool(
                    observation is not None
                    and observation.price_available
                ),
                "yield_available": bool(
                    observation is not None
                    and observation.yield_available
                ),
                "stale_days": stale_days,
            }
        )

    if not rows:
        raise SovereignSnapshotValidationError(
            "The sovereign desk universe is empty."
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "country",
                "maturity_date",
                "isin",
            ]
        )
        .reset_index(
            drop=True
        )
    )

# ---------------------------------------------------------------------------
# Curve-ready exact-yield dataset
# ---------------------------------------------------------------------------


def _validate_curve_readiness_inputs(
    *,
    max_stale_days: int | None,
    minimum_curve_points: int,
) -> None:
    if (
        max_stale_days is not None
        and max_stale_days < 0
    ):
        raise SovereignSnapshotValidationError(
            "max_stale_days must be non-negative or None."
        )

    if minimum_curve_points < 2:
        raise SovereignSnapshotValidationError(
            "minimum_curve_points must be at least 2."
        )


def sovereign_curve_ready_frame(
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
    as_of_date: date | None = None,
    max_stale_days: int | None = None,
    minimum_curve_points: int = 3,
) -> pd.DataFrame:
    """
    Build the exact-yield dataset that feeds sovereign curve/RV analytics.

    This function DOES NOT fit, interpolate or smooth a curve.

    Eligibility rules:
    - exact ISIN-level yield must exist;
    - maturity must be after as_of_date;
    - if max_stale_days is supplied, the observation must not exceed it;
    - country/currency groups are marked curve_ready only when they contain
      at least minimum_curve_points eligible observations.

    minimum_curve_points defaults to 3 because three distinct maturity points
    are the smallest useful set for a curve workflow that intends to observe
    more than a simple two-point line. It is an explicit configurable
    operational threshold, not a market convention.

    The output keeps rejected rows as well as eligible rows so the desk can
    audit why an instrument was excluded from modelling.
    """
    _validate_curve_readiness_inputs(
        max_stale_days=max_stale_days,
        minimum_curve_points=minimum_curve_points,
    )

    if as_of_date is None:
        as_of_date = date.today()

    coverage = sovereign_market_coverage_frame(
        history_path=history_path,
        italy_market_path=italy_market_path,
        as_of_date=as_of_date,
    ).copy()

    coverage[
        "maturity_date"
    ] = pd.to_datetime(
        coverage[
            "maturity_date"
        ],
        errors="coerce",
    )

    coverage[
        "observation_date"
    ] = pd.to_datetime(
        coverage[
            "observation_date"
        ],
        errors="coerce",
    )

    as_of_timestamp = pd.Timestamp(
        as_of_date
    )

    coverage[
        "days_to_maturity"
    ] = (
        coverage[
            "maturity_date"
        ]
        - as_of_timestamp
    ).dt.days

    coverage[
        "years_to_maturity"
    ] = (
        coverage[
            "days_to_maturity"
        ]
        / 365.25
    )

    coverage[
        "has_exact_yield"
    ] = (
        coverage[
            "yield_available"
        ].fillna(
            False
        )
        & pd.to_numeric(
            coverage[
                "yield_percent"
            ],
            errors="coerce",
        ).notna()
    )

    coverage[
        "is_matured_or_maturing_today"
    ] = (
        coverage[
            "days_to_maturity"
        ].fillna(
            -1
        )
        <= 0
    )

    if max_stale_days is None:
        coverage[
            "is_stale"
        ] = False
    else:
        stale_numeric = pd.to_numeric(
            coverage[
                "stale_days"
            ],
            errors="coerce",
        )

        coverage[
            "is_stale"
        ] = (
            stale_numeric.isna()
            | stale_numeric.gt(
                max_stale_days
            )
        )

    coverage[
        "curve_point_eligible"
    ] = (
        coverage[
            "has_exact_yield"
        ]
        & ~coverage[
            "is_matured_or_maturing_today"
        ]
        & ~coverage[
            "is_stale"
        ]
    )

    reason = pd.Series(
        "ELIGIBLE",
        index=coverage.index,
        dtype="string",
    )

    reason = reason.mask(
        ~coverage[
            "has_exact_yield"
        ],
        "NO_EXACT_YIELD",
    )

    reason = reason.mask(
        coverage[
            "is_matured_or_maturing_today"
        ],
        "MATURED",
    )

    if max_stale_days is not None:
        reason = reason.mask(
            coverage[
                "is_stale"
        ]
        & coverage[
            "has_exact_yield"
        ]
        & ~coverage[
            "is_matured_or_maturing_today"
        ],
            "STALE",
        )

    coverage[
        "curve_exclusion_reason"
    ] = reason

    eligible = coverage.loc[
        coverage[
            "curve_point_eligible"
        ]
    ]

    if eligible.empty:
        group_counts = pd.DataFrame(
            columns=[
                "country",
                "currency",
                "eligible_curve_points",
            ]
        )
    else:
        group_counts = (
            eligible
            .groupby(
                [
                    "country",
                    "currency",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "eligible_curve_points",
                }
            )
        )

    coverage = coverage.merge(
        group_counts,
        on=[
            "country",
            "currency",
        ],
        how="left",
    )

    coverage[
        "eligible_curve_points"
    ] = (
        coverage[
            "eligible_curve_points"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    coverage[
        "minimum_curve_points"
    ] = int(
        minimum_curve_points
    )

    coverage[
        "country_curve_ready"
    ] = (
        coverage[
            "eligible_curve_points"
        ]
        >= minimum_curve_points
    )

    return (
        coverage
        .sort_values(
            [
                "country",
                "currency",
                "years_to_maturity",
                "isin",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def sovereign_curve_points(
    *,
    country: str,
    currency: str | None = None,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
    as_of_date: date | None = None,
    max_stale_days: int | None = None,
    minimum_curve_points: int = 3,
    require_curve_ready: bool = True,
) -> pd.DataFrame:
    """
    Return auditable exact-yield points for one sovereign country.

    No interpolation or fair-value estimates are produced here.
    """
    country_name = country.strip()

    if not country_name:
        raise SovereignSnapshotValidationError(
            "country must not be empty."
        )

    frame = sovereign_curve_ready_frame(
        history_path=history_path,
        italy_market_path=italy_market_path,
        as_of_date=as_of_date,
        max_stale_days=max_stale_days,
        minimum_curve_points=minimum_curve_points,
    )

    selected = frame.loc[
        frame[
            "country"
        ].eq(
            country_name
        )
    ].copy()

    if currency is not None:
        currency_code = currency.strip().upper()

        if not currency_code:
            raise SovereignSnapshotValidationError(
                "currency must not be empty when supplied."
            )

        selected = selected.loc[
            selected[
                "currency"
            ].astype(
                str
            ).str.upper().eq(
                currency_code
            )
        ]

    selected = selected.loc[
        selected[
            "curve_point_eligible"
        ]
    ].copy()

    if selected.empty:
        raise SovereignSnapshotValidationError(
            "No eligible exact-yield curve points are available for "
            f"{country_name}."
        )

    ready = bool(
        selected[
            "country_curve_ready"
        ].all()
    )

    if (
        require_curve_ready
        and not ready
    ):
        available_points = int(
            selected[
                "eligible_curve_points"
            ].max()
        )

        raise SovereignSnapshotValidationError(
            f"{country_name} has {available_points} eligible exact-yield "
            f"point(s); at least {minimum_curve_points} are required by the "
            "current RepoLens curve-readiness setting."
        )

    columns = [
        "country",
        "currency",
        "display_name",
        "isin",
        "instrument_type",
        "maturity_date",
        "days_to_maturity",
        "years_to_maturity",
        "price_per_100",
        "yield_percent",
        "observation_date",
        "market_source",
        "market_status",
        "stale_days",
        "eligible_curve_points",
        "minimum_curve_points",
        "country_curve_ready",
    ]

    return (
        selected[
            columns
        ]
        .sort_values(
            [
                "years_to_maturity",
                "isin",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def sovereign_curve_readiness_summary(
    *,
    history_path: Path = SOVEREIGN_HISTORY_PATH,
    italy_market_path: Path = ITALY_EXACT_MARKET_PATH,
    as_of_date: date | None = None,
    max_stale_days: int | None = None,
    minimum_curve_points: int = 3,
) -> pd.DataFrame:
    """
    Summarise modelling readiness by country and currency.

    This is a desk/data-quality diagnostic, not a trading signal.
    """
    frame = sovereign_curve_ready_frame(
        history_path=history_path,
        italy_market_path=italy_market_path,
        as_of_date=as_of_date,
        max_stale_days=max_stale_days,
        minimum_curve_points=minimum_curve_points,
    )

    grouped = (
        frame
        .groupby(
            [
                "country",
                "currency",
            ],
            as_index=False,
        )
        .agg(
            reference_instruments=(
                "isin",
                "count",
            ),
            market_observed=(
                "market_observation_available",
                "sum",
            ),
            priced=(
                "price_available",
                "sum",
            ),
            exact_yields=(
                "has_exact_yield",
                "sum",
            ),
            eligible_curve_points=(
                "curve_point_eligible",
                "sum",
            ),
            oldest_observation_date=(
                "observation_date",
                "min",
            ),
            newest_observation_date=(
                "observation_date",
                "max",
            ),
        )
    )

    grouped[
        "minimum_curve_points"
    ] = int(
        minimum_curve_points
    )

    grouped[
        "curve_ready"
    ] = (
        grouped[
            "eligible_curve_points"
        ]
        >= minimum_curve_points
    )

    grouped[
        "yield_coverage_percent"
    ] = np.where(
        grouped[
            "reference_instruments"
        ].gt(
            0
        ),
        (
            grouped[
                "exact_yields"
            ]
            / grouped[
                "reference_instruments"
            ]
            * 100.0
        ),
        np.nan,
    )

    grouped[
        "eligible_coverage_percent"
    ] = np.where(
        grouped[
            "reference_instruments"
        ].gt(
            0
        ),
        (
            grouped[
                "eligible_curve_points"
            ]
            / grouped[
                "reference_instruments"
            ]
            * 100.0
        ),
        np.nan,
    )

    return (
        grouped
        .sort_values(
            [
                "curve_ready",
                "eligible_curve_points",
                "country",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )