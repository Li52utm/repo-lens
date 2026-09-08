

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
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
    sovereign_curve_points,
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


DEFAULT_SOVEREIGN_RV_RESIDUAL_HISTORY_PATH: Final[Path] = (
    Path("data") / "market" / "sovereign_rv_residual_history.csv"
)

LOCAL_CURVE_MODEL_VERSION: Final[str] = "LOCAL_LINEAR_PLAIN_FIXED_V1"

SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "observation_date", "country", "currency", "isin", "display_name",
    "maturity_date", "years_to_maturity", "actual_yield_percent",
    "fair_yield_percent", "residual_bp", "relative_value_label",
    "curve_location", "shorter_isin", "shorter_gap_years", "longer_isin",
    "longer_gap_years", "total_bracket_width_years",
    "interpolation_quality_score", "interpolation_quality", "market_source",
    "market_status", "model_name", "model_version", "model_status",
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


class CurveInstrumentStructure(StrEnum):
    """
    Structural eligibility for plain nominal sovereign cash-curve modelling.
    """

    PLAIN_FIXED_RATE_NOMINAL = "PLAIN_FIXED_RATE_NOMINAL"
    STRUCTURED_RETAIL_STEP_UP = "STRUCTURED_RETAIL_STEP_UP"
    INFLATION_LINKED = "INFLATION_LINKED"
    STRIP = "STRIP"


def classify_curve_instrument_structure(
    display_name: str,
) -> CurveInstrumentStructure:
    """
    Classify an Italian sovereign instrument from its sourced display name.

    RepoLens' first local cash-curve model is intentionally restricted to
    plain fixed-rate nominal BTPs. Retail step-up products such as BTP Valore,
    BTP Futura and BTP Più are not comparable to a plain fixed-coupon curve
    and therefore must not be used either as targets or interpolation anchors.

    Inflation-linked and strip instruments are also excluded defensively even
    though the upstream nominal-universe filter should already remove them.
    """
    name = (
        display_name
        .strip()
        .upper()
        .replace("Ù", "U")
        .replace("’", "'")
    )

    if "STRIP" in name:
        return CurveInstrumentStructure.STRIP

    if (
        "BTPI" in name
        or "BTP ITALIA" in name
        or "BTP€I" in name
    ):
        return CurveInstrumentStructure.INFLATION_LINKED

    if (
        "VALORE" in name
        or "FUTURA" in name
        or "BTP PIU" in name
        or "BTP PIU'" in name
    ):
        return CurveInstrumentStructure.STRUCTURED_RETAIL_STEP_UP

    return CurveInstrumentStructure.PLAIN_FIXED_RATE_NOMINAL


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

# ---------------------------------------------------------------------------
# Local sovereign curve fair value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalCurveFairValueResult:
    """
    Store one leave-one-out local-curve fair-value result.

    Fair value is estimated from the nearest strictly shorter and strictly
    longer eligible exact-yield bonds after removing the target bond itself.

    No extrapolation is performed. Edge bonds without a two-sided maturity
    bracket are reported as unavailable rather than assigned a synthetic fair
    value.
    """

    isin: str
    display_name: str
    country: str
    currency: str
    maturity_date: date
    years_to_maturity: float
    actual_yield_percent: float
    fair_yield_percent: float
    residual_bp: float
    relative_value_label: str
    shorter_isin: str
    shorter_years_to_maturity: float
    shorter_yield_percent: float
    longer_isin: str
    longer_years_to_maturity: float
    longer_yield_percent: float
    shorter_weight: float
    longer_weight: float
    observation_date: date | None
    market_source: str
    market_status: str


def _validate_curve_residual_input(
    curve_points: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalise exact-yield curve points for local fair value.
    """
    required_columns = {
        "country",
        "currency",
        "display_name",
        "isin",
        "maturity_date",
        "years_to_maturity",
        "yield_percent",
        "observation_date",
        "market_source",
        "market_status",
    }

    missing = sorted(
        required_columns.difference(
            curve_points.columns
        )
    )

    if missing:
        raise RelativeValueValidationError(
            "Curve points are missing required columns: "
            + ", ".join(
                missing
            )
            + "."
        )

    if curve_points.empty:
        raise RelativeValueValidationError(
            "Curve points must not be empty."
        )

    frame = curve_points.copy()

    frame[
        "years_to_maturity"
    ] = pd.to_numeric(
        frame[
            "years_to_maturity"
        ],
        errors="coerce",
    )

    frame[
        "yield_percent"
    ] = pd.to_numeric(
        frame[
            "yield_percent"
        ],
        errors="coerce",
    )

    frame[
        "maturity_date"
    ] = pd.to_datetime(
        frame[
            "maturity_date"
        ],
        errors="coerce",
    )

    frame[
        "observation_date"
    ] = pd.to_datetime(
        frame[
            "observation_date"
        ],
        errors="coerce",
    )

    valid = (
        frame[
            "years_to_maturity"
        ].notna()
        & frame[
            "yield_percent"
        ].notna()
        & frame[
            "maturity_date"
        ].notna()
        & np.isfinite(
            frame[
                "years_to_maturity"
            ]
        )
        & np.isfinite(
            frame[
                "yield_percent"
            ]
        )
        & frame[
            "years_to_maturity"
        ].gt(
            0.0
        )
    )

    frame = frame.loc[
        valid
    ].copy()

    if len(
        frame
    ) < 3:
        raise RelativeValueValidationError(
            "At least three valid exact-yield curve points are required."
        )

    if frame[
        "isin"
    ].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame[
                    "isin"
                ].duplicated(
                    keep=False
                ),
                "isin",
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        raise RelativeValueValidationError(
            "Curve points contain duplicate ISINs: "
            + ", ".join(
                duplicates
            )
            + "."
        )

    country_count = frame[
        "country"
    ].nunique(
        dropna=False
    )

    currency_count = frame[
        "currency"
    ].nunique(
        dropna=False
    )

    if (
        country_count != 1
        or currency_count != 1
    ):
        raise RelativeValueValidationError(
            "Local fair value requires one country and one currency per curve."
        )

    frame[
        "instrument_structure"
    ] = frame[
        "display_name"
    ].astype(
        str
    ).map(
        lambda value: classify_curve_instrument_structure(
            value
        ).value
    )

    frame[
        "curve_model_eligible"
    ] = frame[
        "instrument_structure"
    ].eq(
        CurveInstrumentStructure.PLAIN_FIXED_RATE_NOMINAL.value
    )

    frame[
        "curve_model_exclusion_reason"
    ] = np.where(
        frame[
            "curve_model_eligible"
        ],
        None,
        frame[
            "instrument_structure"
        ],
    )

    if (
        frame[
            "curve_model_eligible"
        ].sum()
        < 3
    ):
        raise RelativeValueValidationError(
            "At least three eligible plain fixed-rate nominal curve points "
            "are required after structural filtering."
        )

    return (
        frame
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


def _relative_value_label(
    residual_bp: float,
) -> str:
    """
    Describe residual direction without imposing an arbitrary signal threshold.

    Positive residual:
        actual yield > fair yield -> bond is cheap to the local curve.

    Negative residual:
        actual yield < fair yield -> bond is rich to the local curve.
    """
    if residual_bp > 1e-12:
        return "CHEAP"

    if residual_bp < -1e-12:
        return "RICH"

    return "ON_CURVE"



def _curve_location_label(
    years_to_maturity: float,
) -> str:
    """
    Return a descriptive maturity-sector label.

    These are presentation buckets only, not trading thresholds:
    - FRONT_END: <= 2Y
    - BELLY: > 2Y and <= 10Y
    - LONG_END: > 10Y
    """
    if years_to_maturity <= 2.0:
        return "FRONT_END"

    if years_to_maturity <= 10.0:
        return "BELLY"

    return "LONG_END"


def _interpolation_quality(
    *,
    shorter_gap_years: float,
    longer_gap_years: float,
    median_bracket_width_years: float,
) -> tuple[
    float,
    str,
]:
    """
    Score the geometry of a two-sided local interpolation.

    The score is intentionally simple and auditable:

        balance = min(shorter gap, longer gap) / max(shorter gap, longer gap)

        density = min(
            1,
            median cross-sectional bracket width / this bracket width
        )

        quality score = balance * density

    A perfectly balanced target in a bracket no wider than the cross-sectional
    median scores 1.0.

    Labels are descriptive:
    - HIGH   >= 0.67
    - MEDIUM >= 0.33
    - LOW    < 0.33

    The label is not a trade recommendation and does not alter residual sign.
    """
    for value, name in (
        (
            shorter_gap_years,
            "shorter_gap_years",
        ),
        (
            longer_gap_years,
            "longer_gap_years",
        ),
        (
            median_bracket_width_years,
            "median_bracket_width_years",
        ),
    ):
        if (
            not np.isfinite(
                value
            )
            or value <= 0.0
        ):
            raise RelativeValueValidationError(
                f"{name} must be finite and positive."
            )

    total_width = (
        shorter_gap_years
        + longer_gap_years
    )

    balance = (
        min(
            shorter_gap_years,
            longer_gap_years,
        )
        / max(
            shorter_gap_years,
            longer_gap_years,
        )
    )

    density = min(
        1.0,
        median_bracket_width_years
        / total_width,
    )

    score = (
        balance
        * density
    )

    if score >= 0.67:
        label = "HIGH"
    elif score >= 0.33:
        label = "MEDIUM"
    else:
        label = "LOW"

    return (
        score,
        label,
    )


def calculate_local_curve_fair_values(
    curve_points: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate leave-one-out local fair values for a sovereign cash curve.

    Method
    ------
    For each eligible plain fixed-rate nominal target bond:
    1. remove the target bond;
    2. exclude non-comparable structures from the anchor set;
    3. find the nearest strictly shorter-maturity eligible bond;
    4. find the nearest strictly longer-maturity eligible bond;
    5. linearly interpolate yield between those two neighbours at the target
       maturity;
    6. residual_bp = (actual yield - fair yield) * 100.

    The method is intentionally transparent. It does not fit through the
    target bond itself, does not extrapolate beyond observed maturities and
    does not claim that the resulting fair value is an executable price.

    Bonds at the shortest/longest edge of the observed curve, or bonds without
    a strict two-sided maturity bracket, are retained with fair value N/A and
    an explicit exclusion reason.
    """
    frame = _validate_curve_residual_input(
        curve_points
    )

    rows: list[
        dict[str, object]
    ] = []

    for target_index, target in frame.iterrows():
        target_years = float(
            target[
                "years_to_maturity"
            ]
        )

        base_row: dict[
            str,
            object,
        ] = {
            "country": target[
                "country"
            ],
            "currency": target[
                "currency"
            ],
            "display_name": target[
                "display_name"
            ],
            "isin": target[
                "isin"
            ],
            "maturity_date": (
                pd.Timestamp(
                    target[
                        "maturity_date"
                    ]
                ).date()
            ),
            "years_to_maturity": target_years,
            "curve_location": _curve_location_label(
                target_years
            ),
            "actual_yield_percent": float(
                target[
                    "yield_percent"
                ]
            ),
            "observation_date": (
                None
                if pd.isna(
                    target[
                        "observation_date"
                    ]
                )
                else pd.Timestamp(
                    target[
                        "observation_date"
                    ]
                ).date()
            ),
            "market_source": target[
                "market_source"
            ],
            "market_status": target[
                "market_status"
            ],
            "instrument_structure": target[
                "instrument_structure"
            ],
            "curve_model_eligible": bool(
                target[
                    "curve_model_eligible"
                ]
            ),
            "curve_model_exclusion_reason": target[
                "curve_model_exclusion_reason"
            ],
        }

        if not bool(
            target[
                "curve_model_eligible"
            ]
        ):
            rows.append(
                {
                    **base_row,
                    "fair_value_available": False,
                    "fair_yield_percent": np.nan,
                    "residual_bp": np.nan,
                    "relative_value_label": "UNAVAILABLE",
                    "shorter_isin": None,
                    "shorter_years_to_maturity": np.nan,
                    "shorter_yield_percent": np.nan,
                    "shorter_gap_years": np.nan,
                    "longer_isin": None,
                    "longer_years_to_maturity": np.nan,
                    "longer_yield_percent": np.nan,
                    "longer_gap_years": np.nan,
                    "total_bracket_width_years": np.nan,
                    "interpolation_balance_ratio": np.nan,
                    "interpolation_quality_score": np.nan,
                    "interpolation_quality": "UNAVAILABLE",
                    "shorter_weight": np.nan,
                    "longer_weight": np.nan,
                    "fair_value_exclusion_reason": (
                        target[
                            "curve_model_exclusion_reason"
                        ]
                    ),
                    "model_name": (
                        "LEAVE_ONE_OUT_LOCAL_LINEAR"
                    ),
                    "model_version": LOCAL_CURVE_MODEL_VERSION,
                    "model_status": (
                        "REPOLENS_DERIVED"
                    ),
                }
            )

            continue

        other = frame.loc[
            frame[
                "curve_model_eligible"
            ]
        ].drop(
            index=target_index,
            errors="ignore",
        )

        shorter = other.loc[
            other[
                "years_to_maturity"
            ].lt(
                target_years
            )
        ]

        longer = other.loc[
            other[
                "years_to_maturity"
            ].gt(
                target_years
            )
        ]

        if (
            shorter.empty
            or longer.empty
        ):
            rows.append(
                {
                    **base_row,
                    "fair_value_available": False,
                    "fair_yield_percent": np.nan,
                    "residual_bp": np.nan,
                    "relative_value_label": "UNAVAILABLE",
                    "shorter_isin": (
                        None
                    ),
                    "shorter_years_to_maturity": np.nan,
                    "shorter_yield_percent": np.nan,
                    "shorter_gap_years": np.nan,
                    "longer_isin": (
                        None
                    ),
                    "longer_years_to_maturity": np.nan,
                    "longer_yield_percent": np.nan,
                    "longer_gap_years": np.nan,
                    "total_bracket_width_years": np.nan,
                    "interpolation_balance_ratio": np.nan,
                    "interpolation_quality_score": np.nan,
                    "interpolation_quality": "UNAVAILABLE",
                    "shorter_weight": np.nan,
                    "longer_weight": np.nan,
                    "fair_value_exclusion_reason": (
                        "NO_TWO_SIDED_MATURITY_BRACKET"
                    ),
                    "model_name": (
                        "LEAVE_ONE_OUT_LOCAL_LINEAR"
                    ),
                    "model_version": LOCAL_CURVE_MODEL_VERSION,
                    "model_status": (
                        "REPOLENS_DERIVED"
                    ),
                }
            )

            continue

        shorter_row = shorter.iloc[
            -1
        ]

        longer_row = longer.iloc[
            0
        ]

        shorter_years = float(
            shorter_row[
                "years_to_maturity"
            ]
        )

        longer_years = float(
            longer_row[
                "years_to_maturity"
            ]
        )

        maturity_span = (
            longer_years
            - shorter_years
        )

        if maturity_span <= 0.0:
            raise RelativeValueValidationError(
                "Neighbour maturities must form a positive interpolation span."
            )

        shorter_gap_years = (
            target_years
            - shorter_years
        )

        longer_gap_years = (
            longer_years
            - target_years
        )

        interpolation_balance_ratio = (
            min(
                shorter_gap_years,
                longer_gap_years,
            )
            / max(
                shorter_gap_years,
                longer_gap_years,
            )
        )

        longer_weight = (
            shorter_gap_years
            / maturity_span
        )

        shorter_weight = (
            1.0
            - longer_weight
        )

        shorter_yield = float(
            shorter_row[
                "yield_percent"
            ]
        )

        longer_yield = float(
            longer_row[
                "yield_percent"
            ]
        )

        fair_yield = (
            shorter_weight
            * shorter_yield
            + longer_weight
            * longer_yield
        )

        actual_yield = float(
            target[
                "yield_percent"
            ]
        )

        residual_bp = (
            actual_yield
            - fair_yield
        ) * 100.0

        rows.append(
            {
                **base_row,
                "fair_value_available": True,
                "fair_yield_percent": fair_yield,
                "residual_bp": residual_bp,
                "relative_value_label": (
                    _relative_value_label(
                        residual_bp
                    )
                ),
                "shorter_isin": shorter_row[
                    "isin"
                ],
                "shorter_years_to_maturity": shorter_years,
                "shorter_yield_percent": shorter_yield,
                "shorter_gap_years": shorter_gap_years,
                "longer_isin": longer_row[
                    "isin"
                ],
                "longer_years_to_maturity": longer_years,
                "longer_yield_percent": longer_yield,
                "longer_gap_years": longer_gap_years,
                "total_bracket_width_years": maturity_span,
                "interpolation_balance_ratio": (
                    interpolation_balance_ratio
                ),
                "interpolation_quality_score": np.nan,
                "interpolation_quality": "PENDING",
                "shorter_weight": shorter_weight,
                "longer_weight": longer_weight,
                "fair_value_exclusion_reason": None,
                "model_name": (
                    "LEAVE_ONE_OUT_LOCAL_LINEAR"
                ),
                "model_version": LOCAL_CURVE_MODEL_VERSION,
                "model_status": (
                    "REPOLENS_DERIVED"
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    available_mask = result[
        "fair_value_available"
    ].eq(
        True
    )

    if available_mask.any():
        median_bracket_width_years = float(
            result.loc[
                available_mask,
                "total_bracket_width_years",
            ].median()
        )

        if (
            not np.isfinite(
                median_bracket_width_years
            )
            or median_bracket_width_years <= 0.0
        ):
            raise RelativeValueValidationError(
                "Median interpolation bracket width must be finite and positive."
            )

        quality_values = result.loc[
            available_mask,
            [
                "shorter_gap_years",
                "longer_gap_years",
            ],
        ].apply(
            lambda row: _interpolation_quality(
                shorter_gap_years=float(
                    row[
                        "shorter_gap_years"
                    ]
                ),
                longer_gap_years=float(
                    row[
                        "longer_gap_years"
                    ]
                ),
                median_bracket_width_years=(
                    median_bracket_width_years
                ),
            ),
            axis=1,
        )

        result.loc[
            available_mask,
            "interpolation_quality_score",
        ] = [
            item[
                0
            ]
            for item in quality_values
        ]

        result.loc[
            available_mask,
            "interpolation_quality",
        ] = [
            item[
                1
            ]
            for item in quality_values
        ]

        result.loc[
            available_mask,
            "median_bracket_width_years",
        ] = median_bracket_width_years
    else:
        result[
            "median_bracket_width_years"
        ] = np.nan

    result[
        "absolute_residual_bp"
    ] = result[
        "residual_bp"
    ].abs()

    return (
        result
        .sort_values(
            [
                "fair_value_available",
                "absolute_residual_bp",
                "years_to_maturity",
                "isin",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def build_country_local_curve_relative_value(
    *,
    country: str,
    currency: str | None = None,
    as_of_date: date | None = None,
    max_stale_days: int | None = None,
    minimum_curve_points: int = 3,
) -> pd.DataFrame:
    """
    Build local fair-value residuals directly from RepoLens exact curve points.

    This function reuses sovereign_snapshot.sovereign_curve_points so the RV
    engine consumes the same exact-ISIN, freshness and curve-readiness rules as
    the rest of RepoLens.
    """
    try:
        points = sovereign_curve_points(
            country=country,
            currency=currency,
            as_of_date=as_of_date,
            max_stale_days=max_stale_days,
            minimum_curve_points=minimum_curve_points,
            require_curve_ready=True,
        )
    except SovereignSnapshotValidationError as error:
        raise RelativeValueValidationError(
            str(
                error
            )
        ) from error

    return calculate_local_curve_fair_values(
        points
    )



def _history_identity(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["observation_date"]),
        str(row["isin"]).strip().upper(),
        str(row["model_version"]),
    )


def _fair_values_to_history_rows(fair_values: pd.DataFrame) -> list[dict[str, object]]:
    required = set(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS) - {"observation_date"}
    missing = sorted(required.difference(fair_values.columns))
    if missing:
        raise RelativeValueValidationError(
            "Fair-value frame is missing residual-history columns: "
            + ", ".join(missing) + "."
        )
    rows: list[dict[str, object]] = []
    available = fair_values.loc[
        fair_values["fair_value_available"].eq(True)
        & fair_values["residual_bp"].notna()
    ].copy()
    for _, row in available.iterrows():
        obs = row["observation_date"]
        if obs is None or pd.isna(obs):
            raise RelativeValueValidationError(
                "Available fair-value rows require observation_date before residual history can be persisted."
            )
        if isinstance(obs, pd.Timestamp):
            obs = obs.date()
        maturity = row["maturity_date"]
        if isinstance(maturity, pd.Timestamp):
            maturity = maturity.date()
        history_row = {
            "observation_date": obs.isoformat() if isinstance(obs, date) else str(obs),
            "country": row["country"], "currency": row["currency"],
            "isin": str(row["isin"]).strip().upper(), "display_name": row["display_name"],
            "maturity_date": maturity.isoformat() if isinstance(maturity, date) else str(maturity),
            "years_to_maturity": float(row["years_to_maturity"]),
            "actual_yield_percent": float(row["actual_yield_percent"]),
            "fair_yield_percent": float(row["fair_yield_percent"]),
            "residual_bp": float(row["residual_bp"]),
            "relative_value_label": row["relative_value_label"],
            "curve_location": row["curve_location"],
            "shorter_isin": row["shorter_isin"],
            "shorter_gap_years": float(row["shorter_gap_years"]),
            "longer_isin": row["longer_isin"],
            "longer_gap_years": float(row["longer_gap_years"]),
            "total_bracket_width_years": float(row["total_bracket_width_years"]),
            "interpolation_quality_score": float(row["interpolation_quality_score"]),
            "interpolation_quality": row["interpolation_quality"],
            "market_source": row["market_source"], "market_status": row["market_status"],
            "model_name": row["model_name"], "model_version": row["model_version"],
            "model_status": row["model_status"],
        }
        rows.append(history_row)
    return rows


def load_local_curve_residual_history(
    path: Path = DEFAULT_SOVEREIGN_RV_RESIDUAL_HISTORY_PATH,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS))
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS))
    missing = sorted(set(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS).difference(frame.columns))
    if missing:
        raise RelativeValueValidationError(
            "Residual-history CSV is missing columns: " + ", ".join(missing) + "."
        )
    frame = frame.loc[:, list(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS)].copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="raise").dt.date
    frame["maturity_date"] = pd.to_datetime(frame["maturity_date"], errors="raise").dt.date
    for col in (
        "years_to_maturity", "actual_yield_percent", "fair_yield_percent", "residual_bp",
        "shorter_gap_years", "longer_gap_years", "total_bracket_width_years",
        "interpolation_quality_score",
    ):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["isin"] = frame["isin"].astype(str).str.strip().str.upper()
    return frame.sort_values(["observation_date", "isin", "model_version"]).reset_index(drop=True)


def persist_local_curve_residual_history(
    fair_values: pd.DataFrame,
    *,
    path: Path = DEFAULT_SOVEREIGN_RV_RESIDUAL_HISTORY_PATH,
) -> pd.DataFrame:
    new_rows = _fair_values_to_history_rows(fair_values)
    existing = load_local_curve_residual_history(path)
    keyed: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in existing.iterrows():
        item = {c: row[c] for c in SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS}
        if isinstance(item["observation_date"], date):
            item["observation_date"] = item["observation_date"].isoformat()
        if isinstance(item["maturity_date"], date):
            item["maturity_date"] = item["maturity_date"].isoformat()
        keyed[_history_identity(item)] = item
    for item in new_rows:
        keyed[_history_identity(item)] = item
    ordered = sorted(keyed.values(), key=lambda r: (str(r["observation_date"]), str(r["isin"]), str(r["model_version"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOVEREIGN_RV_RESIDUAL_HISTORY_COLUMNS))
        writer.writeheader()
        writer.writerows(ordered)
    return load_local_curve_residual_history(path)


def add_residual_history_statistics(
    fair_values: pd.DataFrame,
    *,
    history: pd.DataFrame,
    lookback_observations: int = 60,
    minimum_history_observations: int = 20,
) -> pd.DataFrame:
    if lookback_observations <= 0:
        raise RelativeValueValidationError("lookback_observations must be positive.")
    if minimum_history_observations <= 0:
        raise RelativeValueValidationError("minimum_history_observations must be positive.")
    if minimum_history_observations > lookback_observations:
        raise RelativeValueValidationError(
            "minimum_history_observations must not exceed lookback_observations."
        )
    for label, frame, required in (
        ("Residual history", history, {"observation_date", "isin", "residual_bp", "model_version"}),
        ("Current fair values", fair_values, {"observation_date", "isin", "residual_bp", "model_version"}),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RelativeValueValidationError(
                f"{label} is missing required columns: " + ", ".join(missing) + "."
            )
    h = history.copy()
    h["observation_date"] = pd.to_datetime(h["observation_date"], errors="raise").dt.date
    h["residual_bp"] = pd.to_numeric(h["residual_bp"], errors="raise")
    h["isin"] = h["isin"].astype(str).str.strip().str.upper()
    result = fair_values.copy()
    records=[]
    for _, current in result.iterrows():
        current_date=current["observation_date"]
        if current_date is None or pd.isna(current_date) or pd.isna(current["residual_bp"]):
            records.append((0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,False))
            continue
        if isinstance(current_date,pd.Timestamp): current_date=current_date.date()
        matched=h.loc[
            h["isin"].eq(str(current["isin"]).strip().upper())
            & h["model_version"].astype(str).eq(str(current["model_version"]))
            & h["observation_date"].lt(current_date)
        ].sort_values("observation_date").tail(lookback_observations)
        vals=matched["residual_bp"].to_numpy(dtype=float)
        n=len(vals)
        if n==0:
            records.append((0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,False))
            continue
        cur=float(current["residual_bp"]); mean=float(np.mean(vals)); std=float(np.std(vals,ddof=0)); prev=float(vals[-1])
        pct=float(np.mean(vals<=cur)*100.0); ready=n>=minimum_history_observations
        z=(cur-mean)/std if ready and std>0.0 else np.nan
        records.append((n,mean,std,float(np.min(vals)),float(np.max(vals)),pct,z,prev,cur-prev,ready))
    cols=("history_observation_count","historical_mean_residual_bp","historical_std_residual_bp","historical_min_residual_bp","historical_max_residual_bp","historical_percentile","z_score","previous_residual_bp","change_vs_previous_residual_bp","history_ready")
    for i,col in enumerate(cols): result[col]=[r[i] for r in records]
    result["history_lookback_observations"]=lookback_observations
    result["minimum_history_observations"]=minimum_history_observations
    return result


def build_country_local_curve_relative_value_with_history(
    *, country: str, currency: str | None = None, as_of_date: date | None = None,
    max_stale_days: int | None = None, minimum_curve_points: int = 3,
    history_path: Path = DEFAULT_SOVEREIGN_RV_RESIDUAL_HISTORY_PATH,
    lookback_observations: int = 60, minimum_history_observations: int = 20,
    persist_current: bool = False,
) -> pd.DataFrame:
    current = build_country_local_curve_relative_value(
        country=country, currency=currency, as_of_date=as_of_date,
        max_stale_days=max_stale_days, minimum_curve_points=minimum_curve_points,
    )
    history = (
        persist_local_curve_residual_history(current, path=history_path)
        if persist_current else load_local_curve_residual_history(history_path)
    )
    return add_residual_history_statistics(
        current, history=history, lookback_observations=lookback_observations,
        minimum_history_observations=minimum_history_observations,
    )


def rank_local_curve_opportunities(
    fair_values: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Rank available local-curve residuals by absolute basis-point deviation.

    Interpolation geometry and quality are carried through as descriptive
    context but do not change the rank. This is a descriptive ranking, not a
    trade recommendation. Historical z-scores, repo financing and liquidity
    are separate later stages of the opportunity stack.
    """
    if top_n <= 0:
        raise RelativeValueValidationError(
            "top_n must be positive."
        )

    required = {
        "fair_value_available",
        "residual_bp",
        "absolute_residual_bp",
    }

    missing = sorted(
        required.difference(
            fair_values.columns
        )
    )

    if missing:
        raise RelativeValueValidationError(
            "Fair-value frame is missing required columns: "
            + ", ".join(
                missing
            )
            + "."
        )

    ranked = fair_values.loc[
        fair_values[
            "fair_value_available"
        ].eq(
            True
        )
        & fair_values[
            "residual_bp"
        ].notna()
    ].copy()

    ranked = ranked.sort_values(
        [
            "absolute_residual_bp",
            "years_to_maturity",
            "isin",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).head(
        top_n
    )

    ranked.insert(
        0,
        "rank",
        range(
            1,
            len(
                ranked
            )
            + 1,
        ),
    )

    return ranked.reset_index(
        drop=True
    )



def rank_historical_local_curve_opportunities(
    fair_values_with_history: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    if top_n <= 0:
        raise RelativeValueValidationError("top_n must be positive.")
    required={"fair_value_available","history_ready","z_score","residual_bp","absolute_residual_bp"}
    missing=sorted(required.difference(fair_values_with_history.columns))
    if missing:
        raise RelativeValueValidationError(
            "Historical fair-value frame is missing required columns: " + ", ".join(missing) + "."
        )
    ranked=fair_values_with_history.loc[
        fair_values_with_history["fair_value_available"].eq(True)
        & fair_values_with_history["history_ready"].eq(True)
        & fair_values_with_history["z_score"].notna()
    ].copy()
    ranked["absolute_z_score"]=ranked["z_score"].abs()
    ranked=ranked.sort_values(
        ["absolute_z_score","absolute_residual_bp","years_to_maturity","isin"],
        ascending=[False,False,True,True],
    ).head(top_n)
    ranked.insert(0,"historical_rank",range(1,len(ranked)+1))
    return ranked.reset_index(drop=True)
