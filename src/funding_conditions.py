from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


DEFAULT_ROLLING_WINDOW: Final[int] = 20
DEFAULT_STRESS_Z_THRESHOLD: Final[float] = 2.0
DEFAULT_STALE_BUSINESS_DAYS: Final[int] = 3


class FundingConditionsError(RuntimeError):
    """
    Base exception for funding-condition analytics.
    """


class FundingDataValidationError(FundingConditionsError):
    """
    Raised when the input dataset is missing or invalid.
    """


@dataclass(frozen=True)
class FundingConditionsSummary:
    """
    Store the latest validated funding-condition metrics.
    """

    latest_observation_date: pd.Timestamp
    latest_estr_rate: float
    latest_daily_change_bp: float
    rolling_change_volatility_bp: float
    latest_change_z_score: float
    funding_regime: str
    is_month_end: bool
    is_quarter_end: bool
    business_days_stale: int
    freshness_status: str


def validate_estr_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalise an ECB €STR observation dataset.
    """
    required_columns = {
        "observation_date",
        "value",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise FundingDataValidationError(
            "€STR data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    validated = data.copy()

    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"],
        errors="coerce",
    )

    validated["value"] = pd.to_numeric(
        validated["value"],
        errors="coerce",
    )

    validated = validated.dropna(
        subset=[
            "observation_date",
            "value",
        ]
    )

    if validated.empty:
        raise FundingDataValidationError(
            "€STR data contains no valid observations."
        )

    if (
        validated["value"]
        .lt(-10.0)
        .any()
    ):
        raise FundingDataValidationError(
            "€STR data contains implausibly low values."
        )

    if (
        validated["value"]
        .gt(25.0)
        .any()
    ):
        raise FundingDataValidationError(
            "€STR data contains implausibly high values."
        )

    validated = validated.sort_values(
        "observation_date"
    )

    validated = validated.drop_duplicates(
        subset="observation_date",
        keep="last",
    )

    validated = validated.reset_index(
        drop=True
    )

    return validated


def is_month_end_observation(
    dates: pd.Series,
) -> pd.Series:
    """
    Flag the final available observation in each calendar month.
    """
    month_period = dates.dt.to_period(
        "M"
    )

    next_month_period = month_period.shift(
        -1
    )

    return (
        month_period
        != next_month_period
    )


def is_quarter_end_observation(
    dates: pd.Series,
) -> pd.Series:
    """
    Flag the final available observation in each calendar quarter.
    """
    quarter_period = dates.dt.to_period(
        "Q"
    )

    next_quarter_period = quarter_period.shift(
        -1
    )

    return (
        quarter_period
        != next_quarter_period
    )


def calculate_business_days_stale(
    latest_observation_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> int:
    """
    Count business days between the latest observation and an as-of date.
    """
    latest_date = pd.Timestamp(
        latest_observation_date
    ).normalize()

    current_date = pd.Timestamp(
        as_of_date
    ).normalize()

    if latest_date > current_date:
        raise ValueError(
            "latest_observation_date must not be after as_of_date."
        )

    business_dates = pd.bdate_range(
        start=latest_date,
        end=current_date,
    )

    return max(
        len(business_dates) - 1,
        0,
    )


def classify_freshness(
    business_days_stale: int,
    stale_business_days: int = DEFAULT_STALE_BUSINESS_DAYS,
) -> str:
    """
    Classify data freshness using business-day age.
    """
    if business_days_stale < 0:
        raise ValueError(
            "business_days_stale must not be negative."
        )

    if stale_business_days <= 0:
        raise ValueError(
            "stale_business_days must be positive."
        )

    if business_days_stale == 0:
        return "Current"

    if business_days_stale < stale_business_days:
        return "Delayed"

    return "Stale"


def classify_funding_regime(
    change_z_score: float,
    stress_z_threshold: float = DEFAULT_STRESS_Z_THRESHOLD,
) -> str:
    """
    Classify the latest €STR move using its rolling z-score.
    """
    if stress_z_threshold <= 0:
        raise ValueError(
            "stress_z_threshold must be positive."
        )

    if np.isnan(
        change_z_score
    ):
        return "Insufficient history"

    if change_z_score >= stress_z_threshold:
        return "Sharp tightening"

    if change_z_score <= -stress_z_threshold:
        return "Sharp easing"

    if change_z_score >= 1.0:
        return "Mild tightening"

    if change_z_score <= -1.0:
        return "Mild easing"

    return "Normal"


def build_funding_conditions(
    data: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stress_z_threshold: float = DEFAULT_STRESS_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Create the complete €STR funding-condition research dataset.
    """
    if rolling_window < 5:
        raise ValueError(
            "rolling_window must be at least 5 observations."
        )

    validated = validate_estr_data(
        data
    )

    conditions = validated[
        [
            "observation_date",
            "value",
        ]
    ].copy()

    conditions = conditions.rename(
        columns={
            "value": "estr_rate",
        }
    )

    conditions["daily_change_bp"] = (
        conditions["estr_rate"]
        .diff()
        * 100.0
    )

    rolling_change_mean = (
        conditions["daily_change_bp"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .mean()
    )

    conditions["rolling_change_volatility_bp"] = (
        conditions["daily_change_bp"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .std(ddof=1)
    )

    volatility = (
        conditions[
            "rolling_change_volatility_bp"
        ]
        .replace(
            0.0,
            np.nan,
        )
    )

    conditions["change_z_score"] = (
        conditions["daily_change_bp"]
        - rolling_change_mean
    ) / volatility

    conditions["absolute_change_bp"] = (
        conditions["daily_change_bp"]
        .abs()
    )

    conditions["is_month_end"] = (
        is_month_end_observation(
            conditions["observation_date"]
        )
    )

    conditions["is_quarter_end"] = (
        is_quarter_end_observation(
            conditions["observation_date"]
        )
    )

    conditions["funding_regime"] = (
        conditions["change_z_score"]
        .apply(
            lambda value: classify_funding_regime(
                change_z_score=value,
                stress_z_threshold=stress_z_threshold,
            )
        )
    )

    conditions["is_abnormal_move"] = (
        conditions["change_z_score"]
        .abs()
        .ge(
            stress_z_threshold
        )
        .fillna(False)
    )

    return conditions


def summarise_funding_conditions(
    conditions: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
    stale_business_days: int = DEFAULT_STALE_BUSINESS_DAYS,
) -> FundingConditionsSummary:
    """
    Create a compact latest-state summary.
    """
    required_columns = {
        "observation_date",
        "estr_rate",
        "daily_change_bp",
        "rolling_change_volatility_bp",
        "change_z_score",
        "funding_regime",
        "is_month_end",
        "is_quarter_end",
    }

    missing_columns = (
        required_columns
        - set(conditions.columns)
    )

    if missing_columns:
        raise FundingDataValidationError(
            "Funding conditions are missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if conditions.empty:
        raise FundingDataValidationError(
            "Funding conditions dataset is empty."
        )

    latest = (
        conditions
        .sort_values(
            "observation_date"
        )
        .iloc[-1]
    )

    latest_date = pd.Timestamp(
        latest["observation_date"]
    )

    effective_as_of_date = (
        pd.Timestamp.now(
            tz="UTC"
        ).tz_localize(None)
        if as_of_date is None
        else pd.Timestamp(
            as_of_date
        )
    )

    business_days_stale = (
        calculate_business_days_stale(
            latest_observation_date=latest_date,
            as_of_date=effective_as_of_date,
        )
    )

    return FundingConditionsSummary(
        latest_observation_date=latest_date,
        latest_estr_rate=float(
            latest["estr_rate"]
        ),
        latest_daily_change_bp=float(
            latest["daily_change_bp"]
        )
        if pd.notna(
            latest["daily_change_bp"]
        )
        else float("nan"),
        rolling_change_volatility_bp=float(
            latest[
                "rolling_change_volatility_bp"
            ]
        )
        if pd.notna(
            latest[
                "rolling_change_volatility_bp"
            ]
        )
        else float("nan"),
        latest_change_z_score=float(
            latest["change_z_score"]
        )
        if pd.notna(
            latest["change_z_score"]
        )
        else float("nan"),
        funding_regime=str(
            latest["funding_regime"]
        ),
        is_month_end=bool(
            latest["is_month_end"]
        ),
        is_quarter_end=bool(
            latest["is_quarter_end"]
        ),
        business_days_stale=business_days_stale,
        freshness_status=classify_freshness(
            business_days_stale=business_days_stale,
            stale_business_days=stale_business_days,
        ),
    )


def load_estr_csv(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load a locally saved €STR CSV file.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"€STR input file does not exist: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "input_path must use the .csv extension."
        )

    return pd.read_csv(
        input_path
    )


def save_funding_conditions(
    conditions: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save the completed funding-condition dataset.
    """
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "output_path must use the .csv extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions.to_csv(
        output_path,
        index=False,
    )

    return output_path