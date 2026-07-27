from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


DEFAULT_ROLLING_WINDOW: Final[int] = 252
DEFAULT_MINIMUM_HISTORY: Final[int] = 60
DEFAULT_STALE_BUSINESS_DAYS: Final[int] = 3

VOLUME_WEIGHT: Final[float] = 0.35
ACTIVE_BANKS_WEIGHT: Final[float] = 0.25
TRANSACTION_COUNT_WEIGHT: Final[float] = 0.25
DISPERSION_WEIGHT: Final[float] = 0.15

ZERO_VOLATILITY_FALLBACK_Z: Final[float] = 3.0


class MarketQualityError(RuntimeError):
    """
    Base exception for €STR market-quality analytics.
    """


class MarketQualityValidationError(MarketQualityError):
    """
    Raised when one or more market-quality inputs are invalid.
    """


@dataclass(frozen=True)
class MarketQualitySummary:
    """
    Store the latest RepoLens €STR market-quality assessment.
    """

    latest_observation_date: pd.Timestamp
    latest_estr_rate: float
    latest_total_volume_eur_mn: float
    latest_active_banks: float
    latest_transaction_count: float
    latest_rate_dispersion_bp: float
    latest_quality_score: float
    quality_regime: str
    is_market_quality_alert: bool
    available_component_count: int
    business_days_stale: int
    freshness_status: str


def validate_ecb_series(
    data: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate and normalise one ECB observation dataset.
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
        raise MarketQualityValidationError(
            f"{dataset_name} is missing required columns: "
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
        raise MarketQualityValidationError(
            f"{dataset_name} contains no valid observations."
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

    return validated[
        [
            "observation_date",
            "value",
        ]
    ]


def combine_market_quality_inputs(
    estr_rate_data: pd.DataFrame,
    total_volume_data: pd.DataFrame,
    percentile_25_data: pd.DataFrame,
    percentile_75_data: pd.DataFrame,
    active_banks_data: pd.DataFrame,
    transaction_count_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine the six official ECB series on common observation dates.
    """
    series_inputs = [
        (
            estr_rate_data,
            "€STR rate",
            "estr_rate",
        ),
        (
            total_volume_data,
            "€STR total volume",
            "total_volume_eur_mn",
        ),
        (
            percentile_25_data,
            "€STR 25th percentile rate",
            "rate_25th_percentile",
        ),
        (
            percentile_75_data,
            "€STR 75th percentile rate",
            "rate_75th_percentile",
        ),
        (
            active_banks_data,
            "€STR active banks",
            "active_banks",
        ),
        (
            transaction_count_data,
            "€STR transaction count",
            "transaction_count",
        ),
    ]

    validated_series: list[pd.DataFrame] = []

    for data, dataset_name, output_column in series_inputs:
        validated = validate_ecb_series(
            data=data,
            dataset_name=dataset_name,
        ).rename(
            columns={
                "value": output_column,
            }
        )

        validated_series.append(
            validated
        )

    combined = validated_series[0]

    for next_series in validated_series[1:]:
        combined = combined.merge(
            next_series,
            on="observation_date",
            how="inner",
            validate="one_to_one",
        )

    if combined.empty:
        raise MarketQualityValidationError(
            "The ECB market-quality series have no common "
            "observation dates."
        )

    if (
        combined["total_volume_eur_mn"]
        .le(0.0)
        .any()
    ):
        raise MarketQualityValidationError(
            "Total €STR volume must be positive."
        )

    if (
        combined["active_banks"]
        .le(0.0)
        .any()
    ):
        raise MarketQualityValidationError(
            "The number of active banks must be positive."
        )

    if (
        combined["transaction_count"]
        .le(0.0)
        .any()
    ):
        raise MarketQualityValidationError(
            "The number of transactions must be positive."
        )

    invalid_percentiles = (
        combined["rate_75th_percentile"]
        < combined["rate_25th_percentile"]
    )

    if invalid_percentiles.any():
        raise MarketQualityValidationError(
            "The 75th percentile rate must not be below "
            "the 25th percentile rate."
        )

    return (
        combined
        .sort_values(
            "observation_date"
        )
        .reset_index(
            drop=True
        )
    )


def calculate_point_in_time_z_score(
    values: pd.Series,
    rolling_window: int,
    minimum_history: int,
) -> pd.Series:
    """
    Calculate a point-in-time z-score without look-ahead bias.

    The current observation is excluded from its own historical mean
    and volatility estimate.

    When the historical standard deviation is zero:
    - an unchanged value receives a neutral score of zero;
    - an upward break receives +3;
    - a downward break receives -3.
    """
    if rolling_window < 2:
        raise ValueError(
            "rolling_window must be at least 2 observations."
        )

    if minimum_history < 2:
        raise ValueError(
            "minimum_history must be at least 2 observations."
        )

    if minimum_history > rolling_window:
        raise ValueError(
            "minimum_history must not exceed rolling_window."
        )

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    historical_values = numeric_values.shift(
        1
    )

    rolling_mean = historical_values.rolling(
        window=rolling_window,
        min_periods=minimum_history,
    ).mean()

    rolling_volatility = historical_values.rolling(
        window=rolling_window,
        min_periods=minimum_history,
    ).std(
        ddof=1
    )

    difference_from_mean = (
        numeric_values
        - rolling_mean
    )

    z_scores = pd.Series(
        np.nan,
        index=numeric_values.index,
        dtype=float,
    )

    positive_volatility = (
        rolling_volatility
        .gt(0.0)
        & rolling_volatility.notna()
    )

    z_scores.loc[
        positive_volatility
    ] = (
        difference_from_mean.loc[
            positive_volatility
        ]
        / rolling_volatility.loc[
            positive_volatility
        ]
    )

    zero_volatility = (
        rolling_volatility
        .eq(0.0)
        & rolling_mean.notna()
        & numeric_values.notna()
    )

    unchanged_from_constant_history = (
        zero_volatility
        & np.isclose(
            difference_from_mean,
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    )

    z_scores.loc[
        unchanged_from_constant_history
    ] = 0.0

    positive_break_from_constant_history = (
        zero_volatility
        & difference_from_mean.gt(0.0)
    )

    z_scores.loc[
        positive_break_from_constant_history
    ] = ZERO_VOLATILITY_FALLBACK_Z

    negative_break_from_constant_history = (
        zero_volatility
        & difference_from_mean.lt(0.0)
    )

    z_scores.loc[
        negative_break_from_constant_history
    ] = -ZERO_VOLATILITY_FALLBACK_Z

    return z_scores


def classify_market_quality(
    quality_score: float,
) -> str:
    """
    Convert the derived score into a human-readable regime.
    """
    if pd.isna(
        quality_score
    ):
        return "Insufficient history"

    if quality_score >= 65.0:
        return "Strong market quality"

    if quality_score <= 35.0:
        return "Weak market quality"

    return "Normal market quality"


def build_estr_market_quality(
    estr_rate_data: pd.DataFrame,
    total_volume_data: pd.DataFrame,
    percentile_25_data: pd.DataFrame,
    percentile_75_data: pd.DataFrame,
    active_banks_data: pd.DataFrame,
    transaction_count_data: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    minimum_history: int = DEFAULT_MINIMUM_HISTORY,
) -> pd.DataFrame:
    """
    Build the complete RepoLens €STR market-quality dataset.

    The resulting score is a derived RepoLens research measure.
    It is not an official ECB index.
    """
    if rolling_window < 2:
        raise ValueError(
            "rolling_window must be at least 2 observations."
        )

    if minimum_history < 2:
        raise ValueError(
            "minimum_history must be at least 2 observations."
        )

    if minimum_history > rolling_window:
        raise ValueError(
            "minimum_history must not exceed rolling_window."
        )

    market_quality = combine_market_quality_inputs(
        estr_rate_data=estr_rate_data,
        total_volume_data=total_volume_data,
        percentile_25_data=percentile_25_data,
        percentile_75_data=percentile_75_data,
        active_banks_data=active_banks_data,
        transaction_count_data=transaction_count_data,
    )

    market_quality["rate_dispersion_bp"] = (
        (
            market_quality[
                "rate_75th_percentile"
            ]
            - market_quality[
                "rate_25th_percentile"
            ]
        )
        * 100.0
    )

    market_quality["average_transaction_size_eur_mn"] = (
        market_quality[
            "total_volume_eur_mn"
        ]
        / market_quality[
            "transaction_count"
        ]
    )

    market_quality["volume_z_score"] = (
        calculate_point_in_time_z_score(
            values=market_quality[
                "total_volume_eur_mn"
            ],
            rolling_window=rolling_window,
            minimum_history=minimum_history,
        )
    )

    market_quality["active_banks_z_score"] = (
        calculate_point_in_time_z_score(
            values=market_quality[
                "active_banks"
            ],
            rolling_window=rolling_window,
            minimum_history=minimum_history,
        )
    )

    market_quality["transaction_count_z_score"] = (
        calculate_point_in_time_z_score(
            values=market_quality[
                "transaction_count"
            ],
            rolling_window=rolling_window,
            minimum_history=minimum_history,
        )
    )

    market_quality["dispersion_z_score"] = (
        calculate_point_in_time_z_score(
            values=market_quality[
                "rate_dispersion_bp"
            ],
            rolling_window=rolling_window,
            minimum_history=minimum_history,
        )
    )

    component_columns = [
        "volume_z_score",
        "active_banks_z_score",
        "transaction_count_z_score",
        "dispersion_z_score",
    ]

    market_quality["available_component_count"] = (
        market_quality[
            component_columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    clipped_volume = (
        market_quality[
            "volume_z_score"
        ]
        .clip(
            lower=-3.0,
            upper=3.0,
        )
    )

    clipped_banks = (
        market_quality[
            "active_banks_z_score"
        ]
        .clip(
            lower=-3.0,
            upper=3.0,
        )
    )

    clipped_transactions = (
        market_quality[
            "transaction_count_z_score"
        ]
        .clip(
            lower=-3.0,
            upper=3.0,
        )
    )

    clipped_dispersion = (
        market_quality[
            "dispersion_z_score"
        ]
        .clip(
            lower=-3.0,
            upper=3.0,
        )
    )

    composite_signal = (
        clipped_volume
        * VOLUME_WEIGHT
        + clipped_banks
        * ACTIVE_BANKS_WEIGHT
        + clipped_transactions
        * TRANSACTION_COUNT_WEIGHT
        - clipped_dispersion
        * DISPERSION_WEIGHT
    )

    market_quality["market_quality_score"] = (
        50.0
        + composite_signal
        * 10.0
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    insufficient_components = (
        market_quality[
            "available_component_count"
        ]
        < len(
            component_columns
        )
    )

    market_quality.loc[
        insufficient_components,
        "market_quality_score",
    ] = np.nan

    market_quality["quality_regime"] = (
        market_quality[
            "market_quality_score"
        ]
        .apply(
            classify_market_quality
        )
    )

    market_quality["is_market_quality_alert"] = (
        (
            market_quality[
                "market_quality_score"
            ]
            .le(35.0)
        )
        | (
            market_quality[
                "dispersion_z_score"
            ]
            .ge(2.0)
        )
        | (
            market_quality[
                "volume_z_score"
            ]
            .le(-2.0)
        )
        | (
            market_quality[
                "active_banks_z_score"
            ]
            .le(-2.0)
        )
        | (
            market_quality[
                "transaction_count_z_score"
            ]
            .le(-2.0)
        )
    ).fillna(
        False
    )

    market_quality["data_classification"] = (
        "Official ECB inputs; RepoLens derived score"
    )

    return market_quality


def calculate_business_days_stale(
    latest_observation_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> int:
    """
    Count business days between the latest observation and as-of date.
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


def summarise_estr_market_quality(
    market_quality: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
) -> MarketQualitySummary:
    """
    Summarise the latest RepoLens market-quality assessment.
    """
    required_columns = {
        "observation_date",
        "estr_rate",
        "total_volume_eur_mn",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "market_quality_score",
        "quality_regime",
        "is_market_quality_alert",
        "available_component_count",
    }

    missing_columns = (
        required_columns
        - set(market_quality.columns)
    )

    if missing_columns:
        raise MarketQualityValidationError(
            "Market-quality data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if market_quality.empty:
        raise MarketQualityValidationError(
            "Market-quality dataset is empty."
        )

    latest = (
        market_quality
        .sort_values(
            "observation_date"
        )
        .iloc[-1]
    )

    latest_date = pd.Timestamp(
        latest[
            "observation_date"
        ]
    )

    effective_as_of_date = (
        pd.Timestamp.now(
            tz="UTC"
        ).tz_localize(
            None
        )
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

    latest_score = (
        float(
            latest[
                "market_quality_score"
            ]
        )
        if pd.notna(
            latest[
                "market_quality_score"
            ]
        )
        else float("nan")
    )

    return MarketQualitySummary(
        latest_observation_date=latest_date,
        latest_estr_rate=float(
            latest[
                "estr_rate"
            ]
        ),
        latest_total_volume_eur_mn=float(
            latest[
                "total_volume_eur_mn"
            ]
        ),
        latest_active_banks=float(
            latest[
                "active_banks"
            ]
        ),
        latest_transaction_count=float(
            latest[
                "transaction_count"
            ]
        ),
        latest_rate_dispersion_bp=float(
            latest[
                "rate_dispersion_bp"
            ]
        ),
        latest_quality_score=latest_score,
        quality_regime=str(
            latest[
                "quality_regime"
            ]
        ),
        is_market_quality_alert=bool(
            latest[
                "is_market_quality_alert"
            ]
        ),
        available_component_count=int(
            latest[
                "available_component_count"
            ]
        ),
        business_days_stale=business_days_stale,
        freshness_status=classify_freshness(
            business_days_stale=business_days_stale,
        ),
    )


def load_market_quality_csv(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load one locally stored ECB market-quality CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            "Market-quality input file does not exist: "
            f"{input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "input_path must use the .csv extension."
        )

    return pd.read_csv(
        input_path
    )


def save_estr_market_quality(
    market_quality: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save the processed RepoLens market-quality dataset.
    """
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "output_path must use the .csv extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_quality.to_csv(
        output_path,
        index=False,
    )

    return output_path