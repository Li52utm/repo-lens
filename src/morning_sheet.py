from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class MorningSheetError(RuntimeError):
    """
    Base exception for RepoLens Morning Sheet analytics.
    """


class MorningSheetValidationError(MorningSheetError):
    """
    Raised when a processed input dataset is invalid.
    """


@dataclass(frozen=True)
class MorningSheetSummary:
    """
    Store the latest consolidated RepoLens funding assessment.
    """

    observation_date: pd.Timestamp
    estr_rate: float
    daily_change_bp: float
    funding_regime: str
    policy_spread_bp: float
    policy_spread_z_score: float
    transmission_regime: str
    deposit_facility_rate: float
    policy_rate_changed_today: bool
    market_quality_score: float
    market_quality_change: float
    quality_regime: str
    total_volume_eur_mn: float
    volume_change_pct: float
    active_banks: float
    transaction_count: float
    rate_dispersion_bp: float
    alert_count: int
    overall_status: str


def validate_dataset(
    data: pd.DataFrame,
    dataset_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """
    Validate and normalise one processed RepoLens dataset.
    """
    missing_columns = required_columns - set(
        data.columns
    )

    if missing_columns:
        raise MorningSheetValidationError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    validated = data.copy()

    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"],
        errors="coerce",
    )

    validated = validated.dropna(
        subset=[
            "observation_date",
        ]
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

    if validated.empty:
        raise MorningSheetValidationError(
            f"{dataset_name} contains no valid observations."
        )

    return validated


def normalise_boolean_series(
    values: pd.Series,
) -> pd.Series:
    """
    Convert common CSV boolean representations into real booleans.
    """
    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.fillna(
            False
        ).astype(
            bool
        )

    normalised = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalised.isin(
        {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def prepare_funding_conditions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the processed €STR funding-conditions dataset.
    """
    required_columns = {
        "observation_date",
        "estr_rate",
        "daily_change_bp",
        "rolling_change_volatility_bp",
        "change_z_score",
        "funding_regime",
        "is_abnormal_move",
        "is_month_end",
        "is_quarter_end",
    }

    prepared = validate_dataset(
        data=data,
        dataset_name="Funding conditions",
        required_columns=required_columns,
    )

    numeric_columns = [
        "estr_rate",
        "daily_change_bp",
        "rolling_change_volatility_bp",
        "change_z_score",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    boolean_columns = [
        "is_abnormal_move",
        "is_month_end",
        "is_quarter_end",
    ]

    for column in boolean_columns:
        prepared[column] = normalise_boolean_series(
            prepared[column]
        )

    return prepared[
        [
            "observation_date",
            "estr_rate",
            "daily_change_bp",
            "rolling_change_volatility_bp",
            "change_z_score",
            "funding_regime",
            "is_abnormal_move",
            "is_month_end",
            "is_quarter_end",
        ]
    ]


def prepare_policy_spread(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare policy-spread data and correctly derive policy-change dates.

    The stored is_policy_change_day field is deliberately not trusted
    because an as-of merge can carry a True value into later dates.
    """
    required_columns = {
        "observation_date",
        "deposit_facility_rate",
        "estr_policy_spread_bp",
        "spread_z_score",
        "transmission_regime",
        "is_unusual_spread",
    }

    prepared = validate_dataset(
        data=data,
        dataset_name="Policy spread",
        required_columns=required_columns,
    )

    numeric_columns = [
        "deposit_facility_rate",
        "estr_policy_spread_bp",
        "spread_z_score",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["is_unusual_spread"] = (
        normalise_boolean_series(
            prepared["is_unusual_spread"]
        )
    )

    previous_policy_rate = (
        prepared["deposit_facility_rate"]
        .shift(1)
    )

    prepared["policy_rate_changed_today"] = (
        prepared["deposit_facility_rate"]
        .ne(
            previous_policy_rate
        )
        & previous_policy_rate.notna()
    )

    return prepared[
        [
            "observation_date",
            "deposit_facility_rate",
            "estr_policy_spread_bp",
            "spread_z_score",
            "transmission_regime",
            "is_unusual_spread",
            "policy_rate_changed_today",
        ]
    ]


def prepare_market_quality(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the processed €STR market-quality dataset.
    """
    required_columns = {
        "observation_date",
        "total_volume_eur_mn",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "market_quality_score",
        "quality_regime",
        "is_market_quality_alert",
    }

    prepared = validate_dataset(
        data=data,
        dataset_name="Market quality",
        required_columns=required_columns,
    )

    numeric_columns = [
        "total_volume_eur_mn",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "market_quality_score",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["is_market_quality_alert"] = (
        normalise_boolean_series(
            prepared["is_market_quality_alert"]
        )
    )

    prepared["market_quality_change"] = (
        prepared["market_quality_score"]
        .diff()
    )

    prepared["volume_change_pct"] = (
        prepared["total_volume_eur_mn"]
        .pct_change(
            fill_method=None
        )
        * 100.0
    )

    prepared["active_banks_change"] = (
        prepared["active_banks"]
        .diff()
    )

    prepared["transaction_count_change"] = (
        prepared["transaction_count"]
        .diff()
    )

    prepared["dispersion_change_bp"] = (
        prepared["rate_dispersion_bp"]
        .diff()
    )

    return prepared[
        [
            "observation_date",
            "total_volume_eur_mn",
            "volume_change_pct",
            "active_banks",
            "active_banks_change",
            "transaction_count",
            "transaction_count_change",
            "rate_dispersion_bp",
            "dispersion_change_bp",
            "market_quality_score",
            "market_quality_change",
            "quality_regime",
            "is_market_quality_alert",
        ]
    ]


def classify_overall_status(
    alert_count: int,
    is_month_end: bool,
    is_quarter_end: bool,
) -> str:
    """
    Convert the consolidated alert count into a desk-friendly status.
    """
    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    if alert_count >= 2:
        return "High attention"

    if alert_count == 1:
        return "Monitor"

    if is_quarter_end:
        return "Quarter-end watch"

    if is_month_end:
        return "Month-end watch"

    return "Normal"


def build_morning_sheet(
    funding_conditions: pd.DataFrame,
    policy_spread: pd.DataFrame,
    market_quality: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the consolidated RepoLens euro funding Morning Sheet.
    """
    funding = prepare_funding_conditions(
        data=funding_conditions
    )

    policy = prepare_policy_spread(
        data=policy_spread
    )

    quality = prepare_market_quality(
        data=market_quality
    )

    morning_sheet = funding.merge(
        policy,
        on="observation_date",
        how="inner",
        validate="one_to_one",
    )

    morning_sheet = morning_sheet.merge(
        quality,
        on="observation_date",
        how="inner",
        validate="one_to_one",
    )

    if morning_sheet.empty:
        raise MorningSheetValidationError(
            "The processed RepoLens datasets have no common "
            "observation dates."
        )

    morning_sheet["alert_count"] = (
        morning_sheet[
            [
                "is_abnormal_move",
                "is_unusual_spread",
                "is_market_quality_alert",
            ]
        ]
        .astype(int)
        .sum(
            axis=1
        )
    )

    morning_sheet["overall_status"] = [
        classify_overall_status(
            alert_count=int(alert_count),
            is_month_end=bool(is_month_end),
            is_quarter_end=bool(is_quarter_end),
        )
        for (
            alert_count,
            is_month_end,
            is_quarter_end,
        ) in zip(
            morning_sheet["alert_count"],
            morning_sheet["is_month_end"],
            morning_sheet["is_quarter_end"],
            strict=True,
        )
    ]

    morning_sheet["has_any_alert"] = (
        morning_sheet["alert_count"]
        .gt(0)
    )

    morning_sheet["data_classification"] = (
        "Official ECB inputs; RepoLens derived analytics"
    )

    output_columns = [
        "observation_date",
        "estr_rate",
        "daily_change_bp",
        "rolling_change_volatility_bp",
        "change_z_score",
        "funding_regime",
        "is_abnormal_move",
        "is_month_end",
        "is_quarter_end",
        "deposit_facility_rate",
        "policy_rate_changed_today",
        "estr_policy_spread_bp",
        "spread_z_score",
        "transmission_regime",
        "is_unusual_spread",
        "total_volume_eur_mn",
        "volume_change_pct",
        "active_banks",
        "active_banks_change",
        "transaction_count",
        "transaction_count_change",
        "rate_dispersion_bp",
        "dispersion_change_bp",
        "market_quality_score",
        "market_quality_change",
        "quality_regime",
        "is_market_quality_alert",
        "alert_count",
        "has_any_alert",
        "overall_status",
        "data_classification",
    ]

    return (
        morning_sheet[
            output_columns
        ]
        .sort_values(
            "observation_date"
        )
        .reset_index(
            drop=True
        )
    )


def summarise_morning_sheet(
    morning_sheet: pd.DataFrame,
) -> MorningSheetSummary:
    """
    Summarise the latest consolidated Morning Sheet observation.
    """
    required_columns = {
        "observation_date",
        "estr_rate",
        "daily_change_bp",
        "funding_regime",
        "estr_policy_spread_bp",
        "spread_z_score",
        "transmission_regime",
        "deposit_facility_rate",
        "policy_rate_changed_today",
        "market_quality_score",
        "market_quality_change",
        "quality_regime",
        "total_volume_eur_mn",
        "volume_change_pct",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "alert_count",
        "overall_status",
    }

    missing_columns = required_columns - set(
        morning_sheet.columns
    )

    if missing_columns:
        raise MorningSheetValidationError(
            "Morning Sheet is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if morning_sheet.empty:
        raise MorningSheetValidationError(
            "Morning Sheet dataset is empty."
        )

    latest = (
        morning_sheet
        .sort_values(
            "observation_date"
        )
        .iloc[-1]
    )

    def optional_float(
        value: object,
    ) -> float:
        if pd.isna(
            value
        ):
            return float("nan")

        return float(
            value
        )

    return MorningSheetSummary(
        observation_date=pd.Timestamp(
            latest["observation_date"]
        ),
        estr_rate=float(
            latest["estr_rate"]
        ),
        daily_change_bp=optional_float(
            latest["daily_change_bp"]
        ),
        funding_regime=str(
            latest["funding_regime"]
        ),
        policy_spread_bp=float(
            latest["estr_policy_spread_bp"]
        ),
        policy_spread_z_score=optional_float(
            latest["spread_z_score"]
        ),
        transmission_regime=str(
            latest["transmission_regime"]
        ),
        deposit_facility_rate=float(
            latest["deposit_facility_rate"]
        ),
        policy_rate_changed_today=bool(
            latest["policy_rate_changed_today"]
        ),
        market_quality_score=optional_float(
            latest["market_quality_score"]
        ),
        market_quality_change=optional_float(
            latest["market_quality_change"]
        ),
        quality_regime=str(
            latest["quality_regime"]
        ),
        total_volume_eur_mn=float(
            latest["total_volume_eur_mn"]
        ),
        volume_change_pct=optional_float(
            latest["volume_change_pct"]
        ),
        active_banks=float(
            latest["active_banks"]
        ),
        transaction_count=float(
            latest["transaction_count"]
        ),
        rate_dispersion_bp=float(
            latest["rate_dispersion_bp"]
        ),
        alert_count=int(
            latest["alert_count"]
        ),
        overall_status=str(
            latest["overall_status"]
        ),
    )


def load_processed_csv(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load one processed RepoLens CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed input file does not exist: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "input_path must use the .csv extension."
        )

    return pd.read_csv(
        input_path
    )


def save_morning_sheet(
    morning_sheet: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save the consolidated RepoLens Morning Sheet.
    """
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "output_path must use the .csv extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    morning_sheet.to_csv(
        output_path,
        index=False,
    )

    return output_path