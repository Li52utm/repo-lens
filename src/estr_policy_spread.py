from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


DEFAULT_ROLLING_WINDOW: Final[int] = 60
DEFAULT_ALERT_Z_THRESHOLD: Final[float] = 2.0


class PolicySpreadError(RuntimeError):
    """
    Base exception for €STR policy spread analytics.
    """


class PolicySpreadValidationError(PolicySpreadError):
    """
    Raised when policy spread input data is invalid.
    """


@dataclass(frozen=True)
class PolicySpreadSummary:
    """
    Store the latest €STR versus deposit facility metrics.
    """

    latest_observation_date: pd.Timestamp
    latest_estr_rate: float
    latest_deposit_facility_rate: float
    latest_spread_bp: float
    rolling_mean_spread_bp: float
    rolling_volatility_bp: float
    latest_spread_z_score: float
    transmission_regime: str
    is_policy_change_day: bool
    is_unusual_spread: bool


def validate_rate_data(
    data: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate and normalise an ECB interest rate dataset.
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
        raise PolicySpreadValidationError(
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
        raise PolicySpreadValidationError(
            f"{dataset_name} contains no valid observations."
        )

    if validated["value"].lt(-10.0).any():
        raise PolicySpreadValidationError(
            f"{dataset_name} contains implausibly low values."
        )

    if validated["value"].gt(25.0).any():
        raise PolicySpreadValidationError(
            f"{dataset_name} contains implausibly high values."
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


def align_policy_rate_to_estr(
    estr_data: pd.DataFrame,
    policy_rate_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the latest known deposit facility rate to each €STR date.

    A backward as-of merge ensures only policy information available
    on or before each €STR observation date is used.
    """
    estr = validate_rate_data(
        data=estr_data,
        dataset_name="€STR data",
    ).rename(
        columns={
            "value": "estr_rate",
        }
    )

    policy = validate_rate_data(
        data=policy_rate_data,
        dataset_name="Deposit facility rate data",
    ).rename(
        columns={
            "value": "deposit_facility_rate",
        }
    )

    policy["previous_deposit_facility_rate"] = (
        policy["deposit_facility_rate"]
        .shift(1)
    )

    policy["is_policy_change_day"] = (
        policy["deposit_facility_rate"]
        .ne(
            policy[
                "previous_deposit_facility_rate"
            ]
        )
    )

    policy.loc[
        policy.index[0],
        "is_policy_change_day",
    ] = False

    aligned = pd.merge_asof(
        left=estr.sort_values(
            "observation_date"
        ),
        right=policy[
            [
                "observation_date",
                "deposit_facility_rate",
                "is_policy_change_day",
            ]
        ].sort_values(
            "observation_date"
        ),
        on="observation_date",
        direction="backward",
        allow_exact_matches=True,
    )

    aligned = aligned.dropna(
        subset=[
            "deposit_facility_rate",
        ]
    )

    if aligned.empty:
        raise PolicySpreadValidationError(
            "No overlapping €STR and deposit facility observations "
            "were available after point-in-time alignment."
        )

    aligned["is_policy_change_day"] = (
        aligned["is_policy_change_day"]
        .fillna(False)
        .astype(bool)
    )

    return aligned.reset_index(
        drop=True
    )


def classify_transmission_regime(
    spread_bp: float,
    spread_z_score: float,
    alert_z_threshold: float = DEFAULT_ALERT_Z_THRESHOLD,
) -> str:
    """
    Classify the latest €STR relationship to the policy rate.
    """
    if alert_z_threshold <= 0:
        raise ValueError(
            "alert_z_threshold must be positive."
        )

    if np.isnan(spread_bp):
        return "Insufficient data"

    if (
        not np.isnan(spread_z_score)
        and abs(spread_z_score) >= alert_z_threshold
    ):
        if spread_z_score > 0:
            return "Unusually above policy rate"

        return "Unusually below policy rate"

    if spread_bp > 5.0:
        return "Above policy rate"

    if spread_bp < -15.0:
        return "Materially below policy rate"

    return "Normal transmission"


def build_estr_policy_spread(
    estr_data: pd.DataFrame,
    policy_rate_data: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    alert_z_threshold: float = DEFAULT_ALERT_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Build the €STR versus ECB deposit facility spread dataset.
    """
    if rolling_window < 10:
        raise ValueError(
            "rolling_window must be at least 10 observations."
        )

    if alert_z_threshold <= 0:
        raise ValueError(
            "alert_z_threshold must be positive."
        )

    spread_data = align_policy_rate_to_estr(
        estr_data=estr_data,
        policy_rate_data=policy_rate_data,
    )

    spread_data["estr_policy_spread_bp"] = (
        (
            spread_data["estr_rate"]
            - spread_data[
                "deposit_facility_rate"
            ]
        )
        * 100.0
    )

    spread_data["rolling_mean_spread_bp"] = (
        spread_data["estr_policy_spread_bp"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .mean()
    )

    spread_data["rolling_volatility_bp"] = (
        spread_data["estr_policy_spread_bp"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .std(ddof=1)
    )

    usable_volatility = (
        spread_data["rolling_volatility_bp"]
        .replace(
            0.0,
            np.nan,
        )
    )

    spread_data["spread_z_score"] = (
        spread_data["estr_policy_spread_bp"]
        - spread_data[
            "rolling_mean_spread_bp"
        ]
    ) / usable_volatility

    spread_data["is_unusual_spread"] = (
        spread_data["spread_z_score"]
        .abs()
        .ge(
            alert_z_threshold
        )
        .fillna(False)
    )

    spread_data["transmission_regime"] = [
        classify_transmission_regime(
            spread_bp=float(spread),
            spread_z_score=(
                float(z_score)
                if pd.notna(z_score)
                else float("nan")
            ),
            alert_z_threshold=alert_z_threshold,
        )
        for spread, z_score in zip(
            spread_data[
                "estr_policy_spread_bp"
            ],
            spread_data[
                "spread_z_score"
            ],
            strict=True,
        )
    ]

    return spread_data


def summarise_estr_policy_spread(
    spread_data: pd.DataFrame,
) -> PolicySpreadSummary:
    """
    Summarise the latest policy transmission state.
    """
    required_columns = {
        "observation_date",
        "estr_rate",
        "deposit_facility_rate",
        "estr_policy_spread_bp",
        "rolling_mean_spread_bp",
        "rolling_volatility_bp",
        "spread_z_score",
        "transmission_regime",
        "is_policy_change_day",
        "is_unusual_spread",
    }

    missing_columns = (
        required_columns
        - set(spread_data.columns)
    )

    if missing_columns:
        raise PolicySpreadValidationError(
            "Policy spread data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if spread_data.empty:
        raise PolicySpreadValidationError(
            "Policy spread dataset is empty."
        )

    latest = (
        spread_data
        .sort_values(
            "observation_date"
        )
        .iloc[-1]
    )

    def optional_float(
        value: object,
    ) -> float:
        if pd.isna(value):
            return float("nan")

        return float(value)

    return PolicySpreadSummary(
        latest_observation_date=pd.Timestamp(
            latest["observation_date"]
        ),
        latest_estr_rate=float(
            latest["estr_rate"]
        ),
        latest_deposit_facility_rate=float(
            latest[
                "deposit_facility_rate"
            ]
        ),
        latest_spread_bp=float(
            latest[
                "estr_policy_spread_bp"
            ]
        ),
        rolling_mean_spread_bp=optional_float(
            latest[
                "rolling_mean_spread_bp"
            ]
        ),
        rolling_volatility_bp=optional_float(
            latest[
                "rolling_volatility_bp"
            ]
        ),
        latest_spread_z_score=optional_float(
            latest[
                "spread_z_score"
            ]
        ),
        transmission_regime=str(
            latest[
                "transmission_regime"
            ]
        ),
        is_policy_change_day=bool(
            latest[
                "is_policy_change_day"
            ]
        ),
        is_unusual_spread=bool(
            latest[
                "is_unusual_spread"
            ]
        ),
    )


def load_rate_csv(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load a locally stored ECB rate CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Rate input file does not exist: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "input_path must use the .csv extension."
        )

    return pd.read_csv(
        input_path
    )


def save_policy_spread(
    spread_data: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save the processed policy spread dataset.
    """
    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "output_path must use the .csv extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spread_data.to_csv(
        output_path,
        index=False,
    )

    return output_path