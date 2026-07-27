from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import pandas as pd


class SovereignDataError(RuntimeError):
    """
    Base exception for RepoLens sovereign market data.
    """


class SovereignDataValidationError(SovereignDataError):
    """
    Raised when sovereign market data fails validation.
    """


class DataStatus(StrEnum):
    """
    Describe the timeliness and commercial status of market data.
    """

    LIVE = "LIVE"
    DELAYED_15M = "DELAYED_15M"
    OFFICIAL_DAILY = "OFFICIAL_DAILY"
    REFERENCE_EOD = "REFERENCE_EOD"
    STALE = "STALE"


@dataclass(frozen=True)
class SovereignBenchmarkDefinition:
    """
    Define one official sovereign benchmark yield series.
    """

    country_code: str
    country_name: str
    tenor_years: int
    benchmark_name: str
    provider_name: str
    dataflow: str
    series_key: str
    data_status: DataStatus
    unit: str
    description: str

    def __post_init__(self) -> None:
        if len(
            self.country_code
        ) != 2:
            raise SovereignDataValidationError(
                "country_code must contain exactly two characters."
            )

        if not self.country_name.strip():
            raise SovereignDataValidationError(
                "country_name must not be empty."
            )

        if self.tenor_years <= 0:
            raise SovereignDataValidationError(
                "tenor_years must be positive."
            )

        if not self.benchmark_name.strip():
            raise SovereignDataValidationError(
                "benchmark_name must not be empty."
            )

        if not self.provider_name.strip():
            raise SovereignDataValidationError(
                "provider_name must not be empty."
            )

        if not self.dataflow.strip():
            raise SovereignDataValidationError(
                "dataflow must not be empty."
            )

        if not self.series_key.strip():
            raise SovereignDataValidationError(
                "series_key must not be empty."
            )

        if not self.unit.strip():
            raise SovereignDataValidationError(
                "unit must not be empty."
            )


GERMANY_2Y: Final[SovereignBenchmarkDefinition] = (
    SovereignBenchmarkDefinition(
        country_code="DE",
        country_name="Germany",
        tenor_years=2,
        benchmark_name="Current 2-year Federal Treasury note",
        provider_name="Deutsche Bundesbank",
        dataflow="BBSSY",
        series_key="D.REN.EUR.A610.000000WT0202.A",
        data_status=DataStatus.OFFICIAL_DAILY,
        unit="Percent per annum",
        description=(
            "Official daily yield of the current two-year "
            "Federal Treasury note."
        ),
    )
)


GERMANY_5Y: Final[SovereignBenchmarkDefinition] = (
    SovereignBenchmarkDefinition(
        country_code="DE",
        country_name="Germany",
        tenor_years=5,
        benchmark_name="Current 5-year Federal note",
        provider_name="Deutsche Bundesbank",
        dataflow="BBSSY",
        series_key="D.REN.EUR.A620.000000WT0505.A",
        data_status=DataStatus.OFFICIAL_DAILY,
        unit="Percent per annum",
        description=(
            "Official daily yield of the current five-year "
            "Federal note."
        ),
    )
)


GERMANY_10Y: Final[SovereignBenchmarkDefinition] = (
    SovereignBenchmarkDefinition(
        country_code="DE",
        country_name="Germany",
        tenor_years=10,
        benchmark_name="Current 10-year Federal bond",
        provider_name="Deutsche Bundesbank",
        dataflow="BBSSY",
        series_key="D.REN.EUR.A630.000000WT1010.A",
        data_status=DataStatus.OFFICIAL_DAILY,
        unit="Percent per annum",
        description=(
            "Official daily yield of the current ten-year "
            "Federal bond."
        ),
    )
)


GERMANY_30Y: Final[SovereignBenchmarkDefinition] = (
    SovereignBenchmarkDefinition(
        country_code="DE",
        country_name="Germany",
        tenor_years=30,
        benchmark_name="Current 30-year Federal bond",
        provider_name="Deutsche Bundesbank",
        dataflow="BBSSY",
        series_key="D.REN.EUR.A640.000000WT3030.A",
        data_status=DataStatus.OFFICIAL_DAILY,
        unit="Percent per annum",
        description=(
            "Official daily yield of the current thirty-year "
            "Federal bond."
        ),
    )
)


GERMAN_BENCHMARKS: Final[
    tuple[SovereignBenchmarkDefinition, ...]
] = (
    GERMANY_2Y,
    GERMANY_5Y,
    GERMANY_10Y,
    GERMANY_30Y,
)


def business_days_between(
    observation_date: date | pd.Timestamp,
    as_of_date: date | pd.Timestamp,
) -> int:
    """
    Count elapsed weekdays after an observation date.

    This is a weekday measure rather than a complete German
    trading-calendar calculation.
    """
    observation = pd.Timestamp(
        observation_date
    ).normalize()

    as_of = pd.Timestamp(
        as_of_date
    ).normalize()

    if observation > as_of:
        raise SovereignDataValidationError(
            "observation_date must not be after as_of_date."
        )

    business_dates = pd.bdate_range(
        start=observation,
        end=as_of,
    )

    return max(
        len(
            business_dates
        )
        - 1,
        0,
    )


def classify_data_status(
    original_status: DataStatus,
    business_days_stale: int,
    stale_after_business_days: int = 3,
) -> DataStatus:
    """
    Mark an otherwise valid observation stale when sufficiently old.
    """
    if business_days_stale < 0:
        raise SovereignDataValidationError(
            "business_days_stale must not be negative."
        )

    if stale_after_business_days <= 0:
        raise SovereignDataValidationError(
            "stale_after_business_days must be positive."
        )

    if (
        business_days_stale
        >= stale_after_business_days
    ):
        return DataStatus.STALE

    return original_status


def validate_benchmark_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the standard RepoLens sovereign benchmark contract.
    """
    required_columns = {
        "observation_date",
        "country",
        "country_code",
        "tenor_years",
        "benchmark_name",
        "yield_percent",
        "source_name",
        "source_series",
        "source_timestamp",
        "data_status",
        "business_days_stale",
    }

    missing_columns = (
        required_columns
        - set(
            observations.columns
        )
    )

    if missing_columns:
        raise SovereignDataValidationError(
            "Sovereign benchmark data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    validated = observations.copy()

    validated["observation_date"] = pd.to_datetime(
        validated["observation_date"],
        errors="coerce",
    )

    validated["source_timestamp"] = pd.to_datetime(
        validated["source_timestamp"],
        errors="coerce",
        utc=True,
    )

    validated["tenor_years"] = pd.to_numeric(
        validated["tenor_years"],
        errors="coerce",
    )

    validated["yield_percent"] = pd.to_numeric(
        validated["yield_percent"],
        errors="coerce",
    )

    validated["business_days_stale"] = pd.to_numeric(
        validated["business_days_stale"],
        errors="coerce",
    )

    validated = validated.dropna(
        subset=[
            "observation_date",
            "country",
            "country_code",
            "tenor_years",
            "benchmark_name",
            "yield_percent",
            "source_name",
            "source_series",
            "source_timestamp",
            "data_status",
            "business_days_stale",
        ]
    )

    if validated.empty:
        raise SovereignDataValidationError(
            "Sovereign benchmark data contains no valid rows."
        )

    if validated[
        "tenor_years"
    ].le(
        0
    ).any():
        raise SovereignDataValidationError(
            "tenor_years must be positive."
        )

    if validated[
        "yield_percent"
    ].lt(
        -10.0
    ).any():
        raise SovereignDataValidationError(
            "yield_percent contains implausibly low values."
        )

    if validated[
        "yield_percent"
    ].gt(
        30.0
    ).any():
        raise SovereignDataValidationError(
            "yield_percent contains implausibly high values."
        )

    if validated[
        "business_days_stale"
    ].lt(
        0
    ).any():
        raise SovereignDataValidationError(
            "business_days_stale must not be negative."
        )

    allowed_statuses = {
        status.value
        for status in DataStatus
    }

    invalid_statuses = (
        set(
            validated[
                "data_status"
            ].astype(
                str
            )
        )
        - allowed_statuses
    )

    if invalid_statuses:
        raise SovereignDataValidationError(
            "Sovereign benchmark data contains invalid statuses: "
            f"{sorted(invalid_statuses)}."
        )

    validated["tenor_years"] = (
        validated["tenor_years"]
        .astype(
            int
        )
    )

    validated["business_days_stale"] = (
        validated["business_days_stale"]
        .astype(
            int
        )
    )

    validated = validated.sort_values(
        [
            "observation_date",
            "country_code",
            "tenor_years",
        ]
    )

    validated = validated.drop_duplicates(
        subset=[
            "observation_date",
            "country_code",
            "tenor_years",
        ],
        keep="last",
    )

    return validated.reset_index(
        drop=True
    )