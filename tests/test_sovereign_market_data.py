from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.sovereign_market_data import (
    DataStatus,
    GERMAN_BENCHMARKS,
    SovereignBenchmarkDefinition,
    SovereignDataValidationError,
    business_days_between,
    classify_data_status,
    validate_benchmark_observations,
)


def create_valid_observations() -> pd.DataFrame:
    """
    Create a valid RepoLens sovereign benchmark dataset.
    """
    return pd.DataFrame(
        {
            "observation_date": [
                "2026-07-24",
                "2026-07-24",
            ],
            "country": [
                "Germany",
                "Germany",
            ],
            "country_code": [
                "DE",
                "DE",
            ],
            "tenor_years": [
                2,
                10,
            ],
            "benchmark_name": [
                "Current 2-year Federal Treasury note",
                "Current 10-year Federal bond",
            ],
            "yield_percent": [
                2.10,
                2.85,
            ],
            "source_name": [
                "Deutsche Bundesbank",
                "Deutsche Bundesbank",
            ],
            "source_series": [
                "BBSSY.D.TEST2",
                "BBSSY.D.TEST10",
            ],
            "source_timestamp": [
                "2026-07-27T10:00:00Z",
                "2026-07-27T10:00:00Z",
            ],
            "data_status": [
                "OFFICIAL_DAILY",
                "OFFICIAL_DAILY",
            ],
            "business_days_stale": [
                1,
                1,
            ],
        }
    )


def test_german_registry_contains_expected_tenors() -> None:
    tenors = {
        definition.tenor_years
        for definition in GERMAN_BENCHMARKS
    }

    assert tenors == {
        2,
        5,
        10,
        30,
    }


def test_german_registry_contains_verified_series() -> None:
    full_series = {
        (
            f"{definition.dataflow}."
            f"{definition.series_key}"
        )
        for definition in GERMAN_BENCHMARKS
    }

    assert full_series == {
        "BBSSY.D.REN.EUR.A610.000000WT0202.A",
        "BBSSY.D.REN.EUR.A620.000000WT0505.A",
        "BBSSY.D.REN.EUR.A630.000000WT1010.A",
        "BBSSY.D.REN.EUR.A640.000000WT3030.A",
    }


def test_invalid_country_code_is_rejected() -> None:
    with pytest.raises(
        SovereignDataValidationError,
        match="country_code",
    ):
        SovereignBenchmarkDefinition(
            country_code="GER",
            country_name="Germany",
            tenor_years=10,
            benchmark_name="Bund",
            provider_name="Bundesbank",
            dataflow="BBSSY",
            series_key="D.TEST",
            data_status=DataStatus.OFFICIAL_DAILY,
            unit="Percent",
            description="Test.",
        )


def test_business_days_between_same_day_is_zero() -> None:
    assert business_days_between(
        observation_date=date(
            2026,
            7,
            27,
        ),
        as_of_date=date(
            2026,
            7,
            27,
        ),
    ) == 0


def test_business_days_between_ignores_weekend() -> None:
    assert business_days_between(
        observation_date=date(
            2026,
            7,
            24,
        ),
        as_of_date=date(
            2026,
            7,
            27,
        ),
    ) == 1


def test_future_observation_is_rejected() -> None:
    with pytest.raises(
        SovereignDataValidationError,
        match="must not be after",
    ):
        business_days_between(
            observation_date=date(
                2026,
                7,
                28,
            ),
            as_of_date=date(
                2026,
                7,
                27,
            ),
        )


def test_old_observation_is_classified_stale() -> None:
    assert classify_data_status(
        original_status=DataStatus.OFFICIAL_DAILY,
        business_days_stale=3,
    ) == DataStatus.STALE


def test_recent_observation_preserves_original_status() -> None:
    assert classify_data_status(
        original_status=DataStatus.OFFICIAL_DAILY,
        business_days_stale=1,
    ) == DataStatus.OFFICIAL_DAILY


def test_validate_benchmark_observations() -> None:
    validated = validate_benchmark_observations(
        create_valid_observations()
    )

    assert len(
        validated
    ) == 2

    assert validated[
        "yield_percent"
    ].dtype.kind in {
        "f",
        "i",
    }


def test_missing_contract_column_is_rejected() -> None:
    observations = create_valid_observations().drop(
        columns=[
            "source_series",
        ]
    )

    with pytest.raises(
        SovereignDataValidationError,
        match="missing required columns",
    ):
        validate_benchmark_observations(
            observations
        )


def test_invalid_status_is_rejected() -> None:
    observations = create_valid_observations()

    observations.loc[
        0,
        "data_status",
    ] = "TOTALLY_LIVE_BRO"

    with pytest.raises(
        SovereignDataValidationError,
        match="invalid statuses",
    ):
        validate_benchmark_observations(
            observations
        )