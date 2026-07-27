from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estr_market_quality import (
    MarketQualityValidationError,
    build_estr_market_quality,
    calculate_point_in_time_z_score,
    classify_market_quality,
    combine_market_quality_inputs,
    summarise_estr_market_quality,
    validate_ecb_series,
)


def create_series(
    values: np.ndarray | list[float],
    start_date: str = "2025-01-02",
) -> pd.DataFrame:
    """
    Create one deterministic business-day ECB series.
    """
    dates = pd.bdate_range(
        start=start_date,
        periods=len(values),
    )

    return pd.DataFrame(
        {
            "observation_date": dates,
            "value": values,
        }
    )


def create_market_quality_inputs(
    periods: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    Create deterministic market-quality test datasets.
    """
    phase = np.linspace(
        0.0,
        8.0,
        periods,
    )

    estr_rate = (
        2.90
        + np.sin(phase)
        * 0.002
    )

    total_volume = (
        50_000.0
        + np.sin(phase)
        * 2_000.0
    )

    percentile_25 = (
        estr_rate
        - 0.0005
    )

    percentile_75 = (
        estr_rate
        + 0.0005
    )

    active_banks = (
        45.0
        + np.sin(phase)
        * 2.0
    )

    transaction_count = (
        600.0
        + np.sin(phase)
        * 30.0
    )

    return {
        "estr_rate_data": create_series(
            estr_rate
        ),
        "total_volume_data": create_series(
            total_volume
        ),
        "percentile_25_data": create_series(
            percentile_25
        ),
        "percentile_75_data": create_series(
            percentile_75
        ),
        "active_banks_data": create_series(
            active_banks
        ),
        "transaction_count_data": create_series(
            transaction_count
        ),
    }


def test_validate_ecb_series_normalises_rows() -> None:
    data = pd.DataFrame(
        {
            "observation_date": [
                "2025-01-03",
                "2025-01-02",
                "2025-01-03",
            ],
            "value": [
                "2.91",
                "2.90",
                "2.92",
            ],
        }
    )

    validated = validate_ecb_series(
        data=data,
        dataset_name="Test data",
    )

    assert len(validated) == 2

    assert validated.iloc[-1][
        "value"
    ] == pytest.approx(
        2.92
    )


def test_validate_ecb_series_rejects_missing_columns() -> None:
    with pytest.raises(
        MarketQualityValidationError,
        match="missing required columns",
    ):
        validate_ecb_series(
            data=pd.DataFrame(
                {
                    "date": [
                        "2025-01-02"
                    ],
                    "rate": [
                        2.9
                    ],
                }
            ),
            dataset_name="Broken data",
        )


def test_combine_market_quality_inputs() -> None:
    combined = combine_market_quality_inputs(
        **create_market_quality_inputs(
            periods=30
        )
    )

    assert len(combined) == 30

    assert {
        "estr_rate",
        "total_volume_eur_mn",
        "rate_25th_percentile",
        "rate_75th_percentile",
        "active_banks",
        "transaction_count",
    }.issubset(
        combined.columns
    )


def test_invalid_percentile_order_is_rejected() -> None:
    inputs = create_market_quality_inputs(
        periods=30
    )

    inputs[
        "percentile_75_data"
    ]["value"] = (
        inputs[
            "percentile_25_data"
        ]["value"]
        - 0.01
    )

    with pytest.raises(
        MarketQualityValidationError,
        match="75th percentile",
    ):
        combine_market_quality_inputs(
            **inputs
        )


def test_point_in_time_z_score_excludes_current_value() -> None:
    values = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            100.0,
        ]
    )

    scores = calculate_point_in_time_z_score(
        values=values,
        rolling_window=4,
        minimum_history=4,
    )

    historical_mean = np.mean(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    historical_std = np.std(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        ddof=1,
    )

    expected = (
        100.0
        - historical_mean
    ) / historical_std

    assert scores.iloc[-1] == pytest.approx(
        expected
    )


def test_market_quality_score_is_bounded() -> None:
    market_quality = build_estr_market_quality(
        **create_market_quality_inputs(),
        rolling_window=60,
        minimum_history=20,
    )

    usable_scores = market_quality[
        "market_quality_score"
    ].dropna()

    assert not usable_scores.empty

    assert usable_scores.between(
        0.0,
        100.0,
    ).all()


def test_rate_dispersion_is_calculated_in_basis_points() -> None:
    market_quality = build_estr_market_quality(
        **create_market_quality_inputs(),
        rolling_window=60,
        minimum_history=20,
    )

    expected = (
        (
            market_quality.iloc[-1][
                "rate_75th_percentile"
            ]
            - market_quality.iloc[-1][
                "rate_25th_percentile"
            ]
        )
        * 100.0
    )

    assert market_quality.iloc[-1][
        "rate_dispersion_bp"
    ] == pytest.approx(
        expected
    )


def test_weak_final_conditions_trigger_alert() -> None:
    inputs = create_market_quality_inputs(
        periods=100
    )

    inputs[
        "total_volume_data"
    ].loc[
        99,
        "value",
    ] = 5_000.0

    inputs[
        "active_banks_data"
    ].loc[
        99,
        "value",
    ] = 10.0

    inputs[
        "transaction_count_data"
    ].loc[
        99,
        "value",
    ] = 50.0

    inputs[
        "percentile_25_data"
    ].loc[
        99,
        "value",
    ] = 2.80

    inputs[
        "percentile_75_data"
    ].loc[
        99,
        "value",
    ] = 3.00

    market_quality = build_estr_market_quality(
        **inputs,
        rolling_window=60,
        minimum_history=20,
    )

    latest = market_quality.iloc[-1]

    assert bool(
        latest[
            "is_market_quality_alert"
        ]
    )

    assert latest[
        "market_quality_score"
    ] < 35.0


def test_classify_market_quality() -> None:
    assert classify_market_quality(
        70.0
    ) == "Strong market quality"

    assert classify_market_quality(
        30.0
    ) == "Weak market quality"

    assert classify_market_quality(
        50.0
    ) == "Normal market quality"

    assert classify_market_quality(
        float("nan")
    ) == "Insufficient history"


def test_summary_returns_latest_metrics() -> None:
    market_quality = build_estr_market_quality(
        **create_market_quality_inputs(),
        rolling_window=60,
        minimum_history=20,
    )

    latest_date = market_quality.iloc[-1][
        "observation_date"
    ]

    summary = summarise_estr_market_quality(
        market_quality=market_quality,
        as_of_date=latest_date,
    )

    latest = market_quality.iloc[-1]

    assert summary.latest_estr_rate == pytest.approx(
        latest[
            "estr_rate"
        ]
    )

    assert summary.latest_quality_score == pytest.approx(
        latest[
            "market_quality_score"
        ]
    )

    assert summary.business_days_stale == 0
    assert summary.freshness_status == "Current"