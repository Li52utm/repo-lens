from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estr_policy_spread import (
    PolicySpreadValidationError,
    align_policy_rate_to_estr,
    build_estr_policy_spread,
    classify_transmission_regime,
    summarise_estr_policy_spread,
    validate_rate_data,
)


def create_estr_data(
    periods: int = 100,
) -> pd.DataFrame:
    """
    Create deterministic daily €STR observations.
    """
    dates = pd.bdate_range(
        start="2025-01-02",
        periods=periods,
    )

    values = (
        2.92
        + np.sin(
            np.linspace(
                0.0,
                8.0,
                periods,
            )
        )
        * 0.005
    )

    return pd.DataFrame(
        {
            "observation_date": dates,
            "value": values,
        }
    )


def create_policy_rate_data() -> pd.DataFrame:
    """
    Create deterministic deposit facility observations.
    """
    return pd.DataFrame(
        {
            "observation_date": [
                "2024-12-18",
                "2025-02-05",
                "2025-04-23",
            ],
            "value": [
                3.00,
                2.75,
                2.50,
            ],
        }
    )


def test_validate_rate_data_normalises_rows() -> None:
    data = pd.DataFrame(
        {
            "observation_date": [
                "2025-01-03",
                "2025-01-02",
                "2025-01-03",
            ],
            "value": [
                "2.91",
                "2.92",
                "2.90",
            ],
        }
    )

    validated = validate_rate_data(
        data=data,
        dataset_name="Test data",
    )

    assert len(validated) == 2

    assert validated.iloc[0][
        "observation_date"
    ] == pd.Timestamp(
        "2025-01-02"
    )

    assert validated.iloc[-1][
        "value"
    ] == pytest.approx(
        2.90
    )


def test_validate_rate_data_rejects_missing_columns() -> None:
    with pytest.raises(
        PolicySpreadValidationError,
        match="missing required columns",
    ):
        validate_rate_data(
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


def test_align_policy_rate_uses_latest_known_rate() -> None:
    estr_data = pd.DataFrame(
        {
            "observation_date": [
                "2025-01-10",
                "2025-02-10",
                "2025-04-24",
            ],
            "value": [
                2.92,
                2.67,
                2.42,
            ],
        }
    )

    aligned = align_policy_rate_to_estr(
        estr_data=estr_data,
        policy_rate_data=create_policy_rate_data(),
    )

    assert aligned[
        "deposit_facility_rate"
    ].tolist() == [
        3.00,
        2.75,
        2.50,
    ]


def test_alignment_does_not_use_future_policy_rate() -> None:
    estr_data = pd.DataFrame(
        {
            "observation_date": [
                "2025-02-04",
            ],
            "value": [
                2.92,
            ],
        }
    )

    aligned = align_policy_rate_to_estr(
        estr_data=estr_data,
        policy_rate_data=create_policy_rate_data(),
    )

    assert aligned.iloc[0][
        "deposit_facility_rate"
    ] == pytest.approx(
        3.00
    )


def test_policy_change_day_is_flagged() -> None:
    estr_data = pd.DataFrame(
        {
            "observation_date": [
                "2025-02-04",
                "2025-02-05",
                "2025-02-06",
            ],
            "value": [
                2.92,
                2.67,
                2.67,
            ],
        }
    )

    aligned = align_policy_rate_to_estr(
        estr_data=estr_data,
        policy_rate_data=create_policy_rate_data(),
    )

    assert bool(
        aligned.loc[
            aligned["observation_date"]
            == pd.Timestamp("2025-02-05"),
            "is_policy_change_day",
        ].iloc[0]
    )


def test_build_policy_spread_calculates_basis_points() -> None:
    spread_data = build_estr_policy_spread(
        estr_data=create_estr_data(),
        policy_rate_data=create_policy_rate_data(),
        rolling_window=20,
    )

    first = spread_data.iloc[0]

    expected_spread = (
        first["estr_rate"]
        - first["deposit_facility_rate"]
    ) * 100.0

    assert first[
        "estr_policy_spread_bp"
    ] == pytest.approx(
        expected_spread
    )


def test_build_output_contains_runner_columns() -> None:
    spread_data = build_estr_policy_spread(
        estr_data=create_estr_data(),
        policy_rate_data=create_policy_rate_data(),
        rolling_window=20,
    )

    assert {
        "is_policy_change_day",
        "is_unusual_spread",
        "transmission_regime",
    }.issubset(
        spread_data.columns
    )


def test_large_spread_is_flagged() -> None:
    estr_data = create_estr_data(
        periods=100
    )

    estr_data.loc[
        estr_data.index[-1],
        "value",
    ] += 0.20

    spread_data = build_estr_policy_spread(
        estr_data=estr_data,
        policy_rate_data=create_policy_rate_data(),
        rolling_window=20,
        alert_z_threshold=2.0,
    )

    assert bool(
        spread_data.iloc[-1][
            "is_unusual_spread"
        ]
    )


def test_classify_transmission_regime() -> None:
    assert classify_transmission_regime(
        spread_bp=8.0,
        spread_z_score=0.5,
    ) == "Above policy rate"

    assert classify_transmission_regime(
        spread_bp=-20.0,
        spread_z_score=-0.5,
    ) == "Materially below policy rate"

    assert classify_transmission_regime(
        spread_bp=2.0,
        spread_z_score=2.5,
    ) == "Unusually above policy rate"

    assert classify_transmission_regime(
        spread_bp=-8.0,
        spread_z_score=-2.5,
    ) == "Unusually below policy rate"

    assert classify_transmission_regime(
        spread_bp=-7.0,
        spread_z_score=0.2,
    ) == "Normal transmission"


def test_summary_returns_latest_metrics() -> None:
    spread_data = build_estr_policy_spread(
        estr_data=create_estr_data(),
        policy_rate_data=create_policy_rate_data(),
        rolling_window=20,
    )

    summary = summarise_estr_policy_spread(
        spread_data=spread_data
    )

    latest = spread_data.iloc[-1]

    assert summary.latest_estr_rate == pytest.approx(
        latest["estr_rate"]
    )

    assert (
        summary.latest_deposit_facility_rate
        == pytest.approx(
            latest["deposit_facility_rate"]
        )
    )

    assert summary.latest_spread_bp == pytest.approx(
        latest["estr_policy_spread_bp"]
    )

    assert (
        summary.is_policy_change_day
        == bool(
            latest[
                "is_policy_change_day"
            ]
        )
    )