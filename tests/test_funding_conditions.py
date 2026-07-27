from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.funding_conditions import (
    FundingDataValidationError,
    build_funding_conditions,
    calculate_business_days_stale,
    classify_freshness,
    classify_funding_regime,
    summarise_funding_conditions,
    validate_estr_data,
)


def create_sample_estr_data(
    periods: int = 40,
) -> pd.DataFrame:
    """
    Create deterministic €STR test observations.
    """
    dates = pd.bdate_range(
        start="2026-01-02",
        periods=periods,
    )

    values = (
        2.0
        + np.linspace(
            0.0,
            0.039,
            periods,
        )
    )

    return pd.DataFrame(
        {
            "observation_date": dates,
            "value": values,
        }
    )


def test_validate_estr_data_normalises_input() -> None:
    data = pd.DataFrame(
        {
            "observation_date": [
                "2026-01-05",
                "2026-01-02",
                "2026-01-05",
            ],
            "value": [
                "2.100",
                "2.000",
                "2.200",
            ],
        }
    )

    validated = validate_estr_data(
        data
    )

    assert len(validated) == 2

    assert validated.iloc[0][
        "observation_date"
    ] == pd.Timestamp(
        "2026-01-02"
    )

    assert validated.iloc[-1][
        "value"
    ] == pytest.approx(
        2.200
    )


def test_validate_estr_data_rejects_missing_columns() -> None:
    with pytest.raises(
        FundingDataValidationError,
        match="missing required columns",
    ):
        validate_estr_data(
            pd.DataFrame(
                {
                    "date": [
                        "2026-01-02"
                    ],
                    "rate": [
                        2.0
                    ],
                }
            )
        )


def test_validate_estr_data_rejects_empty_valid_data() -> None:
    with pytest.raises(
        FundingDataValidationError,
        match="no valid observations",
    ):
        validate_estr_data(
            pd.DataFrame(
                {
                    "observation_date": [
                        "not-a-date"
                    ],
                    "value": [
                        "not-a-number"
                    ],
                }
            )
        )


def test_build_funding_conditions_calculates_changes() -> None:
    conditions = build_funding_conditions(
        data=create_sample_estr_data(),
        rolling_window=10,
    )

    assert {
        "estr_rate",
        "daily_change_bp",
        "rolling_change_volatility_bp",
        "change_z_score",
        "is_month_end",
        "is_quarter_end",
        "funding_regime",
        "is_abnormal_move",
    }.issubset(
        conditions.columns
    )

    assert conditions.iloc[1][
        "daily_change_bp"
    ] == pytest.approx(
        0.1
    )


def test_build_funding_conditions_flags_large_move() -> None:
    data = create_sample_estr_data(
        periods=45
    )

    data.loc[
        data.index[-1],
        "value",
    ] += 0.25

    conditions = build_funding_conditions(
        data=data,
        rolling_window=20,
        stress_z_threshold=2.0,
    )

    assert bool(
        conditions.iloc[-1][
            "is_abnormal_move"
        ]
    )

    assert conditions.iloc[-1][
        "funding_regime"
    ] == "Sharp tightening"


def test_classify_funding_regime() -> None:
    assert classify_funding_regime(
        2.5
    ) == "Sharp tightening"

    assert classify_funding_regime(
        -2.5
    ) == "Sharp easing"

    assert classify_funding_regime(
        1.2
    ) == "Mild tightening"

    assert classify_funding_regime(
        -1.2
    ) == "Mild easing"

    assert classify_funding_regime(
        0.2
    ) == "Normal"

    assert classify_funding_regime(
        float("nan")
    ) == "Insufficient history"


def test_calculate_business_days_stale() -> None:
    result = calculate_business_days_stale(
        latest_observation_date=pd.Timestamp(
            "2026-07-24"
        ),
        as_of_date=pd.Timestamp(
            "2026-07-27"
        ),
    )

    assert result == 1


def test_classify_freshness() -> None:
    assert classify_freshness(
        0
    ) == "Current"

    assert classify_freshness(
        1
    ) == "Delayed"

    assert classify_freshness(
        3
    ) == "Stale"


def test_summarise_funding_conditions() -> None:
    conditions = build_funding_conditions(
        data=create_sample_estr_data(),
        rolling_window=10,
    )

    latest_date = conditions.iloc[-1][
        "observation_date"
    ]

    summary = summarise_funding_conditions(
        conditions=conditions,
        as_of_date=latest_date,
    )

    assert summary.latest_estr_rate == pytest.approx(
        conditions.iloc[-1][
            "estr_rate"
        ]
    )

    assert summary.business_days_stale == 0
    assert summary.freshness_status == "Current"