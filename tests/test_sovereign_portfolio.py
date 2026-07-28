from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.sovereign_instruments import (
    GERMANY_5Y_BOBL,
    GERMANY_10Y_BUND,
    ITALY_10Y_BTP,
    ITALY_30Y_BTP,
)
from src.sovereign_portfolio import (
    SovereignPortfolioPosition,
    SovereignPortfolioValidationError,
    aggregate_dv01_by_country,
    aggregate_dv01_by_tenor,
    build_italy_germany_spread_scenarios,
    build_parallel_portfolio_scenarios,
    build_portfolio,
    build_portfolio_results,
    positions_to_frame,
    risk_contribution_frame,
    summarise_portfolio,
)
from src.sovereign_relative_value import PositionDirection
from src.sovereign_snapshot import SovereignYieldInput


SETTLEMENT_DATE = date(
    2026,
    7,
    28,
)


def create_german_curve() -> pd.DataFrame:
    """
    Create deterministic German benchmark observations.
    """
    return pd.DataFrame(
        [
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 2,
                "yield_percent": 2.20,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 5,
                "yield_percent": 2.45,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 10,
                "yield_percent": 2.85,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
            {
                "observation_date": "2026-07-28",
                "country_code": "DE",
                "tenor_years": 30,
                "yield_percent": 3.15,
                "source_name": "Deutsche Bundesbank",
                "data_status": "OFFICIAL_DAILY",
            },
        ]
    )


def create_italy_yield_input(
    isin: str,
    yield_percent: float,
) -> SovereignYieldInput:
    """
    Create a deterministic Italian desk-input yield.
    """
    return SovereignYieldInput(
        isin=isin,
        yield_percent=yield_percent,
        observation_date=SETTLEMENT_DATE,
        source_name="Desk input",
    )


def create_standard_positions() -> tuple[
    SovereignPortfolioPosition,
    ...,
]:
    """
    Create a mixed long-short sovereign portfolio.
    """
    return (
        SovereignPortfolioPosition(
            position_id="BTP10-LONG",
            instrument=ITALY_10Y_BTP,
            direction=PositionDirection.LONG,
            notional_eur=15_000_000.0,
            yield_input=create_italy_yield_input(
                isin=ITALY_10Y_BTP.isin,
                yield_percent=3.85,
            ),
        ),
        SovereignPortfolioPosition(
            position_id="BUND10-SHORT",
            instrument=GERMANY_10Y_BUND,
            direction=PositionDirection.SHORT,
            notional_eur=12_000_000.0,
        ),
        SovereignPortfolioPosition(
            position_id="BOBL5-LONG",
            instrument=GERMANY_5Y_BOBL,
            direction=PositionDirection.LONG,
            notional_eur=8_000_000.0,
        ),
        SovereignPortfolioPosition(
            position_id="BTP30-SHORT",
            instrument=ITALY_30Y_BTP,
            direction=PositionDirection.SHORT,
            notional_eur=5_000_000.0,
            yield_input=create_italy_yield_input(
                isin=ITALY_30Y_BTP.isin,
                yield_percent=4.65,
            ),
        ),
    )


def test_position_requires_positive_notional() -> None:
    with pytest.raises(
        SovereignPortfolioValidationError,
        match="must be positive",
    ):
        SovereignPortfolioPosition(
            position_id="INVALID",
            instrument=GERMANY_10Y_BUND,
            direction=PositionDirection.LONG,
            notional_eur=0.0,
        )


def test_position_requires_non_empty_id() -> None:
    with pytest.raises(
        SovereignPortfolioValidationError,
        match="must not be empty",
    ):
        SovereignPortfolioPosition(
            position_id="",
            instrument=GERMANY_10Y_BUND,
            direction=PositionDirection.LONG,
            notional_eur=10_000_000.0,
        )


def test_position_yield_input_must_match_isin() -> None:
    mismatched_input = create_italy_yield_input(
        isin=ITALY_30Y_BTP.isin,
        yield_percent=4.65,
    )

    with pytest.raises(
        SovereignPortfolioValidationError,
        match="must match",
    ):
        SovereignPortfolioPosition(
            position_id="BTP10",
            instrument=ITALY_10Y_BTP,
            direction=PositionDirection.LONG,
            notional_eur=10_000_000.0,
            yield_input=mismatched_input,
        )


def test_duplicate_position_ids_are_rejected() -> None:
    position = SovereignPortfolioPosition(
        position_id="DUPLICATE",
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.LONG,
        notional_eur=10_000_000.0,
    )

    with pytest.raises(
        SovereignPortfolioValidationError,
        match="must be unique",
    ):
        build_portfolio_results(
            positions=(
                position,
                position,
            ),
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )


def test_portfolio_builds_all_positions() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    assert len(
        results
    ) == 4

    assert {
        result.position_id
        for result in results
    } == {
        "BTP10-LONG",
        "BUND10-SHORT",
        "BOBL5-LONG",
        "BTP30-SHORT",
    }


def test_long_and_short_positions_have_correct_signs() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    result_by_id = {
        result.position_id: result
        for result in results
    }

    assert (
        result_by_id[
            "BTP10-LONG"
        ].signed_notional_eur
        > 0.0
    )

    assert (
        result_by_id[
            "BUND10-SHORT"
        ].signed_notional_eur
        < 0.0
    )

    assert (
        result_by_id[
            "BTP10-LONG"
        ].signed_dv01_eur
        > 0.0
    )

    assert (
        result_by_id[
            "BUND10-SHORT"
        ].signed_dv01_eur
        < 0.0
    )


def test_portfolio_summary_contract() -> None:
    results, summary = build_portfolio(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    assert len(
        results
    ) == 4

    assert summary.position_count == 4

    assert summary.gross_notional_eur == pytest.approx(
        40_000_000.0
    )

    assert summary.net_notional_eur == pytest.approx(
        6_000_000.0
    )

    assert summary.gross_dv01_eur > 0.0

    assert summary.largest_risk_position_id in {
        "BTP10-LONG",
        "BUND10-SHORT",
        "BOBL5-LONG",
        "BTP30-SHORT",
    }

    assert 0.0 < summary.largest_risk_share <= 1.0


def test_summary_concentration_warning() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    summary = summarise_portfolio(
        results=results,
        settlement_date=SETTLEMENT_DATE,
        concentration_warning_threshold=0.20,
    )

    assert summary.concentration_warning


def test_positions_frame_contains_expected_columns() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    frame = positions_to_frame(
        results
    )

    assert len(
        frame
    ) == 4

    assert {
        "position_id",
        "isin",
        "country",
        "benchmark_tenor_years",
        "direction",
        "notional_eur",
        "yield_percent",
        "dirty_price",
        "absolute_dv01_eur",
        "signed_dv01_eur",
        "data_status",
    }.issubset(
        frame.columns
    )


def test_country_aggregation() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    country_risk = aggregate_dv01_by_country(
        results
    )

    assert set(
        country_risk["country"]
    ) == {
        "Germany",
        "Italy",
    }

    assert country_risk[
        "gross_dv01_eur"
    ].gt(
        0.0
    ).all()


def test_tenor_aggregation() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    tenor_risk = aggregate_dv01_by_tenor(
        results
    )

    assert set(
        tenor_risk[
            "benchmark_tenor_years"
        ]
    ) == {
        5,
        10,
        30,
    }

    ten_year = tenor_risk.loc[
        tenor_risk[
            "benchmark_tenor_years"
        ].eq(
            10
        )
    ].iloc[0]

    assert int(
        ten_year[
            "position_count"
        ]
    ) == 2


def test_parallel_scenario_direction_for_long_bund() -> None:
    position = SovereignPortfolioPosition(
        position_id="BUND-LONG",
        instrument=GERMANY_10Y_BUND,
        direction=PositionDirection.LONG,
        notional_eur=10_000_000.0,
    )

    results = build_portfolio_results(
        positions=(
            position,
        ),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    scenarios = build_parallel_portfolio_scenarios(
        results=results,
        yield_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    assert scenarios.iloc[0][
        "portfolio_pnl_eur"
    ] > 0.0

    assert scenarios.iloc[1][
        "portfolio_pnl_eur"
    ] < 0.0


def test_long_btp_short_bund_loses_when_spread_widens() -> None:
    positions = (
        SovereignPortfolioPosition(
            position_id="BTP10-LONG",
            instrument=ITALY_10Y_BTP,
            direction=PositionDirection.LONG,
            notional_eur=10_000_000.0,
            yield_input=create_italy_yield_input(
                isin=ITALY_10Y_BTP.isin,
                yield_percent=3.85,
            ),
        ),
        SovereignPortfolioPosition(
            position_id="BUND10-SHORT",
            instrument=GERMANY_10Y_BUND,
            direction=PositionDirection.SHORT,
            notional_eur=10_000_000.0,
        ),
    )

    results = build_portfolio_results(
        positions=positions,
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    scenarios = build_italy_germany_spread_scenarios(
        results=results,
        spread_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    narrowing = scenarios.iloc[0]
    widening = scenarios.iloc[1]

    assert narrowing[
        "portfolio_pnl_eur"
    ] > 0.0

    assert widening[
        "portfolio_pnl_eur"
    ] < 0.0


def test_risk_contribution_shares_sum_to_one() -> None:
    results = build_portfolio_results(
        positions=create_standard_positions(),
        german_curve=create_german_curve(),
        settlement_date=SETTLEMENT_DATE,
    )

    contributions = risk_contribution_frame(
        results
    )

    assert contributions[
        "gross_dv01_share"
    ].sum() == pytest.approx(
        1.0
    )

    assert contributions.iloc[0][
        "absolute_dv01_eur"
    ] >= contributions.iloc[-1][
        "absolute_dv01_eur"
    ]


def test_missing_italian_yield_is_rejected() -> None:
    position = SovereignPortfolioPosition(
        position_id="BTP-MISSING-YIELD",
        instrument=ITALY_10Y_BTP,
        direction=PositionDirection.LONG,
        notional_eur=10_000_000.0,
    )

    with pytest.raises(
        SovereignPortfolioValidationError,
        match="explicit instrument-level yield",
    ):
        build_portfolio_results(
            positions=(
                position,
            ),
            german_curve=create_german_curve(),
            settlement_date=SETTLEMENT_DATE,
        )