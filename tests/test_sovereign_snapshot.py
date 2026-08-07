from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.sovereign_instrument_catalog import (
    instrument_by_isin,
)
from src.sovereign_instruments import (
    GERMANY_10Y_BUND,
    ITALY_10Y_BTP,
)
from src.sovereign_snapshot import (
    SnapshotDataStatus,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    build_registry_snapshot,
    german_benchmark_for_tenor,
    optional_german_benchmark_for_tenor,
    prepare_german_benchmark_curve,
    snapshot_scenarios,
)


def create_german_curve() -> pd.DataFrame:
    """
    Create deterministic official German benchmark yields.
    """
    rows: list[
        dict[str, object]
    ] = []

    yields = {
        2: 2.20,
        5: 2.45,
        10: 2.85,
        30: 3.15,
    }

    for (
        tenor,
        yield_percent,
    ) in yields.items():
        rows.append(
            {
                "observation_date": "2026-07-27",
                "country": "Germany",
                "country_code": "DE",
                "tenor_years": tenor,
                "benchmark_name": (
                    f"Germany {tenor}Y"
                ),
                "yield_percent": (
                    yield_percent
                    - 0.05
                ),
                "source_name": (
                    "Deutsche Bundesbank"
                ),
                "source_series": (
                    f"BBSSY.TEST.{tenor}"
                ),
                "source_timestamp": (
                    "2026-07-27T18:00:00Z"
                ),
                "data_status": (
                    "OFFICIAL_DAILY"
                ),
                "business_days_stale": 0,
            }
        )

        rows.append(
            {
                "observation_date": "2026-07-28",
                "country": "Germany",
                "country_code": "DE",
                "tenor_years": tenor,
                "benchmark_name": (
                    f"Germany {tenor}Y"
                ),
                "yield_percent": (
                    yield_percent
                ),
                "source_name": (
                    "Deutsche Bundesbank"
                ),
                "source_series": (
                    f"BBSSY.TEST.{tenor}"
                ),
                "source_timestamp": (
                    "2026-07-28T18:00:00Z"
                ),
                "data_status": (
                    "OFFICIAL_DAILY"
                ),
                "business_days_stale": 0,
            }
        )

    return pd.DataFrame(
        rows
    )


def test_prepare_german_curve_selects_latest_tenors() -> None:
    prepared = prepare_german_benchmark_curve(
        create_german_curve()
    )

    assert len(
        prepared
    ) == 4

    assert set(
        prepared[
            "tenor_years"
        ]
    ) == {
        2,
        5,
        10,
        30,
    }

    ten_year = prepared.loc[
        prepared[
            "tenor_years"
        ].eq(
            10
        )
    ].iloc[
        0
    ]

    assert ten_year[
        "yield_percent"
    ] == pytest.approx(
        2.85
    )


def test_missing_curve_column_is_rejected() -> None:
    broken_curve = (
        create_german_curve()
        .drop(
            columns=[
                "yield_percent",
            ]
        )
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="missing required columns",
    ):
        prepare_german_benchmark_curve(
            broken_curve
        )


def test_optional_benchmark_returns_none_for_missing_tenor() -> None:
    prepared = prepare_german_benchmark_curve(
        create_german_curve()
    )

    benchmark = (
        optional_german_benchmark_for_tenor(
            prepared_curve=prepared,
            tenor_years=3,
        )
    )

    assert benchmark is None


def test_strict_benchmark_still_rejects_missing_tenor() -> None:
    prepared = prepare_german_benchmark_curve(
        create_german_curve()
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="No German benchmark yield is available",
    ):
        german_benchmark_for_tenor(
            prepared_curve=prepared,
            tenor_years=3,
        )


def test_german_instrument_uses_official_benchmark() -> None:
    snapshot = build_instrument_snapshot(
        instrument=GERMANY_10Y_BUND,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            10_000_000.0
        ),
    )

    assert snapshot.market_data_available

    assert snapshot.data_status == (
        SnapshotDataStatus
        .OFFICIAL_DAILY
    )

    assert (
        snapshot.yield_percent
        == pytest.approx(
            2.85
        )
    )

    assert (
        snapshot
        .spread_to_germany_bp
        == pytest.approx(
            0.0
        )
    )

    assert snapshot.clean_price > 0.0

    assert (
        snapshot.position_dv01_eur
        > 0.0
    )


def test_italian_instrument_requires_explicit_yield() -> None:
    with pytest.raises(
        SovereignSnapshotValidationError,
        match="explicit instrument-level yield",
    ):
        build_instrument_snapshot(
            instrument=ITALY_10Y_BTP,
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            position_notional_eur=(
                10_000_000.0
            ),
        )


def test_italian_spread_is_calculated_against_germany() -> None:
    yield_input = SovereignYieldInput(
        isin=ITALY_10Y_BTP.isin,
        yield_percent=3.85,
        observation_date=date(
            2026,
            7,
            28,
        ),
        source_name=(
            "Desk input from market terminal"
        ),
    )

    snapshot = build_instrument_snapshot(
        instrument=ITALY_10Y_BTP,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            25_000_000.0
        ),
        explicit_yield_input=(
            yield_input
        ),
    )

    assert snapshot.data_status == (
        SnapshotDataStatus
        .DESK_INPUT
    )

    assert (
        snapshot.yield_percent
        == pytest.approx(
            3.85
        )
    )

    assert (
        snapshot
        .german_benchmark_yield_percent
        == pytest.approx(
            2.85
        )
    )

    assert (
        snapshot
        .spread_to_germany_bp
        == pytest.approx(
            100.0
        )
    )

    assert (
        snapshot.position_dv01_eur
        > 0.0
    )


def test_italian_three_year_can_price_without_german_three_year() -> None:
    instrument = instrument_by_isin(
        "IT0005467482"
    )

    yield_input = SovereignYieldInput(
        isin=instrument.isin,
        yield_percent=2.80,
        observation_date=date(
            2026,
            7,
            28,
        ),
        source_name="Desk input",
    )

    snapshot = build_instrument_snapshot(
        instrument=instrument,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            10_000_000.0
        ),
        explicit_yield_input=(
            yield_input
        ),
    )

    assert (
        snapshot.market_data_available
    )

    assert snapshot.data_status == (
        SnapshotDataStatus
        .DESK_INPUT
    )

    assert (
        snapshot.yield_percent
        == pytest.approx(
            2.80
        )
    )

    assert pd.isna(
        snapshot
        .german_benchmark_yield_percent
    )

    assert pd.isna(
        snapshot
        .spread_to_germany_bp
    )

    assert snapshot.clean_price > 0.0


def test_german_seven_year_can_use_desk_yield_without_exact_benchmark() -> None:
    instrument = instrument_by_isin(
        "DE000BU27014"
    )

    yield_input = SovereignYieldInput(
        isin=instrument.isin,
        yield_percent=2.70,
        observation_date=date(
            2026,
            7,
            28,
        ),
        source_name="Desk input",
    )

    snapshot = build_instrument_snapshot(
        instrument=instrument,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        explicit_yield_input=(
            yield_input
        ),
    )

    assert (
        snapshot.market_data_available
    )

    assert snapshot.data_status == (
        SnapshotDataStatus
        .DESK_INPUT
    )

    assert (
        snapshot.yield_percent
        == pytest.approx(
            2.70
        )
    )

    assert pd.isna(
        snapshot
        .german_benchmark_yield_percent
    )

    assert pd.isna(
        snapshot
        .spread_to_germany_bp
    )


def test_german_seven_year_requires_desk_yield_when_benchmark_missing() -> None:
    instrument = instrument_by_isin(
        "DE000BU27014"
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="will not interpolate",
    ):
        build_instrument_snapshot(
            instrument=instrument,
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
        )


def test_explicit_input_isin_must_match() -> None:
    yield_input = SovereignYieldInput(
        isin="IT0005668238",
        yield_percent=3.85,
        observation_date=date(
            2026,
            7,
            28,
        ),
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="does not match",
    ):
        build_instrument_snapshot(
            instrument=ITALY_10Y_BTP,
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_input=(
                yield_input
            ),
        )


def test_future_market_observation_is_rejected() -> None:
    yield_input = SovereignYieldInput(
        isin=ITALY_10Y_BTP.isin,
        yield_percent=3.85,
        observation_date=date(
            2026,
            7,
            29,
        ),
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="must not be after settlement",
    ):
        build_instrument_snapshot(
            instrument=ITALY_10Y_BTP,
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_input=(
                yield_input
            ),
        )


def test_registry_snapshot_marks_missing_italy_data_unavailable() -> None:
    snapshot = build_registry_snapshot(
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            10_000_000.0
        ),
    )

    assert len(
        snapshot
    ) == 8

    germany = snapshot.loc[
        snapshot[
            "country"
        ].eq(
            "Germany"
        )
    ]

    italy = snapshot.loc[
        snapshot[
            "country"
        ].eq(
            "Italy"
        )
    ]

    assert germany[
        "market_data_available"
    ].all()

    assert not italy[
        "market_data_available"
    ].any()

    assert set(
        italy[
            "data_status"
        ]
    ) == {
        "UNAVAILABLE",
    }

    assert italy[
        "yield_percent"
    ].isna().all()


def test_registry_snapshot_applies_supplied_italian_input() -> None:
    yield_input = SovereignYieldInput(
        isin=ITALY_10Y_BTP.isin,
        yield_percent=3.85,
        observation_date=date(
            2026,
            7,
            28,
        ),
    )

    snapshot = build_registry_snapshot(
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        explicit_yield_inputs=(
            yield_input,
        ),
    )

    italy_10y = snapshot.loc[
        snapshot[
            "isin"
        ].eq(
            ITALY_10Y_BTP.isin
        )
    ].iloc[
        0
    ]

    assert bool(
        italy_10y[
            "market_data_available"
        ]
    )

    assert (
        italy_10y[
            "yield_percent"
        ]
        == pytest.approx(
            3.85
        )
    )

    assert (
        italy_10y[
            "spread_to_germany_bp"
        ]
        == pytest.approx(
            100.0
        )
    )


def test_registry_snapshot_supports_expanded_instruments() -> None:
    italian_three_year = (
        instrument_by_isin(
            "IT0005467482"
        )
    )

    yield_input = SovereignYieldInput(
        isin=(
            italian_three_year.isin
        ),
        yield_percent=2.80,
        observation_date=date(
            2026,
            7,
            28,
        ),
    )

    snapshot = build_registry_snapshot(
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        explicit_yield_inputs=(
            yield_input,
        ),
        instruments=(
            italian_three_year,
        ),
    )

    assert len(
        snapshot
    ) == 1

    result = snapshot.iloc[
        0
    ]

    assert bool(
        result[
            "market_data_available"
        ]
    )

    assert (
        result[
            "yield_percent"
        ]
        == pytest.approx(
            2.80
        )
    )

    assert pd.isna(
        result[
            "german_benchmark_yield_percent"
        ]
    )

    assert pd.isna(
        result[
            "spread_to_germany_bp"
        ]
    )


def test_duplicate_explicit_inputs_are_rejected() -> None:
    yield_input = SovereignYieldInput(
        isin=ITALY_10Y_BTP.isin,
        yield_percent=3.85,
        observation_date=date(
            2026,
            7,
            28,
        ),
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="Duplicate explicit yield",
    ):
        build_registry_snapshot(
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_inputs=(
                yield_input,
                yield_input,
            ),
        )


def test_unknown_explicit_input_is_rejected_against_supplied_collection() -> None:
    yield_input = SovereignYieldInput(
        isin="IT0005467482",
        yield_percent=2.80,
        observation_date=date(
            2026,
            7,
            28,
        ),
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="not present in the supplied",
    ):
        build_registry_snapshot(
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_inputs=(
                yield_input,
            ),
            instruments=(
                GERMANY_10Y_BUND,
            ),
        )


def test_position_dv01_scales_with_notional() -> None:
    ten_million = build_instrument_snapshot(
        instrument=GERMANY_10Y_BUND,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            10_000_000.0
        ),
    )

    twenty_million = build_instrument_snapshot(
        instrument=GERMANY_10Y_BUND,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=(
            20_000_000.0
        ),
    )

    assert (
        twenty_million
        .position_dv01_eur
        == pytest.approx(
            ten_million
            .position_dv01_eur
            * 2.0
        )
    )


def test_scenario_pnl_has_correct_direction() -> None:
    scenarios = snapshot_scenarios(
        instrument=ITALY_10Y_BTP,
        settlement_date=date(
            2026,
            7,
            28,
        ),
        yield_percent=3.85,
        position_notional_eur=(
            10_000_000.0
        ),
        yield_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    yield_fall = scenarios.iloc[
        0
    ]

    yield_rise = scenarios.iloc[
        1
    ]

    assert (
        yield_fall[
            "position_pnl_eur"
        ]
        > 0.0
    )

    assert (
        yield_rise[
            "position_pnl_eur"
        ]
        < 0.0
    )


def test_scenario_output_contract() -> None:
    scenarios = snapshot_scenarios(
        instrument=GERMANY_10Y_BUND,
        settlement_date=date(
            2026,
            7,
            28,
        ),
        yield_percent=2.85,
        position_notional_eur=(
            5_000_000.0
        ),
    )

    assert list(
        scenarios.columns
    ) == [
        "isin",
        "yield_shock_bp",
        "shocked_yield_percent",
        "shocked_clean_price",
        "clean_price_change",
        "position_pnl_eur",
    ]

    assert len(
        scenarios
    ) == 6


def test_negative_position_notional_is_rejected() -> None:
    with pytest.raises(
        SovereignSnapshotValidationError,
        match="must not be negative",
    ):
        build_instrument_snapshot(
            instrument=GERMANY_10Y_BUND,
            german_curve=(
                create_german_curve()
            ),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            position_notional_eur=(
                -1_000_000.0
            ),
        )