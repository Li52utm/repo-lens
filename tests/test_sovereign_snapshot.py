from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.sovereign_instruments import (
    GERMANY_10Y_BUND,
    ITALY_10Y_BTP,
)
from src.sovereign_snapshot import (
    ExactSovereignMarketObservation,
    SnapshotDataStatus,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    build_registry_snapshot,
    latest_exact_market_observations_by_isin,
    load_italy_exact_market_observations,
    load_persisted_exact_market_observations,
    prepare_german_benchmark_curve,
    snapshot_scenarios,
    sovereign_curve_points,
    sovereign_curve_readiness_summary,
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

    for tenor, yield_percent in yields.items():
        rows.append(
            {
                "observation_date": "2026-07-27",
                "country": "Germany",
                "country_code": "DE",
                "tenor_years": tenor,
                "benchmark_name": f"Germany {tenor}Y",
                "yield_percent": yield_percent - 0.05,
                "source_name": "Deutsche Bundesbank",
                "source_series": f"BBSSY.TEST.{tenor}",
                "source_timestamp": (
                    "2026-07-27T18:00:00Z"
                ),
                "data_status": "OFFICIAL_DAILY",
                "business_days_stale": 0,
            }
        )

        rows.append(
            {
                "observation_date": "2026-07-28",
                "country": "Germany",
                "country_code": "DE",
                "tenor_years": tenor,
                "benchmark_name": f"Germany {tenor}Y",
                "yield_percent": yield_percent,
                "source_name": "Deutsche Bundesbank",
                "source_series": f"BBSSY.TEST.{tenor}",
                "source_timestamp": (
                    "2026-07-28T18:00:00Z"
                ),
                "data_status": "OFFICIAL_DAILY",
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
        prepared["tenor_years"]
    ) == {
        2,
        5,
        10,
        30,
    }

    ten_year = prepared.loc[
        prepared["tenor_years"]
        .eq(10)
    ].iloc[0]

    assert ten_year[
        "yield_percent"
    ] == pytest.approx(
        2.85
    )


def test_missing_curve_column_is_rejected() -> None:
    broken_curve = create_german_curve().drop(
        columns=[
            "yield_percent",
        ]
    )

    with pytest.raises(
        SovereignSnapshotValidationError,
        match="missing required columns",
    ):
        prepare_german_benchmark_curve(
            broken_curve
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
        position_notional_eur=10_000_000.0,
    )

    assert snapshot.market_data_available
    assert snapshot.data_status == (
        SnapshotDataStatus.OFFICIAL_DAILY
    )

    assert snapshot.yield_percent == pytest.approx(
        2.85
    )

    assert snapshot.spread_to_germany_bp == pytest.approx(
        0.0
    )

    assert snapshot.clean_price > 0.0
    assert snapshot.position_dv01_eur > 0.0


def test_italian_instrument_requires_explicit_yield() -> None:
    with pytest.raises(
        SovereignSnapshotValidationError,
        match="explicit instrument-level yield",
    ):
        build_instrument_snapshot(
            instrument=ITALY_10Y_BTP,
            german_curve=create_german_curve(),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            position_notional_eur=10_000_000.0,
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
        source_name="Desk input from market terminal",
    )

    snapshot = build_instrument_snapshot(
        instrument=ITALY_10Y_BTP,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=25_000_000.0,
        explicit_yield_input=yield_input,
    )

    assert snapshot.data_status == (
        SnapshotDataStatus.DESK_INPUT
    )

    assert snapshot.yield_percent == pytest.approx(
        3.85
    )

    assert (
        snapshot.german_benchmark_yield_percent
        == pytest.approx(
            2.85
        )
    )

    assert snapshot.spread_to_germany_bp == pytest.approx(
        100.0
    )

    assert snapshot.position_dv01_eur > 0.0


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
            german_curve=create_german_curve(),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_input=yield_input,
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
            german_curve=create_german_curve(),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            explicit_yield_input=yield_input,
        )


def test_registry_snapshot_marks_missing_italy_data_unavailable() -> None:
    snapshot = build_registry_snapshot(
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=10_000_000.0,
    )

    assert len(
        snapshot
    ) == 8

    germany = snapshot.loc[
        snapshot["country"]
        .eq("Germany")
    ]

    italy = snapshot.loc[
        snapshot["country"]
        .eq("Italy")
    ]

    assert germany[
        "market_data_available"
    ].all()

    assert not italy[
        "market_data_available"
    ].any()

    assert set(
        italy["data_status"]
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
        snapshot["isin"]
        .eq(
            ITALY_10Y_BTP.isin
        )
    ].iloc[0]

    assert bool(
        italy_10y[
            "market_data_available"
        ]
    )

    assert italy_10y[
        "yield_percent"
    ] == pytest.approx(
        3.85
    )

    assert italy_10y[
        "spread_to_germany_bp"
    ] == pytest.approx(
        100.0
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
            german_curve=create_german_curve(),
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


def test_position_dv01_scales_with_notional() -> None:
    ten_million = build_instrument_snapshot(
        instrument=GERMANY_10Y_BUND,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=10_000_000.0,
    )

    twenty_million = build_instrument_snapshot(
        instrument=GERMANY_10Y_BUND,
        german_curve=create_german_curve(),
        settlement_date=date(
            2026,
            7,
            28,
        ),
        position_notional_eur=20_000_000.0,
    )

    assert (
        twenty_million.position_dv01_eur
        == pytest.approx(
            ten_million.position_dv01_eur
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
        position_notional_eur=10_000_000.0,
        yield_shocks_bp=(
            -10.0,
            10.0,
        ),
    )

    yield_fall = scenarios.iloc[0]
    yield_rise = scenarios.iloc[1]

    assert yield_fall[
        "position_pnl_eur"
    ] > 0.0

    assert yield_rise[
        "position_pnl_eur"
    ] < 0.0


def test_scenario_output_contract() -> None:
    scenarios = snapshot_scenarios(
        instrument=GERMANY_10Y_BUND,
        settlement_date=date(
            2026,
            7,
            28,
        ),
        yield_percent=2.85,
        position_notional_eur=5_000_000.0,
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
            german_curve=create_german_curve(),
            settlement_date=date(
                2026,
                7,
                28,
            ),
            position_notional_eur=-1_000_000.0,
        )

def test_latest_exact_market_observation_prefers_newer_date() -> None:
    older_complete = ExactSovereignMarketObservation(
        isin=ITALY_10Y_BTP.isin,
        observation_date=date(
            2026,
            9,
            6,
        ),
        price_per_100=99.10,
        yield_percent=3.90,
        source_name="Older source",
        data_status=SnapshotDataStatus.PUBLIC_REFERENCE,
    )

    newer_price_only = ExactSovereignMarketObservation(
        isin=ITALY_10Y_BTP.isin,
        observation_date=date(
            2026,
            9,
            7,
        ),
        price_per_100=99.25,
        yield_percent=None,
        source_name="Newer source",
        data_status=SnapshotDataStatus.PUBLIC_REFERENCE,
    )

    selected = latest_exact_market_observations_by_isin(
        (
            older_complete,
            newer_price_only,
        )
    )

    assert selected[
        ITALY_10Y_BTP.isin
    ] == newer_price_only


def test_latest_exact_market_observation_same_date_prefers_complete() -> None:
    price_only = ExactSovereignMarketObservation(
        isin=ITALY_10Y_BTP.isin,
        observation_date=date(
            2026,
            9,
            7,
        ),
        price_per_100=99.25,
        yield_percent=None,
        source_name="Price source",
        data_status=SnapshotDataStatus.PUBLIC_REFERENCE,
    )

    complete = ExactSovereignMarketObservation(
        isin=ITALY_10Y_BTP.isin,
        observation_date=date(
            2026,
            9,
            7,
        ),
        price_per_100=99.25,
        yield_percent=3.82,
        source_name="Complete source",
        data_status=SnapshotDataStatus.PUBLIC_REFERENCE,
    )

    selected = latest_exact_market_observations_by_isin(
        (
            price_only,
            complete,
        )
    )

    assert selected[
        ITALY_10Y_BTP.isin
    ] == complete


def test_persisted_exact_market_loader_keeps_price_and_yield(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "sovereign_history.csv"

    history_path.write_text(
        "\n".join(
            [
                (
                    "observation_date,isin,source_name,data_status,"
                    "price_per_100,yield_percent"
                ),
                (
                    f"2026-09-07,{ITALY_10Y_BTP.isin},"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE,98.7500,3.8500"
                ),
            ]
        ),
        encoding="utf-8",
    )

    observations = load_persisted_exact_market_observations(
        history_path
    )

    assert len(
        observations
    ) == 1

    observation = observations[
        0
    ]

    assert observation.isin == ITALY_10Y_BTP.isin
    assert observation.price_per_100 == pytest.approx(
        98.75
    )
    assert observation.yield_percent == pytest.approx(
        3.85
    )
    assert observation.data_status == (
        SnapshotDataStatus.PUBLIC_REFERENCE
    )


def test_curve_points_only_use_exact_yield_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import sovereign_desk_universe

    history_path = tmp_path / "sovereign_history.csv"
    italy_market_path = tmp_path / "italy_market.csv"

    history_path.write_text(
        "\n".join(
            [
                (
                    "observation_date,isin,source_name,data_status,"
                    "price_per_100,yield_percent"
                ),
                (
                    f"2026-09-07,{ITALY_10Y_BTP.isin},"
                    "Exact source,PUBLIC_REFERENCE,98.7500,3.8500"
                ),
            ]
        ),
        encoding="utf-8",
    )

    italy_market_path.write_text(
        "\n".join(
            [
                (
                    "country_code,isin,description,maturity_date,last_price,"
                    "source_name,data_status"
                ),
                (
                    "IT,IT0000000001,Price only BTP,2031-06-01,99.1000,"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    class Country:
        value = "Italy"

    class DynamicInstrument:
        def __init__(
            self,
            *,
            isin: str,
            name: str,
            maturity: date,
        ) -> None:
            self.country = Country()
            self.isin = isin
            self.display_name = name
            self.instrument_type = "BTP"
            self.currency = "EUR"
            self.maturity_date = maturity
            self.benchmark_tenor_years = 10
            self.source_name = "Reference source"
            self.data_status = "REFERENCE"

    dynamic_universe = (
        DynamicInstrument(
            isin=ITALY_10Y_BTP.isin,
            name="Exact yield BTP",
            maturity=date(
                2036,
                7,
                1,
            ),
        ),
        DynamicInstrument(
            isin="IT0000000001",
            name="Price only BTP",
            maturity=date(
                2031,
                6,
                1,
            ),
        ),
    )

    monkeypatch.setattr(
        sovereign_desk_universe,
        "load_desk_sovereign_universe",
        lambda: dynamic_universe,
    )

    points = sovereign_curve_points(
        country="Italy",
        history_path=history_path,
        italy_market_path=italy_market_path,
        as_of_date=date(
            2026,
            9,
            7,
        ),
        minimum_curve_points=2,
        require_curve_ready=False,
    )

    assert list(
        points[
            "isin"
        ]
    ) == [
        ITALY_10Y_BTP.isin,
    ]

    assert points.iloc[
        0
    ][
        "yield_percent"
    ] == pytest.approx(
        3.85
    )


def test_curve_readiness_summary_reports_exact_yield_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import sovereign_desk_universe

    history_path = tmp_path / "sovereign_history.csv"
    italy_market_path = tmp_path / "italy_market.csv"

    history_path.write_text(
        "\n".join(
            [
                (
                    "observation_date,isin,source_name,data_status,"
                    "price_per_100,yield_percent"
                ),
                (
                    "2026-09-07,IT0000000001,Exact source,"
                    "PUBLIC_REFERENCE,99.0000,3.2000"
                ),
                (
                    "2026-09-07,IT0000000002,Exact source,"
                    "PUBLIC_REFERENCE,98.0000,3.6000"
                ),
                (
                    "2026-09-07,IT0000000003,Exact source,"
                    "PUBLIC_REFERENCE,97.0000,4.0000"
                ),
            ]
        ),
        encoding="utf-8",
    )

    italy_market_path.write_text(
        (
            "country_code,isin,description,maturity_date,last_price,"
            "source_name,data_status\n"
        ),
        encoding="utf-8",
    )

    class Country:
        value = "Italy"

    class DynamicInstrument:
        def __init__(
            self,
            *,
            isin: str,
            maturity: date,
        ) -> None:
            self.country = Country()
            self.isin = isin
            self.display_name = isin
            self.instrument_type = "BTP"
            self.currency = "EUR"
            self.maturity_date = maturity
            self.benchmark_tenor_years = 10
            self.source_name = "Reference source"
            self.data_status = "REFERENCE"

    dynamic_universe = (
        DynamicInstrument(
            isin="IT0000000001",
            maturity=date(
                2029,
                6,
                1,
            ),
        ),
        DynamicInstrument(
            isin="IT0000000002",
            maturity=date(
                2033,
                6,
                1,
            ),
        ),
        DynamicInstrument(
            isin="IT0000000003",
            maturity=date(
                2039,
                6,
                1,
            ),
        ),
    )

    monkeypatch.setattr(
        sovereign_desk_universe,
        "load_desk_sovereign_universe",
        lambda: dynamic_universe,
    )

    summary = sovereign_curve_readiness_summary(
        history_path=history_path,
        italy_market_path=italy_market_path,
        as_of_date=date(
            2026,
            9,
            7,
        ),
        minimum_curve_points=3,
    )

    italy = summary.loc[
        summary[
            "country"
        ].eq(
            "Italy"
        )
    ].iloc[
        0
    ]

    assert italy[
        "reference_instruments"
    ] == 3

    assert italy[
        "exact_yields"
    ] == 3

    assert italy[
        "eligible_curve_points"
    ] == 3

    assert bool(
        italy[
            "curve_ready"
        ]
    )

    assert italy[
        "yield_coverage_percent"
    ] == pytest.approx(
        100.0
    )


def test_curve_readiness_rejects_invalid_minimum_points() -> None:
    with pytest.raises(
        SovereignSnapshotValidationError,
        match="at least 2",
    ):
        sovereign_curve_readiness_summary(
            minimum_curve_points=1
        )

def test_italy_market_loader_reads_exact_yield_and_reference_date(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "italy_nominal_btp_market.csv"

    market_path.write_text(
        "\n".join(
            [
                (
                    "country_code,isin,description,maturity_date,last_price,"
                    "reference_price,yield_percent,observation_date,"
                    "source_name,data_status"
                ),
                (
                    "IT,IT0005707614,Btp Fx 3.15% Jun31 Eur,2031-06-01,"
                    "98.440000,98.440000,3.540000,2026-09-04,"
                    "Borsa Italiana / Skipper Informatica,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    observations = load_italy_exact_market_observations(
        market_path
    )

    assert len(
        observations
    ) == 1

    observation = observations[
        0
    ]

    assert observation.isin == "IT0005707614"
    assert observation.price_per_100 == pytest.approx(
        98.44
    )
    assert observation.yield_percent == pytest.approx(
        3.54
    )
    assert observation.observation_date == date(
        2026,
        9,
        4,
    )
    assert observation.source_name == (
        "Borsa Italiana / Skipper Informatica"
    )
    assert observation.data_status == (
        SnapshotDataStatus.PUBLIC_REFERENCE
    )


def test_italy_market_loader_prefers_reference_price_over_last_price(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "italy_nominal_btp_market.csv"

    market_path.write_text(
        "\n".join(
            [
                (
                    "country_code,isin,description,maturity_date,last_price,"
                    "reference_price,yield_percent,observation_date,"
                    "source_name,data_status"
                ),
                (
                    "IT,IT0005707614,Btp Fx 3.15% Jun31 Eur,2031-06-01,"
                    "98.500000,98.440000,3.540000,2026-09-04,"
                    "Borsa Italiana / Skipper Informatica,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    observations = load_italy_exact_market_observations(
        market_path
    )

    assert observations[
        0
    ].price_per_100 == pytest.approx(
        98.44
    )


def test_italy_market_loader_keeps_old_price_only_files_compatible(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "italy_nominal_btp_market.csv"

    market_path.write_text(
        "\n".join(
            [
                (
                    "country_code,isin,description,maturity_date,last_price,"
                    "source_name,data_status"
                ),
                (
                    "IT,IT0005707614,Btp Fx 3.15% Jun31 Eur,2031-06-01,"
                    "98.440000,Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    observations = load_italy_exact_market_observations(
        market_path
    )

    assert len(
        observations
    ) == 1

    observation = observations[
        0
    ]

    assert observation.price_per_100 == pytest.approx(
        98.44
    )
    assert observation.yield_percent is None


def test_curve_readiness_uses_italy_exact_yields_from_market_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import sovereign_desk_universe

    history_path = tmp_path / "sovereign_history.csv"
    market_path = tmp_path / "italy_nominal_btp_market.csv"

    history_path.write_text(
        (
            "observation_date,isin,source_name,data_status,"
            "price_per_100,yield_percent\n"
        ),
        encoding="utf-8",
    )

    market_path.write_text(
        "\n".join(
            [
                (
                    "country_code,isin,description,maturity_date,last_price,"
                    "reference_price,yield_percent,observation_date,"
                    "source_name,data_status"
                ),
                (
                    "IT,IT0000000001,BTP One,2029-06-01,99.000000,"
                    "99.000000,3.200000,2026-09-07,"
                    "Borsa Italiana / Skipper Informatica,PUBLIC_REFERENCE"
                ),
                (
                    "IT,IT0000000002,BTP Two,2033-06-01,98.000000,"
                    "98.000000,3.600000,2026-09-07,"
                    "Borsa Italiana / Skipper Informatica,PUBLIC_REFERENCE"
                ),
                (
                    "IT,IT0000000003,BTP Three,2039-06-01,97.000000,"
                    "97.000000,4.000000,2026-09-07,"
                    "Borsa Italiana / Skipper Informatica,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    class Country:
        value = "Italy"

    class DynamicInstrument:
        def __init__(
            self,
            *,
            isin: str,
            maturity: date,
        ) -> None:
            self.country = Country()
            self.isin = isin
            self.display_name = isin
            self.instrument_type = "BTP"
            self.currency = "EUR"
            self.maturity_date = maturity
            self.benchmark_tenor_years = 10
            self.source_name = "Reference source"
            self.data_status = "REFERENCE"

    dynamic_universe = (
        DynamicInstrument(
            isin="IT0000000001",
            maturity=date(
                2029,
                6,
                1,
            ),
        ),
        DynamicInstrument(
            isin="IT0000000002",
            maturity=date(
                2033,
                6,
                1,
            ),
        ),
        DynamicInstrument(
            isin="IT0000000003",
            maturity=date(
                2039,
                6,
                1,
            ),
        ),
    )

    monkeypatch.setattr(
        sovereign_desk_universe,
        "load_desk_sovereign_universe",
        lambda: dynamic_universe,
    )

    summary = sovereign_curve_readiness_summary(
        history_path=history_path,
        italy_market_path=market_path,
        as_of_date=date(
            2026,
            9,
            7,
        ),
        minimum_curve_points=3,
    )

    italy = summary.iloc[
        0
    ]

    assert italy[
        "exact_yields"
    ] == 3

    assert italy[
        "eligible_curve_points"
    ] == 3

    assert bool(
        italy[
            "curve_ready"
        ]
    )