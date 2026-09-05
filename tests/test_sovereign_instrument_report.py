from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from src.sovereign_historical_context import (
    HistoricalWindow,
    SovereignHistoricalObservation,
    build_sovereign_historical_context,
)
from src.sovereign_instrument_report import (
    InstrumentReportEvent,
    InstrumentReportIdentity,
    InstrumentReportMarketSnapshot,
    InstrumentReportNarrative,
    InstrumentReportProvenance,
    InstrumentReportRepoSnapshot,
    SovereignInstrumentReportValidationError,
    build_sovereign_instrument_report_model,
)


ISIN = "IT0005706285"


def historical_context():
    latest = date(
        2026,
        9,
        5,
    )

    observations = [
        SovereignHistoricalObservation(
            isin=ISIN,
            observation_date=latest - timedelta(
                days=5
            ),
            source_name="Borsa Italiana MOT",
            data_status="DELAYED",
            price_per_100=97.20,
            yield_percent=4.20,
            benchmark_spread_bp=112.0,
            benchmark_name="Germany 10Y",
        ),
        SovereignHistoricalObservation(
            isin=ISIN,
            observation_date=latest,
            source_name="Borsa Italiana MOT",
            data_status="DELAYED",
            price_per_100=97.55,
            yield_percent=4.14,
            benchmark_spread_bp=110.0,
            benchmark_name="Germany 10Y",
        ),
    ]

    return build_sovereign_historical_context(
        observations,
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )


def identity() -> InstrumentReportIdentity:
    return InstrumentReportIdentity(
        isin=ISIN,
        display_name="BTP 3.80% Jul-2036",
        country="Italy",
        currency="EUR",
        maturity_date=date(
            2036,
            7,
            1,
        ),
        coupon_percent=3.80,
    )


def market() -> InstrumentReportMarketSnapshot:
    return InstrumentReportMarketSnapshot(
        as_of_date=date(
            2026,
            9,
            5,
        ),
        source_name="Borsa Italiana MOT",
        data_status="DELAYED",
        price_per_100=97.55,
        yield_percent=4.14,
        benchmark_spread_bp=110.0,
        benchmark_name="Germany 10Y",
    )


def narrative() -> InstrumentReportNarrative:
    return InstrumentReportNarrative(
        what_happened=(
            "The BTP yield declined over the selected weekly window."
        ),
        historical_context=(
            "Current levels are evaluated against sourced observations only."
        ),
        repo_context=(
            "Repo context is shown separately from cash-market history."
        ),
        uncertainty=(
            "Historical observations are descriptive and are not a forecast."
        ),
    )


def test_build_report_model_from_historical_context() -> None:
    report = build_sovereign_instrument_report_model(
        identity=identity(),
        market=market(),
        historical_context=historical_context(),
        narrative=narrative(),
        generated_at=datetime(
            2026,
            9,
            5,
            10,
            30,
        ),
    )

    assert report.identity.isin == ISIN
    assert len(
        report.historical_windows
    ) == 1
    assert report.historical_windows[0].label == "1W"
    assert report.historical_windows[0].yield_percent is not None
    assert report.historical_windows[0].yield_percent.current == pytest.approx(
        4.14
    )


def test_report_model_preserves_repo_snapshot() -> None:
    repo = InstrumentReportRepoSnapshot(
        repo_days=30,
        source_name="Desk input",
        data_status="DESK INPUT",
        specific_repo_rate_percent=1.90,
        gc_repo_rate_percent=2.20,
        specialness_bp=30.0,
        historical_percentile=92.0,
        change_1d_bp=5.0,
        funding_advantage_vs_gc_eur=15000.0,
    )

    report = build_sovereign_instrument_report_model(
        identity=identity(),
        market=market(),
        historical_context=historical_context(),
        narrative=narrative(),
        generated_at=datetime(
            2026,
            9,
            5,
            10,
            30,
        ),
        repo=repo,
    )

    assert report.repo == repo
    assert report.repo.specialness_bp == pytest.approx(
        30.0
    )


def test_events_are_sorted_chronologically() -> None:
    report = build_sovereign_instrument_report_model(
        identity=identity(),
        market=market(),
        historical_context=historical_context(),
        narrative=narrative(),
        generated_at=datetime(
            2026,
            9,
            5,
            10,
            30,
        ),
        events=(
            InstrumentReportEvent(
                event_date=date(
                    2026,
                    9,
                    12,
                ),
                category="CENTRAL_BANK",
                title="ECB policy event",
                relevance="Relevant EUR rates event",
                source_name="ECB",
            ),
            InstrumentReportEvent(
                event_date=date(
                    2026,
                    9,
                    8,
                ),
                category="AUCTION",
                title="Italian government bond auction",
                relevance="Potential supply event",
                source_name="MEF",
            ),
        ),
    )

    assert report.events[0].event_date == date(
        2026,
        9,
        8,
    )


def test_provenance_is_preserved() -> None:
    item = InstrumentReportProvenance(
        label="Cash market",
        source_name="Borsa Italiana MOT",
        data_status="DELAYED",
        as_of="2026-09-05",
        notes="Public market reference.",
    )

    report = build_sovereign_instrument_report_model(
        identity=identity(),
        market=market(),
        historical_context=historical_context(),
        narrative=narrative(),
        generated_at=datetime(
            2026,
            9,
            5,
            10,
            30,
        ),
        provenance=(
            item,
        ),
    )

    assert report.provenance == (
        item,
    )


def test_identity_must_match_historical_context() -> None:
    wrong_identity = InstrumentReportIdentity(
        isin="IT0005692410",
        display_name="Other BTP",
        country="Italy",
        currency="EUR",
        maturity_date=date(
            2028,
            2,
            28,
        ),
    )

    with pytest.raises(
        SovereignInstrumentReportValidationError,
    ):
        build_sovereign_instrument_report_model(
            identity=wrong_identity,
            market=market(),
            historical_context=historical_context(),
            narrative=narrative(),
            generated_at=datetime(
                2026,
                9,
                5,
                10,
                30,
            ),
        )


def test_market_date_must_match_latest_history_date() -> None:
    stale_market = InstrumentReportMarketSnapshot(
        as_of_date=date(
            2026,
            9,
            4,
        ),
        source_name="Borsa Italiana MOT",
        data_status="DELAYED",
        yield_percent=4.15,
    )

    with pytest.raises(
        SovereignInstrumentReportValidationError,
    ):
        build_sovereign_instrument_report_model(
            identity=identity(),
            market=stale_market,
            historical_context=historical_context(),
            narrative=narrative(),
            generated_at=datetime(
                2026,
                9,
                5,
                10,
                30,
            ),
        )


def test_repo_percentile_must_be_valid() -> None:
    with pytest.raises(
        SovereignInstrumentReportValidationError,
    ):
        InstrumentReportRepoSnapshot(
            repo_days=30,
            source_name="Desk input",
            data_status="DESK INPUT",
            historical_percentile=120.0,
        )


def test_market_spread_requires_benchmark_name() -> None:
    with pytest.raises(
        SovereignInstrumentReportValidationError,
    ):
        InstrumentReportMarketSnapshot(
            as_of_date=date(
                2026,
                9,
                5,
            ),
            source_name="Public source",
            data_status="DELAYED",
            benchmark_spread_bp=110.0,
        )


def test_narrative_requires_observation_and_history_sections() -> None:
    with pytest.raises(
        SovereignInstrumentReportValidationError,
    ):
        InstrumentReportNarrative(
            what_happened=" ",
            historical_context="Historical context.",
        )
