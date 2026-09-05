from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO

from pypdf import PdfReader

from src.sovereign_historical_context import (
    HistoricalWindow,
    SovereignHistoricalObservation,
    build_sovereign_historical_context,
)
from src.sovereign_instrument_pdf import (
    render_sovereign_instrument_report_pdf,
)
from src.sovereign_instrument_report import (
    InstrumentReportEvent,
    InstrumentReportIdentity,
    InstrumentReportMarketSnapshot,
    InstrumentReportNarrative,
    InstrumentReportProvenance,
    InstrumentReportRepoSnapshot,
    build_sovereign_instrument_report_model,
)


ISIN = "IT0005706285"


def build_report(
    *,
    include_repo: bool = True,
    include_events: bool = True,
):
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

    history = build_sovereign_historical_context(
        observations,
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    repo = (
        InstrumentReportRepoSnapshot(
            repo_days=30,
            source_name="Desk input",
            data_status="DESK INPUT",
            specific_repo_rate_percent=1.90,
            gc_repo_rate_percent=2.20,
            specialness_bp=30.0,
            historical_percentile=92.0,
            change_1d_bp=5.0,
            change_1w_bp=11.0,
            funding_advantage_vs_gc_eur=15000.0,
            repo_adjusted_carry_eur=22000.0,
        )
        if include_repo
        else None
    )

    events = (
        (
            InstrumentReportEvent(
                event_date=date(
                    2026,
                    9,
                    8,
                ),
                category="AUCTION",
                title="Italian government bond auction",
                relevance="Potential sovereign supply event",
                source_name="MEF",
            ),
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
        )
        if include_events
        else ()
    )

    return build_sovereign_instrument_report_model(
        identity=InstrumentReportIdentity(
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
        ),
        market=InstrumentReportMarketSnapshot(
            as_of_date=latest,
            source_name="Borsa Italiana MOT",
            data_status="DELAYED",
            price_per_100=97.55,
            yield_percent=4.14,
            benchmark_spread_bp=110.0,
            benchmark_name="Germany 10Y",
        ),
        historical_context=history,
        repo=repo,
        events=events,
        narrative=InstrumentReportNarrative(
            what_happened=(
                "The BTP yield declined over the selected weekly window."
            ),
            historical_context=(
                "Current levels are evaluated against sourced observations only."
            ),
            repo_context=(
                "Specific repo is shown against a matched GC assumption."
            ),
            event_risk=(
                "Italian auction supply and an ECB policy event are approaching."
            ),
            interpretation=(
                "Cash-market and financing context should be assessed together."
            ),
            uncertainty=(
                "Historical observations are descriptive and are not a forecast."
            ),
        ),
        provenance=(
            InstrumentReportProvenance(
                label="Cash market",
                source_name="Borsa Italiana MOT",
                data_status="DELAYED",
                as_of=latest.isoformat(),
                notes="Public/delayed market reference.",
            ),
            InstrumentReportProvenance(
                label="Repo",
                source_name="Desk input",
                data_status="DESK INPUT",
                as_of=latest.isoformat(),
                notes="Explicit input, not a RepoLens executable quote.",
            ),
        ),
        generated_at=datetime(
            2026,
            9,
            5,
            10,
            30,
        ),
    )


def extracted_text(
    pdf_bytes: bytes,
) -> str:
    reader = PdfReader(
        BytesIO(
            pdf_bytes
        )
    )

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def test_pdf_has_valid_header_and_nontrivial_size() -> None:
    pdf = render_sovereign_instrument_report_pdf(
        build_report()
    )

    assert pdf.startswith(
        b"%PDF"
    )
    assert len(
        pdf
    ) > 5000


def test_pdf_contains_instrument_identity_and_market_snapshot() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report()
        )
    )

    assert "RepoLens Instrument Report" in text
    assert "BTP 3.80% Jul-2036" in text
    assert ISIN in text
    assert "Market snapshot" in text
    assert "4.140%" in text


def test_pdf_contains_historical_context() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report()
        )
    )

    assert "Historical trading context" in text
    assert "1W" in text
    assert "Yield low" in text
    assert "Percentile" in text


def test_pdf_contains_repo_intelligence_when_supplied() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report(
                include_repo=True
            )
        )
    )

    assert "Repo intelligence" in text
    assert "Specific repo" in text
    assert "Matched GC" in text
    assert "Specialness" in text
    assert "Funding advantage vs GC" in text


def test_pdf_omits_repo_section_when_repo_not_supplied() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report(
                include_repo=False
            )
        )
    )

    assert "Repo intelligence" not in text


def test_pdf_contains_narrative_and_uncertainty() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report()
        )
    )

    assert "RepoLens narrative" in text
    assert "What happened" in text
    assert "Uncertainty" in text
    assert "not a forecast" in text


def test_pdf_contains_events_when_supplied() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report(
                include_events=True
            )
        )
    )

    assert "Upcoming events" in text
    assert "Italian government bond auction" in text
    assert "ECB policy event" in text


def test_pdf_omits_events_when_none_supplied() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report(
                include_events=False
            )
        )
    )

    assert "Upcoming events" not in text


def test_pdf_contains_provenance_and_disclaimer() -> None:
    text = extracted_text(
        render_sovereign_instrument_report_pdf(
            build_report()
        )
    )

    assert "Data provenance" in text
    assert "Borsa Italiana MOT" in text
    assert "DESK INPUT" in text
    assert "not an executable quote or trade recommendation" in text
