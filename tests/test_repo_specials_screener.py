from datetime import datetime, timezone

import pytest

from src.repo_market_state import (
    GCReference,
    RepoClearingType,
    RepoCounterpartySegment,
    RepoQuoteSourceType,
    SpecificRepoQuote,
    compare_specific_to_gc,
)
from src.repo_specialness_store import (
    stored_record_from_market_state,
)
from src.repo_specials_screener import (
    RepoSpecialsScreenerValidationError,
    build_repo_specials_screen,
)


def stored_record(
    *,
    isin: str,
    repo_days: int,
    specialness_bp: float,
    day: int,
    hour: int = 9,
    specific_venue: str | None = "Venue A",
    gc_venue: str | None = "Venue A",
    specific_clearing: RepoClearingType = (
        RepoClearingType.CCP_CLEARED
    ),
    gc_clearing: RepoClearingType = (
        RepoClearingType.CCP_CLEARED
    ),
    specific_segment: RepoCounterpartySegment = (
        RepoCounterpartySegment.DEALER_TO_DEALER
    ),
    gc_segment: RepoCounterpartySegment = (
        RepoCounterpartySegment.DEALER_TO_DEALER
    ),
    gc_basket_name: str | None = "EUR sovereign GC",
):
    timestamp = datetime(
        2026,
        8,
        day,
        hour,
        0,
        tzinfo=timezone.utc,
    )

    gc_rate = 2.00
    specific_rate = (
        gc_rate
        - specialness_bp
        / 100.0
    )

    gc = GCReference(
        currency="EUR",
        repo_days=repo_days,
        rate_percent=gc_rate,
        quote_timestamp=timestamp,
        source_name="GC source",
        source_type=RepoQuoteSourceType.MARKET_FEED,
        basket_name=gc_basket_name,
        venue=gc_venue,
        clearing_type=gc_clearing,
        counterparty_segment=gc_segment,
    )

    specific = SpecificRepoQuote(
        isin=isin,
        currency="EUR",
        repo_days=repo_days,
        rate_percent=specific_rate,
        quote_timestamp=timestamp,
        source_name="Specific source",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
        venue=specific_venue,
        clearing_type=specific_clearing,
        counterparty_segment=specific_segment,
    )

    result = compare_specific_to_gc(
        specific_quote=specific,
        gc_reference=gc,
        purchase_price_eur=10_000_000.0,
        day_count_basis=360,
    )

    return stored_record_from_market_state(
        specific_quote=specific,
        gc_reference=gc,
        result=result,
    )


def test_screen_keeps_latest_observation_per_exact_market() -> None:
    records = (
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=10.0,
            day=18,
        ),
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=30.0,
            day=19,
        ),
        stored_record(
            isin="DE000BU22148",
            repo_days=7,
            specialness_bp=50.0,
            day=19,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert len(screen) == 2

    one_day = next(
        row
        for row in screen
        if row.repo_days == 1
    )

    assert one_day.specialness_bp == pytest.approx(
        30.0
    )
    assert one_day.historical_observation_count == 1
    assert one_day.change_vs_previous_bp == pytest.approx(
        20.0
    )


def test_screen_ranks_by_current_specialness_descending() -> None:
    records = (
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=25.0,
            day=19,
        ),
        stored_record(
            isin="FR0014018YR0",
            repo_days=1,
            specialness_bp=80.0,
            day=19,
        ),
        stored_record(
            isin="IT0005706285",
            repo_days=1,
            specialness_bp=-5.0,
            day=19,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        row.isin
        for row in screen
    ] == [
        "FR0014018YR0",
        "DE000BU22148",
        "IT0005706285",
    ]


def test_screen_calculates_historical_percentile_without_static_label() -> None:
    records = (
        stored_record(
            isin="FR0014018YR0",
            repo_days=7,
            specialness_bp=10.0,
            day=16,
        ),
        stored_record(
            isin="FR0014018YR0",
            repo_days=7,
            specialness_bp=20.0,
            day=17,
        ),
        stored_record(
            isin="FR0014018YR0",
            repo_days=7,
            specialness_bp=30.0,
            day=18,
        ),
        stored_record(
            isin="FR0014018YR0",
            repo_days=7,
            specialness_bp=25.0,
            day=19,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    row = screen[0]

    assert row.historical_observation_count == 3
    assert row.historical_median_bp == pytest.approx(
        20.0
    )
    assert row.historical_percentile == pytest.approx(
        66.6666666667
    )
    assert row.z_score is not None


def test_screen_reports_quote_age() -> None:
    records = (
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=25.0,
            day=20,
            hour=9,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert screen[0].quote_age_seconds == pytest.approx(
        5_400.0
    )


def test_future_observation_is_not_used_as_current_state() -> None:
    records = (
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=20.0,
            day=19,
        ),
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=90.0,
            day=20,
            hour=12,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert len(screen) == 1
    assert screen[0].specialness_bp == pytest.approx(
        20.0
    )


def test_context_quality_flags_mismatched_observation() -> None:
    records = (
        stored_record(
            isin="FR0014018YR0",
            repo_days=1,
            specialness_bp=40.0,
            day=19,
            specific_venue="Broker",
            gc_venue="CCP",
            specific_clearing=RepoClearingType.BILATERAL,
            gc_clearing=RepoClearingType.CCP_CLEARED,
            specific_segment=(
                RepoCounterpartySegment.DEALER_TO_CLIENT
            ),
            gc_segment=(
                RepoCounterpartySegment.DEALER_TO_DEALER
            ),
            gc_basket_name=None,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    row = screen[0]

    assert row.same_venue is False
    assert row.same_clearing_type is False
    assert row.same_counterparty_segment is False
    assert row.gc_basket_identified is False
    assert row.context_warning_count == 4
    assert not row.fully_context_matched


def test_context_quality_recognises_fully_matched_observation() -> None:
    records = (
        stored_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=40.0,
            day=19,
        ),
    )

    screen = build_repo_specials_screen(
        records=records,
        as_of=datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    row = screen[0]

    assert row.context_warning_count == 0
    assert row.fully_context_matched


def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(
        RepoSpecialsScreenerValidationError,
        match="timezone-aware",
    ):
        build_repo_specials_screen(
            records=(),
            as_of=datetime(
                2026,
                8,
                20,
                10,
                0,
            ),
        )