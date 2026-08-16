from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.repo_market_state import (
    GCReference,
    RepoQuoteSourceType,
    SpecificRepoQuote,
    compare_specific_to_gc,
)
from src.repo_specialness_store import (
    REPO_SPECIALNESS_HISTORY_COLUMNS,
    RepoSpecialnessStoreValidationError,
    append_repo_specialness_record,
    history_observations_for_market,
    load_repo_specialness_records,
    stored_record_from_market_state,
)


def market_state_record(
    *,
    isin: str = "DE000BU22148",
    currency: str = "EUR",
    repo_days: int = 1,
    specialness_bp: float = 25.0,
    minute: int = 0,
):
    timestamp = datetime(
        2026,
        8,
        16,
        9,
        minute,
        tzinfo=timezone.utc,
    )

    gc_rate = 2.00
    specific_rate = (
        gc_rate
        - specialness_bp
        / 100.0
    )

    gc = GCReference(
        currency=currency,
        repo_days=repo_days,
        rate_percent=gc_rate,
        quote_timestamp=timestamp,
        source_name="Desk GC",
        source_type=RepoQuoteSourceType.DESK_INPUT,
        basket_name="EUR GC",
        venue="Desk",
    )

    specific = SpecificRepoQuote(
        isin=isin,
        currency=currency,
        repo_days=repo_days,
        rate_percent=specific_rate,
        quote_timestamp=timestamp,
        source_name="Broker quote",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
        venue="Broker",
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


def test_first_append_creates_schema_and_round_trips(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "repo_specialness.csv"
    )

    record = market_state_record()

    append_repo_specialness_record(
        record,
        path,
    )

    assert path.exists()

    header = path.read_text(
        encoding="utf-8"
    ).splitlines()[0]

    assert header.split(",") == list(
        REPO_SPECIALNESS_HISTORY_COLUMNS
    )

    loaded = load_repo_specialness_records(
        path
    )

    assert loaded == (
        record,
    )


def test_missing_file_loads_as_empty_history(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "missing.csv"
    )

    assert load_repo_specialness_records(
        path
    ) == ()


def test_multiple_records_append_in_order(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "repo_specialness.csv"
    )

    first = market_state_record(
        specialness_bp=10.0,
        minute=0,
    )

    second = market_state_record(
        specialness_bp=30.0,
        minute=5,
    )

    append_repo_specialness_record(
        first,
        path,
    )

    append_repo_specialness_record(
        second,
        path,
    )

    loaded = load_repo_specialness_records(
        path
    )

    assert loaded == (
        first,
        second,
    )


def test_exact_duplicate_quote_pair_is_rejected(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "repo_specialness.csv"
    )

    record = market_state_record()

    append_repo_specialness_record(
        record,
        path,
    )

    with pytest.raises(
        RepoSpecialnessStoreValidationError,
        match="already exists",
    ):
        append_repo_specialness_record(
            record,
            path,
        )


def test_history_filter_requires_exact_market_identity() -> None:
    records = (
        market_state_record(
            isin="DE000BU22148",
            repo_days=1,
            specialness_bp=10.0,
            minute=0,
        ),
        market_state_record(
            isin="DE000BU22148",
            repo_days=7,
            specialness_bp=20.0,
            minute=1,
        ),
        market_state_record(
            isin="FR0014018YR0",
            repo_days=1,
            specialness_bp=30.0,
            minute=2,
        ),
    )

    history = history_observations_for_market(
        records=records,
        isin="DE000BU22148",
        currency="EUR",
        repo_days=1,
    )

    assert len(
        history
    ) == 1

    assert history[
        0
    ].specialness_bp == pytest.approx(
        10.0
    )


def test_history_filter_can_exclude_current_and_future_quotes() -> None:
    records = (
        market_state_record(
            specialness_bp=10.0,
            minute=0,
        ),
        market_state_record(
            specialness_bp=20.0,
            minute=5,
        ),
        market_state_record(
            specialness_bp=30.0,
            minute=10,
        ),
    )

    cutoff = datetime(
        2026,
        8,
        16,
        9,
        10,
        tzinfo=timezone.utc,
    )

    history = history_observations_for_market(
        records=records,
        isin="DE000BU22148",
        currency="EUR",
        repo_days=1,
        before_timestamp=cutoff,
    )

    assert [
        item.specialness_bp
        for item
        in history
    ] == pytest.approx(
        [
            10.0,
            20.0,
        ]
    )


def test_market_state_record_preserves_provenance_and_cash_economics() -> None:
    record = market_state_record(
        specialness_bp=69.0,
    )

    assert record.specific_source_name == "Broker quote"
    assert record.specific_source_type == RepoQuoteSourceType.BROKER_INPUT
    assert record.specific_venue == "Broker"

    assert record.gc_source_name == "Desk GC"
    assert record.gc_source_type == RepoQuoteSourceType.DESK_INPUT
    assert record.gc_basket_name == "EUR GC"
    assert record.gc_venue == "Desk"

    assert record.purchase_price_eur == pytest.approx(
        10_000_000.0
    )

    assert record.day_count_basis == 360

    assert record.financing_benefit_vs_gc_eur is not None
    assert record.financing_benefit_vs_gc_eur > 0.0


def test_bad_csv_schema_is_rejected(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "bad.csv"
    )

    path.write_text(
        "isin,currency\nDE000BU22148,EUR\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RepoSpecialnessStoreValidationError,
        match="schema",
    ):
        load_repo_specialness_records(
            path
        )