from datetime import datetime, timezone

import pytest

from src.repo_market_state import (
    GCReference,
    RepoMarketStateValidationError,
    RepoQuoteSourceType,
    SpecificRepoQuote,
    calculate_specialness_bp,
    compare_specific_to_gc,
)


def test_specialness_is_gc_minus_specific_in_basis_points() -> None:
    assert calculate_specialness_bp(
        gc_repo_rate_percent=2.10,
        specific_repo_rate_percent=1.41,
    ) == pytest.approx(
        69.0
    )


def test_specific_repo_above_gc_has_negative_specialness() -> None:
    assert calculate_specialness_bp(
        gc_repo_rate_percent=2.10,
        specific_repo_rate_percent=2.25,
    ) == pytest.approx(
        -15.0
    )


def test_compare_specific_to_gc_preserves_provenance() -> None:
    gc = GCReference(
        currency="EUR",
        repo_days=1,
        rate_percent=2.10,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        source_name="Desk GC input",
        source_type=RepoQuoteSourceType.DESK_INPUT,
        basket_name="EUR GC",
        venue="Desk",
    )

    specific = SpecificRepoQuote(
        isin="DE000BU22148",
        currency="EUR",
        repo_days=1,
        rate_percent=1.41,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            30,
            tzinfo=timezone.utc,
        ),
        source_name="Broker quote",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
        venue="Broker",
    )

    result = compare_specific_to_gc(
        specific_quote=specific,
        gc_reference=gc,
    )

    assert result.isin == "DE000BU22148"
    assert result.currency == "EUR"
    assert result.repo_days == 1
    assert result.specialness_bp == pytest.approx(
        69.0
    )
    assert result.quote_time_difference_seconds == pytest.approx(
        30.0
    )
    assert result.specific_source_name == "Broker quote"
    assert result.gc_source_name == "Desk GC input"


def test_compare_rejects_currency_mismatch() -> None:
    gc = GCReference(
        currency="EUR",
        repo_days=1,
        rate_percent=2.10,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        source_name="EUR GC",
        source_type=RepoQuoteSourceType.DESK_INPUT,
    )

    specific = SpecificRepoQuote(
        isin="GB0000000001",
        currency="GBP",
        repo_days=1,
        rate_percent=4.20,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        source_name="Gilt repo",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
    )

    with pytest.raises(
        RepoMarketStateValidationError,
        match="same currency",
    ):
        compare_specific_to_gc(
            specific_quote=specific,
            gc_reference=gc,
        )


def test_compare_rejects_term_mismatch() -> None:
    gc = GCReference(
        currency="EUR",
        repo_days=7,
        rate_percent=2.10,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        source_name="One-week GC",
        source_type=RepoQuoteSourceType.DESK_INPUT,
    )

    specific = SpecificRepoQuote(
        isin="FR0014018YR0",
        currency="EUR",
        repo_days=1,
        rate_percent=1.80,
        quote_timestamp=datetime(
            2026,
            8,
            15,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        source_name="Overnight OAT repo",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
    )

    with pytest.raises(
        RepoMarketStateValidationError,
        match="same repo_days",
    ):
        compare_specific_to_gc(
            specific_quote=specific,
            gc_reference=gc,
        )


def test_invalid_isin_is_rejected() -> None:
    with pytest.raises(
        RepoMarketStateValidationError,
        match="12 alphanumeric",
    ):
        SpecificRepoQuote(
            isin="BAD",
            currency="EUR",
            repo_days=1,
            rate_percent=1.50,
            quote_timestamp=datetime(
                2026,
                8,
                15,
                9,
                0,
                tzinfo=timezone.utc,
            ),
            source_name="Broker quote",
            source_type=RepoQuoteSourceType.BROKER_INPUT,
        )