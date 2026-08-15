from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Final


BASIS_POINTS_PER_PERCENT: Final[float] = 100.0


class RepoMarketStateError(RuntimeError):
    """
    Base exception for RepoLens repo-market-state analytics.
    """


class RepoMarketStateValidationError(RepoMarketStateError):
    """
    Raised when GC or specific-repo quote inputs are inconsistent.
    """


class RepoQuoteSourceType(StrEnum):
    """
    Describe the provenance of a repo-market quote.
    """

    DESK_INPUT = "DESK_INPUT"
    BROKER_INPUT = "BROKER_INPUT"
    MARKET_FEED = "MARKET_FEED"
    OFFICIAL_REFERENCE = "OFFICIAL_REFERENCE"


@dataclass(frozen=True)
class GCReference:
    """
    Define one general-collateral repo reference.

    rate_percent is expressed in percentage points.

    Example:
        2.05 means 2.05%.

    repo_days is the contractual repo term in calendar days.
    """

    currency: str
    repo_days: int
    rate_percent: float
    quote_timestamp: datetime
    source_name: str
    source_type: RepoQuoteSourceType
    basket_name: str | None = None
    venue: str | None = None

    def __post_init__(self) -> None:
        currency = self.currency.strip().upper()

        if len(currency) != 3 or not currency.isalpha():
            raise RepoMarketStateValidationError(
                "currency must be a three-letter alphabetic code."
            )

        if self.repo_days <= 0:
            raise RepoMarketStateValidationError(
                "repo_days must be positive."
            )

        if not isfinite(self.rate_percent):
            raise RepoMarketStateValidationError(
                "rate_percent must be finite."
            )

        if self.rate_percent <= -100.0:
            raise RepoMarketStateValidationError(
                "rate_percent must be greater than -100%."
            )

        if not self.source_name.strip():
            raise RepoMarketStateValidationError(
                "source_name must not be empty."
            )

        if self.basket_name is not None and not self.basket_name.strip():
            raise RepoMarketStateValidationError(
                "basket_name must be non-empty when supplied."
            )

        if self.venue is not None and not self.venue.strip():
            raise RepoMarketStateValidationError(
                "venue must be non-empty when supplied."
            )


@dataclass(frozen=True)
class SpecificRepoQuote:
    """
    Define one repo quote for a specific collateral security.

    rate_percent is expressed in percentage points.

    Example:
        1.40 means 1.40%.
    """

    isin: str
    currency: str
    repo_days: int
    rate_percent: float
    quote_timestamp: datetime
    source_name: str
    source_type: RepoQuoteSourceType
    venue: str | None = None

    def __post_init__(self) -> None:
        isin = self.isin.strip().upper()

        if len(isin) != 12 or not isin.isalnum():
            raise RepoMarketStateValidationError(
                "isin must contain exactly 12 alphanumeric characters."
            )

        currency = self.currency.strip().upper()

        if len(currency) != 3 or not currency.isalpha():
            raise RepoMarketStateValidationError(
                "currency must be a three-letter alphabetic code."
            )

        if self.repo_days <= 0:
            raise RepoMarketStateValidationError(
                "repo_days must be positive."
            )

        if not isfinite(self.rate_percent):
            raise RepoMarketStateValidationError(
                "rate_percent must be finite."
            )

        if self.rate_percent <= -100.0:
            raise RepoMarketStateValidationError(
                "rate_percent must be greater than -100%."
            )

        if not self.source_name.strip():
            raise RepoMarketStateValidationError(
                "source_name must not be empty."
            )

        if self.venue is not None and not self.venue.strip():
            raise RepoMarketStateValidationError(
                "venue must be non-empty when supplied."
            )


@dataclass(frozen=True)
class RepoSpecialnessResult:
    """
    Store the comparison between one specific-repo quote and GC.

    specialness_bp follows the repo-desk convention:

        GC rate - specific repo rate

    Positive values mean the specific collateral funds below GC.
    Negative values mean the specific collateral funds above GC.

    No qualitative "special" threshold is imposed here.
    """

    isin: str
    currency: str
    repo_days: int
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float
    quote_time_difference_seconds: float
    specific_quote_timestamp: datetime
    gc_quote_timestamp: datetime
    specific_source_name: str
    gc_source_name: str


def calculate_specialness_bp(
    *,
    gc_repo_rate_percent: float,
    specific_repo_rate_percent: float,
) -> float:
    """
    Return GC minus specific repo in basis points.
    """
    if not isfinite(gc_repo_rate_percent):
        raise RepoMarketStateValidationError(
            "gc_repo_rate_percent must be finite."
        )

    if not isfinite(specific_repo_rate_percent):
        raise RepoMarketStateValidationError(
            "specific_repo_rate_percent must be finite."
        )

    if gc_repo_rate_percent <= -100.0:
        raise RepoMarketStateValidationError(
            "gc_repo_rate_percent must be greater than -100%."
        )

    if specific_repo_rate_percent <= -100.0:
        raise RepoMarketStateValidationError(
            "specific_repo_rate_percent must be greater than -100%."
        )

    return (
        gc_repo_rate_percent
        - specific_repo_rate_percent
    ) * BASIS_POINTS_PER_PERCENT


def compare_specific_to_gc(
    *,
    specific_quote: SpecificRepoQuote,
    gc_reference: GCReference,
) -> RepoSpecialnessResult:
    """
    Compare a specific-repo quote with a term- and currency-matched GC quote.

    RepoLens deliberately rejects unmatched currency or repo term rather than
    silently comparing economically different funding observations.
    """
    specific_currency = specific_quote.currency.strip().upper()
    gc_currency = gc_reference.currency.strip().upper()

    if specific_currency != gc_currency:
        raise RepoMarketStateValidationError(
            "Specific repo and GC reference must use the same currency."
        )

    if specific_quote.repo_days != gc_reference.repo_days:
        raise RepoMarketStateValidationError(
            "Specific repo and GC reference must use the same repo_days."
        )

    specialness_bp = calculate_specialness_bp(
        gc_repo_rate_percent=gc_reference.rate_percent,
        specific_repo_rate_percent=specific_quote.rate_percent,
    )

    quote_time_difference_seconds = abs(
        (
            specific_quote.quote_timestamp
            - gc_reference.quote_timestamp
        ).total_seconds()
    )

    return RepoSpecialnessResult(
        isin=specific_quote.isin.strip().upper(),
        currency=specific_currency,
        repo_days=specific_quote.repo_days,
        specific_repo_rate_percent=specific_quote.rate_percent,
        gc_repo_rate_percent=gc_reference.rate_percent,
        specialness_bp=specialness_bp,
        quote_time_difference_seconds=quote_time_difference_seconds,
        specific_quote_timestamp=specific_quote.quote_timestamp,
        gc_quote_timestamp=gc_reference.quote_timestamp,
        specific_source_name=specific_quote.source_name,
        gc_source_name=gc_reference.source_name,
    )