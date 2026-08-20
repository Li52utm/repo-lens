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


class RepoClearingType(StrEnum):
    """
    Describe how a repo observation is cleared or settled operationally.

    UNSPECIFIED is explicit: it means RepoLens does not know the clearing
    context and therefore should not imply comparability on that dimension.
    """

    UNSPECIFIED = "UNSPECIFIED"
    BILATERAL = "BILATERAL"
    CCP_CLEARED = "CCP_CLEARED"
    TRI_PARTY = "TRI_PARTY"


class RepoCounterpartySegment(StrEnum):
    """
    Describe broad counterparty context without storing a named counterparty.
    """

    UNSPECIFIED = "UNSPECIFIED"
    DEALER_TO_DEALER = "DEALER_TO_DEALER"
    DEALER_TO_CLIENT = "DEALER_TO_CLIENT"


@dataclass(frozen=True)
class RepoComparisonContext:
    """
    Assess whether two repo observations share important market context.

    Currency and repo term are hard matching requirements elsewhere.
    Venue, clearing and counterparty segment are softer dimensions because
    legitimate desk analysis may compare observations from different venues
    or market segments. RepoLens exposes those differences rather than
    silently treating the quotes as identical market conditions.
    """

    same_venue: bool | None
    same_clearing_type: bool | None
    same_counterparty_segment: bool | None
    gc_basket_identified: bool
    warnings: tuple[str, ...]

    @property
    def is_fully_context_matched(self) -> bool:
        """
        Return True only when every optional context dimension is known/matched.
        """
        return (
            self.same_venue is True
            and self.same_clearing_type is True
            and self.same_counterparty_segment is True
            and self.gc_basket_identified
            and not self.warnings
        )


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
    clearing_type: RepoClearingType = RepoClearingType.UNSPECIFIED
    counterparty_segment: RepoCounterpartySegment = (
        RepoCounterpartySegment.UNSPECIFIED
    )

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
    clearing_type: RepoClearingType = RepoClearingType.UNSPECIFIED
    counterparty_segment: RepoCounterpartySegment = (
        RepoCounterpartySegment.UNSPECIFIED
    )

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
    purchase_price_eur: float | None
    day_count_basis: int | None
    financing_benefit_vs_gc_eur: float | None
    comparison_context: RepoComparisonContext | None = None


def _optional_context_match(
    left: object,
    right: object,
    *,
    unspecified_value: object,
) -> bool | None:
    """
    Compare one optional market-context dimension.

    None means at least one side is explicitly unspecified.
    """
    if (
        left == unspecified_value
        or right == unspecified_value
    ):
        return None

    return left == right


def _optional_text_match(
    left: str | None,
    right: str | None,
) -> bool | None:
    """
    Compare optional free-text context such as venue.
    """
    if left is None or right is None:
        return None

    return (
        left.strip().casefold()
        == right.strip().casefold()
    )


def assess_repo_comparison_context(
    *,
    specific_quote: SpecificRepoQuote,
    gc_reference: GCReference,
) -> RepoComparisonContext:
    """
    Assess softer market-microstructure differences between matched quotes.

    This function does not reject different venues or clearing arrangements.
    It makes those differences explicit so a trader can judge whether the
    observed GC-minus-specific spread is economically comparable.
    """
    same_venue = _optional_text_match(
        specific_quote.venue,
        gc_reference.venue,
    )

    same_clearing_type = _optional_context_match(
        specific_quote.clearing_type,
        gc_reference.clearing_type,
        unspecified_value=RepoClearingType.UNSPECIFIED,
    )

    same_counterparty_segment = _optional_context_match(
        specific_quote.counterparty_segment,
        gc_reference.counterparty_segment,
        unspecified_value=RepoCounterpartySegment.UNSPECIFIED,
    )

    gc_basket_identified = (
        gc_reference.basket_name is not None
        and bool(
            gc_reference.basket_name.strip()
        )
    )

    warnings: list[str] = []

    if same_venue is False:
        warnings.append(
            "Specific repo and GC reference come from different venues."
        )
    elif same_venue is None:
        warnings.append(
            "Venue is not identified for both observations."
        )

    if same_clearing_type is False:
        warnings.append(
            "Specific repo and GC reference use different clearing types."
        )
    elif same_clearing_type is None:
        warnings.append(
            "Clearing type is not identified for both observations."
        )

    if same_counterparty_segment is False:
        warnings.append(
            "Specific repo and GC reference use different counterparty segments."
        )
    elif same_counterparty_segment is None:
        warnings.append(
            "Counterparty segment is not identified for both observations."
        )

    if not gc_basket_identified:
        warnings.append(
            "GC basket or reference identity is not specified."
        )

    return RepoComparisonContext(
        same_venue=same_venue,
        same_clearing_type=same_clearing_type,
        same_counterparty_segment=same_counterparty_segment,
        gc_basket_identified=gc_basket_identified,
        warnings=tuple(
            warnings
        ),
    )


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


def calculate_financing_benefit_vs_gc_eur(
    *,
    purchase_price_eur: float,
    gc_repo_rate_percent: float,
    specific_repo_rate_percent: float,
    repo_days: int,
    day_count_basis: int,
) -> float:
    """
    Calculate the cash financing advantage of a specific repo rate versus GC.

    Positive values mean the specific collateral finances more cheaply than
    the matched GC reference over the repo term.

    The calculation uses simple money-market interest:

        purchase price
        × (GC rate - specific rate)
        × repo days / day-count basis
    """
    if not isfinite(purchase_price_eur):
        raise RepoMarketStateValidationError(
            "purchase_price_eur must be finite."
        )

    if purchase_price_eur < 0.0:
        raise RepoMarketStateValidationError(
            "purchase_price_eur must not be negative."
        )

    if repo_days <= 0:
        raise RepoMarketStateValidationError(
            "repo_days must be positive."
        )

    if day_count_basis not in {
        360,
        365,
    }:
        raise RepoMarketStateValidationError(
            "day_count_basis must be 360 or 365."
        )

    specialness_bp = calculate_specialness_bp(
        gc_repo_rate_percent=gc_repo_rate_percent,
        specific_repo_rate_percent=specific_repo_rate_percent,
    )

    rate_difference_decimal = (
        specialness_bp
        / 10_000.0
    )

    return (
        purchase_price_eur
        * rate_difference_decimal
        * repo_days
        / day_count_basis
    )


def compare_specific_to_gc(
    *,
    specific_quote: SpecificRepoQuote,
    gc_reference: GCReference,
    purchase_price_eur: float | None = None,
    day_count_basis: int | None = None,
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

    if (
        (purchase_price_eur is None)
        != (day_count_basis is None)
    ):
        raise RepoMarketStateValidationError(
            "purchase_price_eur and day_count_basis must be supplied together."
        )

    financing_benefit_vs_gc_eur: float | None = None

    if (
        purchase_price_eur is not None
        and day_count_basis is not None
    ):
        financing_benefit_vs_gc_eur = (
            calculate_financing_benefit_vs_gc_eur(
                purchase_price_eur=purchase_price_eur,
                gc_repo_rate_percent=gc_reference.rate_percent,
                specific_repo_rate_percent=specific_quote.rate_percent,
                repo_days=specific_quote.repo_days,
                day_count_basis=day_count_basis,
            )
        )

    quote_time_difference_seconds = abs(
        (
            specific_quote.quote_timestamp
            - gc_reference.quote_timestamp
        ).total_seconds()
    )

    comparison_context = (
        assess_repo_comparison_context(
            specific_quote=specific_quote,
            gc_reference=gc_reference,
        )
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
        purchase_price_eur=purchase_price_eur,
        day_count_basis=day_count_basis,
        financing_benefit_vs_gc_eur=financing_benefit_vs_gc_eur,
        comparison_context=comparison_context,
    )