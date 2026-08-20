from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable

from src.repo_market_state import (
    GCReference,
    RepoClearingType,
    RepoCounterpartySegment,
    RepoQuoteSourceType,
    RepoSpecialnessResult,
    SpecificRepoQuote,
)
from src.repo_specialness_history import SpecialnessObservation


DEFAULT_REPO_SPECIALNESS_HISTORY_PATH: Final[Path] = Path(
    "data"
) / "market" / "repo_specialness_history.csv"

REPO_SPECIALNESS_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "currency",
    "repo_days",
    "specific_repo_rate_percent",
    "gc_repo_rate_percent",
    "specialness_bp",
    "specific_quote_timestamp",
    "gc_quote_timestamp",
    "specific_source_name",
    "specific_source_type",
    "specific_venue",
    "specific_clearing_type",
    "specific_counterparty_segment",
    "gc_source_name",
    "gc_source_type",
    "gc_basket_name",
    "gc_venue",
    "gc_clearing_type",
    "gc_counterparty_segment",
    "quote_time_difference_seconds",
    "purchase_price_eur",
    "day_count_basis",
    "financing_benefit_vs_gc_eur",
)


class RepoSpecialnessStoreError(RuntimeError):
    """
    Base exception for RepoLens repo-specialness persistence.
    """


class RepoSpecialnessStoreValidationError(
    RepoSpecialnessStoreError
):
    """
    Raised when a persisted repo-market observation is invalid.
    """


@dataclass(frozen=True)
class RepoSpecialnessStoredRecord:
    """
    Persist one matched specific-repo versus GC observation.

    This record keeps the quote inputs, provenance and derived RepoLens
    analytics together so historical calculations remain auditable.
    """

    isin: str
    currency: str
    repo_days: int
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float
    specific_quote_timestamp: datetime
    gc_quote_timestamp: datetime
    specific_source_name: str
    specific_source_type: RepoQuoteSourceType
    specific_venue: str | None
    specific_clearing_type: RepoClearingType
    specific_counterparty_segment: RepoCounterpartySegment
    gc_source_name: str
    gc_source_type: RepoQuoteSourceType
    gc_basket_name: str | None
    gc_venue: str | None
    gc_clearing_type: RepoClearingType
    gc_counterparty_segment: RepoCounterpartySegment
    quote_time_difference_seconds: float
    purchase_price_eur: float | None
    day_count_basis: int | None
    financing_benefit_vs_gc_eur: float | None

    def __post_init__(self) -> None:
        if len(self.isin.strip()) != 12 or not self.isin.strip().isalnum():
            raise RepoSpecialnessStoreValidationError(
                "isin must contain exactly 12 alphanumeric characters."
            )

        currency = self.currency.strip().upper()

        if len(currency) != 3 or not currency.isalpha():
            raise RepoSpecialnessStoreValidationError(
                "currency must be a three-letter alphabetic code."
            )

        if self.repo_days <= 0:
            raise RepoSpecialnessStoreValidationError(
                "repo_days must be positive."
            )

        expected_specialness = (
            self.gc_repo_rate_percent
            - self.specific_repo_rate_percent
        ) * 100.0

        if abs(
            self.specialness_bp
            - expected_specialness
        ) > 1e-8:
            raise RepoSpecialnessStoreValidationError(
                "specialness_bp must equal GC minus specific repo in basis points."
            )

        if (
            (self.purchase_price_eur is None)
            != (self.day_count_basis is None)
        ):
            raise RepoSpecialnessStoreValidationError(
                "purchase_price_eur and day_count_basis must be supplied together."
            )

        if self.day_count_basis is not None and self.day_count_basis not in {
            360,
            365,
        }:
            raise RepoSpecialnessStoreValidationError(
                "day_count_basis must be 360 or 365."
            )

        if (
            self.purchase_price_eur is None
            and self.financing_benefit_vs_gc_eur is not None
        ):
            raise RepoSpecialnessStoreValidationError(
                "financing_benefit_vs_gc_eur requires purchase_price_eur."
            )

        if self.quote_time_difference_seconds < 0.0:
            raise RepoSpecialnessStoreValidationError(
                "quote_time_difference_seconds must not be negative."
            )

    @property
    def identity_key(
        self,
    ) -> tuple[
        str,
        str,
        int,
        datetime,
        datetime,
    ]:
        """
        Return the immutable quote-pair identity used for duplicate detection.
        """
        return (
            self.isin.strip().upper(),
            self.currency.strip().upper(),
            self.repo_days,
            self.specific_quote_timestamp,
            self.gc_quote_timestamp,
        )

    def to_history_observation(
        self,
    ) -> SpecialnessObservation:
        """
        Convert the stored record into the historical analytics contract.
        """
        return SpecialnessObservation(
            isin=self.isin,
            currency=self.currency,
            repo_days=self.repo_days,
            quote_timestamp=self.specific_quote_timestamp,
            specific_repo_rate_percent=self.specific_repo_rate_percent,
            gc_repo_rate_percent=self.gc_repo_rate_percent,
            specialness_bp=self.specialness_bp,
        )


def stored_record_from_market_state(
    *,
    specific_quote: SpecificRepoQuote,
    gc_reference: GCReference,
    result: RepoSpecialnessResult,
) -> RepoSpecialnessStoredRecord:
    """
    Build one auditable persistence record from matched market-state objects.
    """
    if result.isin.strip().upper() != specific_quote.isin.strip().upper():
        raise RepoSpecialnessStoreValidationError(
            "result ISIN must match the specific repo quote."
        )

    if result.currency.strip().upper() != specific_quote.currency.strip().upper():
        raise RepoSpecialnessStoreValidationError(
            "result currency must match the specific repo quote."
        )

    if result.repo_days != specific_quote.repo_days:
        raise RepoSpecialnessStoreValidationError(
            "result repo_days must match the specific repo quote."
        )

    if specific_quote.currency.strip().upper() != gc_reference.currency.strip().upper():
        raise RepoSpecialnessStoreValidationError(
            "specific repo and GC reference currencies must match."
        )

    if specific_quote.repo_days != gc_reference.repo_days:
        raise RepoSpecialnessStoreValidationError(
            "specific repo and GC reference repo_days must match."
        )

    return RepoSpecialnessStoredRecord(
        isin=result.isin.strip().upper(),
        currency=result.currency.strip().upper(),
        repo_days=result.repo_days,
        specific_repo_rate_percent=result.specific_repo_rate_percent,
        gc_repo_rate_percent=result.gc_repo_rate_percent,
        specialness_bp=result.specialness_bp,
        specific_quote_timestamp=result.specific_quote_timestamp,
        gc_quote_timestamp=result.gc_quote_timestamp,
        specific_source_name=result.specific_source_name,
        specific_source_type=specific_quote.source_type,
        specific_venue=specific_quote.venue,
        specific_clearing_type=(
            specific_quote.clearing_type
        ),
        specific_counterparty_segment=(
            specific_quote.counterparty_segment
        ),
        gc_source_name=result.gc_source_name,
        gc_source_type=gc_reference.source_type,
        gc_basket_name=gc_reference.basket_name,
        gc_venue=gc_reference.venue,
        gc_clearing_type=(
            gc_reference.clearing_type
        ),
        gc_counterparty_segment=(
            gc_reference.counterparty_segment
        ),
        quote_time_difference_seconds=result.quote_time_difference_seconds,
        purchase_price_eur=result.purchase_price_eur,
        day_count_basis=result.day_count_basis,
        financing_benefit_vs_gc_eur=result.financing_benefit_vs_gc_eur,
    )


def _optional_float_to_text(
    value: float | None,
) -> str:
    if value is None:
        return ""
    return repr(float(value))


def _optional_int_to_text(
    value: int | None,
) -> str:
    if value is None:
        return ""
    return str(int(value))


def _record_to_row(
    record: RepoSpecialnessStoredRecord,
) -> dict[str, str]:
    return {
        "isin": record.isin.strip().upper(),
        "currency": record.currency.strip().upper(),
        "repo_days": str(record.repo_days),
        "specific_repo_rate_percent": repr(
            float(record.specific_repo_rate_percent)
        ),
        "gc_repo_rate_percent": repr(
            float(record.gc_repo_rate_percent)
        ),
        "specialness_bp": repr(
            float(record.specialness_bp)
        ),
        "specific_quote_timestamp": (
            record.specific_quote_timestamp.isoformat()
        ),
        "gc_quote_timestamp": (
            record.gc_quote_timestamp.isoformat()
        ),
        "specific_source_name": record.specific_source_name,
        "specific_source_type": record.specific_source_type.value,
        "specific_venue": record.specific_venue or "",
        "specific_clearing_type": (
            record.specific_clearing_type.value
        ),
        "specific_counterparty_segment": (
            record.specific_counterparty_segment.value
        ),
        "gc_source_name": record.gc_source_name,
        "gc_source_type": record.gc_source_type.value,
        "gc_basket_name": record.gc_basket_name or "",
        "gc_venue": record.gc_venue or "",
        "gc_clearing_type": (
            record.gc_clearing_type.value
        ),
        "gc_counterparty_segment": (
            record.gc_counterparty_segment.value
        ),
        "quote_time_difference_seconds": repr(
            float(record.quote_time_difference_seconds)
        ),
        "purchase_price_eur": _optional_float_to_text(
            record.purchase_price_eur
        ),
        "day_count_basis": _optional_int_to_text(
            record.day_count_basis
        ),
        "financing_benefit_vs_gc_eur": _optional_float_to_text(
            record.financing_benefit_vs_gc_eur
        ),
    }


def _optional_float_from_text(
    value: str,
) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _optional_int_from_text(
    value: str,
) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def _row_to_record(
    row: dict[str, str],
) -> RepoSpecialnessStoredRecord:
    missing = [
        column
        for column in REPO_SPECIALNESS_HISTORY_COLUMNS
        if column not in row
    ]

    if missing:
        raise RepoSpecialnessStoreValidationError(
            "Repo specialness history row is missing required columns: "
            + ", ".join(missing)
        )

    try:
        return RepoSpecialnessStoredRecord(
            isin=row["isin"].strip().upper(),
            currency=row["currency"].strip().upper(),
            repo_days=int(row["repo_days"]),
            specific_repo_rate_percent=float(
                row["specific_repo_rate_percent"]
            ),
            gc_repo_rate_percent=float(
                row["gc_repo_rate_percent"]
            ),
            specialness_bp=float(
                row["specialness_bp"]
            ),
            specific_quote_timestamp=datetime.fromisoformat(
                row["specific_quote_timestamp"]
            ),
            gc_quote_timestamp=datetime.fromisoformat(
                row["gc_quote_timestamp"]
            ),
            specific_source_name=row["specific_source_name"],
            specific_source_type=RepoQuoteSourceType(
                row["specific_source_type"]
            ),
            specific_venue=(
                row["specific_venue"].strip()
                or None
            ),
            specific_clearing_type=RepoClearingType(
                row["specific_clearing_type"]
            ),
            specific_counterparty_segment=RepoCounterpartySegment(
                row["specific_counterparty_segment"]
            ),
            gc_source_name=row["gc_source_name"],
            gc_source_type=RepoQuoteSourceType(
                row["gc_source_type"]
            ),
            gc_basket_name=(
                row["gc_basket_name"].strip()
                or None
            ),
            gc_venue=(
                row["gc_venue"].strip()
                or None
            ),
            gc_clearing_type=RepoClearingType(
                row["gc_clearing_type"]
            ),
            gc_counterparty_segment=RepoCounterpartySegment(
                row["gc_counterparty_segment"]
            ),
            quote_time_difference_seconds=float(
                row["quote_time_difference_seconds"]
            ),
            purchase_price_eur=_optional_float_from_text(
                row["purchase_price_eur"]
            ),
            day_count_basis=_optional_int_from_text(
                row["day_count_basis"]
            ),
            financing_benefit_vs_gc_eur=_optional_float_from_text(
                row["financing_benefit_vs_gc_eur"]
            ),
        )
    except (
        ValueError,
        TypeError,
    ) as error:
        raise RepoSpecialnessStoreValidationError(
            "Repo specialness history contains an invalid persisted value."
        ) from error


def load_repo_specialness_records(
    path: Path = DEFAULT_REPO_SPECIALNESS_HISTORY_PATH,
) -> tuple[
    RepoSpecialnessStoredRecord,
    ...,
]:
    """
    Load the full persisted repo-specialness history.

    Missing files are treated as an empty history so first-run workflows do
    not require a placeholder CSV.
    """
    if not path.exists():
        return ()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RepoSpecialnessStoreValidationError(
                "Repo specialness history CSV has no header."
            )

        if tuple(reader.fieldnames) != REPO_SPECIALNESS_HISTORY_COLUMNS:
            raise RepoSpecialnessStoreValidationError(
                "Repo specialness history CSV schema does not match the "
                "RepoLens persistence contract."
            )

        return tuple(
            _row_to_record(
                dict(row)
            )
            for row in reader
        )


def append_repo_specialness_record(
    record: RepoSpecialnessStoredRecord,
    path: Path = DEFAULT_REPO_SPECIALNESS_HISTORY_PATH,
) -> None:
    """
    Append one record, rejecting an exact duplicate quote pair.

    The function creates the parent directory and CSV header on first write.
    """
    existing = load_repo_specialness_records(
        path
    )

    if any(
        item.identity_key
        == record.identity_key
        for item
        in existing
    ):
        raise RepoSpecialnessStoreValidationError(
            "This specific-repo / GC quote pair already exists in history."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                REPO_SPECIALNESS_HISTORY_COLUMNS
            ),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            _record_to_row(
                record
            )
        )


def history_observations_for_market(
    *,
    records: Iterable[
        RepoSpecialnessStoredRecord
    ],
    isin: str,
    currency: str,
    repo_days: int,
    before_timestamp: datetime | None = None,
) -> tuple[
    SpecialnessObservation,
    ...,
]:
    """
    Return history observations for exactly one ISIN/currency/repo-term market.
    """
    normalised_isin = isin.strip().upper()
    normalised_currency = currency.strip().upper()

    matched = [
        record
        for record
        in records
        if (
            record.isin.strip().upper()
            == normalised_isin
            and record.currency.strip().upper()
            == normalised_currency
            and record.repo_days
            == repo_days
            and (
                before_timestamp is None
                or record.specific_quote_timestamp
                < before_timestamp
            )
        )
    ]

    matched.sort(
        key=lambda record: (
            record.specific_quote_timestamp
        )
    )

    return tuple(
        record.to_history_observation()
        for record
        in matched
    )