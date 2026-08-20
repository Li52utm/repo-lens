from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from src.repo_market_state import (
    RepoClearingType,
    RepoCounterpartySegment,
)
from src.repo_specialness_history import (
    RepoSpecialnessHistoryValidationError,
    analyse_specialness_history,
)
from src.repo_specialness_store import (
    RepoSpecialnessStoredRecord,
)


class RepoSpecialsScreenerError(RuntimeError):
    """
    Base exception for RepoLens specials-screening analytics.
    """


class RepoSpecialsScreenerValidationError(
    RepoSpecialsScreenerError
):
    """
    Raised when the specials universe cannot be analysed safely.
    """


@dataclass(frozen=True)
class RepoSpecialsScreenRow:
    """
    Latest screen state for one exact ISIN/currency/repo-term market.
    """

    isin: str
    currency: str
    repo_days: int
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float
    observation_timestamp: datetime
    quote_age_seconds: float
    quote_time_gap_seconds: float
    financing_benefit_vs_gc_eur: float | None
    purchase_price_eur: float | None
    historical_observation_count: int
    historical_median_bp: float | None
    historical_percentile: float | None
    z_score: float | None
    change_vs_previous_bp: float | None
    same_venue: bool | None
    same_clearing_type: bool | None
    same_counterparty_segment: bool | None
    gc_basket_identified: bool
    context_warning_count: int
    fully_context_matched: bool
    specific_source_name: str
    gc_source_name: str
    specific_venue: str | None
    gc_venue: str | None
    gc_basket_name: str | None


def _normalise_as_of(
    as_of: datetime,
) -> datetime:
    if as_of.tzinfo is None:
        raise RepoSpecialsScreenerValidationError(
            "as_of must be timezone-aware."
        )

    return as_of.astimezone(
        timezone.utc
    )


def _record_identity(
    record: RepoSpecialnessStoredRecord,
) -> tuple[
    str,
    str,
    int,
]:
    return (
        record.isin.strip().upper(),
        record.currency.strip().upper(),
        record.repo_days,
    )


def _optional_text_match(
    left: str | None,
    right: str | None,
) -> bool | None:
    if left is None or right is None:
        return None

    return (
        left.strip().casefold()
        == right.strip().casefold()
    )


def _optional_enum_match(
    left: object,
    right: object,
    *,
    unspecified_value: object,
) -> bool | None:
    if (
        left == unspecified_value
        or right == unspecified_value
    ):
        return None

    return left == right


def _context_quality(
    record: RepoSpecialnessStoredRecord,
) -> tuple[
    bool | None,
    bool | None,
    bool | None,
    bool,
    int,
    bool,
]:
    same_venue = _optional_text_match(
        record.specific_venue,
        record.gc_venue,
    )

    same_clearing_type = _optional_enum_match(
        record.specific_clearing_type,
        record.gc_clearing_type,
        unspecified_value=(
            RepoClearingType.UNSPECIFIED
        ),
    )

    same_counterparty_segment = _optional_enum_match(
        record.specific_counterparty_segment,
        record.gc_counterparty_segment,
        unspecified_value=(
            RepoCounterpartySegment.UNSPECIFIED
        ),
    )

    gc_basket_identified = (
        record.gc_basket_name is not None
        and bool(
            record.gc_basket_name.strip()
        )
    )

    warning_count = sum(
        (
            same_venue is not True,
            same_clearing_type is not True,
            same_counterparty_segment is not True,
            not gc_basket_identified,
        )
    )

    fully_context_matched = (
        warning_count == 0
    )

    return (
        same_venue,
        same_clearing_type,
        same_counterparty_segment,
        gc_basket_identified,
        warning_count,
        fully_context_matched,
    )


def _latest_record_per_market(
    records: Iterable[
        RepoSpecialnessStoredRecord
    ],
    *,
    as_of: datetime,
) -> dict[
    tuple[
        str,
        str,
        int,
    ],
    RepoSpecialnessStoredRecord,
]:
    latest: dict[
        tuple[
            str,
            str,
            int,
        ],
        RepoSpecialnessStoredRecord,
    ] = {}

    for record in records:
        observation_time = (
            record.specific_quote_timestamp
        )

        if observation_time.tzinfo is None:
            raise RepoSpecialsScreenerValidationError(
                "Persisted specific quote timestamps must be timezone-aware."
            )

        observation_time_utc = (
            observation_time.astimezone(
                timezone.utc
            )
        )

        if observation_time_utc > as_of:
            continue

        identity = _record_identity(
            record
        )

        previous = latest.get(
            identity
        )

        if (
            previous is None
            or observation_time
            > previous.specific_quote_timestamp
        ):
            latest[
                identity
            ] = record

    return latest


def build_repo_specials_screen(
    *,
    records: Iterable[
        RepoSpecialnessStoredRecord
    ],
    as_of: datetime,
) -> tuple[
    RepoSpecialsScreenRow,
    ...,
]:
    """
    Build a ranked specials screen from persisted matched repo observations.

    One row is returned per exact ISIN/currency/repo-term market. The latest
    observation on or before as_of is the current state. Earlier observations
    for that same market provide historical context.

    Ranking is descending by current specialness, then historical percentile,
    then most recent observation. No qualitative "special" threshold is used.
    """
    as_of_utc = _normalise_as_of(
        as_of
    )

    all_records = tuple(
        records
    )

    latest_by_market = (
        _latest_record_per_market(
            all_records,
            as_of=as_of_utc,
        )
    )

    rows: list[
        RepoSpecialsScreenRow
    ] = []

    for identity, current in (
        latest_by_market.items()
    ):
        current_time = (
            current
            .specific_quote_timestamp
            .astimezone(
                timezone.utc
            )
        )

        historical_records = tuple(
            record
            for record
            in all_records
            if (
                _record_identity(
                    record
                )
                == identity
                and record
                .specific_quote_timestamp
                < current
                .specific_quote_timestamp
            )
        )

        historical_count = 0
        historical_median_bp: float | None = None
        historical_percentile: float | None = None
        z_score: float | None = None
        change_vs_previous_bp: float | None = None

        if historical_records:
            try:
                analysis = (
                    analyse_specialness_history(
                        historical_observations=(
                            record.to_history_observation()
                            for record
                            in historical_records
                        ),
                        current_observation=(
                            current
                            .to_history_observation()
                        ),
                    )
                )
            except RepoSpecialnessHistoryValidationError as error:
                raise RepoSpecialsScreenerValidationError(
                    "RepoLens could not analyse one matched specialness history."
                ) from error

            historical_count = (
                analysis
                .historical_observation_count
            )
            historical_median_bp = (
                analysis
                .historical_median_bp
            )
            historical_percentile = (
                analysis
                .historical_percentile
            )
            z_score = (
                analysis.z_score
            )
            change_vs_previous_bp = (
                analysis
                .change_vs_previous_bp
            )

        (
            same_venue,
            same_clearing_type,
            same_counterparty_segment,
            gc_basket_identified,
            context_warning_count,
            fully_context_matched,
        ) = _context_quality(
            current
        )

        quote_age_seconds = (
            as_of_utc
            - current_time
        ).total_seconds()

        rows.append(
            RepoSpecialsScreenRow(
                isin=(
                    current.isin
                    .strip()
                    .upper()
                ),
                currency=(
                    current.currency
                    .strip()
                    .upper()
                ),
                repo_days=(
                    current.repo_days
                ),
                specific_repo_rate_percent=(
                    current
                    .specific_repo_rate_percent
                ),
                gc_repo_rate_percent=(
                    current
                    .gc_repo_rate_percent
                ),
                specialness_bp=(
                    current.specialness_bp
                ),
                observation_timestamp=(
                    current_time
                ),
                quote_age_seconds=(
                    quote_age_seconds
                ),
                quote_time_gap_seconds=(
                    current
                    .quote_time_difference_seconds
                ),
                financing_benefit_vs_gc_eur=(
                    current
                    .financing_benefit_vs_gc_eur
                ),
                purchase_price_eur=(
                    current.purchase_price_eur
                ),
                historical_observation_count=(
                    historical_count
                ),
                historical_median_bp=(
                    historical_median_bp
                ),
                historical_percentile=(
                    historical_percentile
                ),
                z_score=z_score,
                change_vs_previous_bp=(
                    change_vs_previous_bp
                ),
                same_venue=(
                    same_venue
                ),
                same_clearing_type=(
                    same_clearing_type
                ),
                same_counterparty_segment=(
                    same_counterparty_segment
                ),
                gc_basket_identified=(
                    gc_basket_identified
                ),
                context_warning_count=(
                    context_warning_count
                ),
                fully_context_matched=(
                    fully_context_matched
                ),
                specific_source_name=(
                    current
                    .specific_source_name
                ),
                gc_source_name=(
                    current.gc_source_name
                ),
                specific_venue=(
                    current.specific_venue
                ),
                gc_venue=(
                    current.gc_venue
                ),
                gc_basket_name=(
                    current.gc_basket_name
                ),
            )
        )

    def sort_key(
        row: RepoSpecialsScreenRow,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        percentile = (
            row.historical_percentile
            if row.historical_percentile
            is not None
            else -1.0
        )

        return (
            row.specialness_bp,
            percentile,
            row.observation_timestamp.timestamp(),
        )

    return tuple(
        sorted(
            rows,
            key=sort_key,
            reverse=True,
        )
    )