from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Sequence

from src.sovereign_historical_context import (
    HistoricalMetricContext,
    SovereignHistoricalContext,
)


class SovereignInstrumentReportError(Exception):
    """Base exception for RepoLens sovereign instrument reports."""


class SovereignInstrumentReportValidationError(
    SovereignInstrumentReportError
):
    """Raised when report-model inputs are invalid."""


@dataclass(frozen=True)
class InstrumentReportIdentity:
    isin: str
    display_name: str
    country: str
    currency: str
    maturity_date: date
    coupon_percent: float | None = None

    def __post_init__(self) -> None:
        clean_isin = self.isin.strip().upper()

        if len(clean_isin) != 12 or not clean_isin.isalnum():
            raise SovereignInstrumentReportValidationError(
                "ISIN must be a 12-character alphanumeric identifier."
            )

        if not self.display_name.strip():
            raise SovereignInstrumentReportValidationError(
                "display_name must not be blank."
            )

        if not self.country.strip():
            raise SovereignInstrumentReportValidationError(
                "country must not be blank."
            )

        clean_currency = self.currency.strip().upper()

        if len(clean_currency) != 3 or not clean_currency.isalpha():
            raise SovereignInstrumentReportValidationError(
                "currency must be a three-letter alphabetic code."
            )

        if self.coupon_percent is not None:
            if not isfinite(
                float(
                    self.coupon_percent
                )
            ):
                raise SovereignInstrumentReportValidationError(
                    "coupon_percent must be finite when supplied."
                )

        object.__setattr__(
            self,
            "isin",
            clean_isin,
        )
        object.__setattr__(
            self,
            "currency",
            clean_currency,
        )


@dataclass(frozen=True)
class InstrumentReportMarketSnapshot:
    as_of_date: date
    source_name: str
    data_status: str
    price_per_100: float | None = None
    yield_percent: float | None = None
    benchmark_spread_bp: float | None = None
    benchmark_name: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise SovereignInstrumentReportValidationError(
                "Market source_name must not be blank."
            )

        if not self.data_status.strip():
            raise SovereignInstrumentReportValidationError(
                "Market data_status must not be blank."
            )

        supplied = (
            self.price_per_100,
            self.yield_percent,
            self.benchmark_spread_bp,
        )

        if all(
            value is None
            for value in supplied
        ):
            raise SovereignInstrumentReportValidationError(
                "Market snapshot must contain at least one market value."
            )

        for field_name, value in (
            (
                "price_per_100",
                self.price_per_100,
            ),
            (
                "yield_percent",
                self.yield_percent,
            ),
            (
                "benchmark_spread_bp",
                self.benchmark_spread_bp,
            ),
        ):
            if value is not None and not isfinite(
                float(
                    value
                )
            ):
                raise SovereignInstrumentReportValidationError(
                    f"{field_name} must be finite when supplied."
                )

        if self.price_per_100 is not None and self.price_per_100 <= 0.0:
            raise SovereignInstrumentReportValidationError(
                "price_per_100 must be positive when supplied."
            )

        if (
            self.benchmark_spread_bp is not None
            and not (self.benchmark_name or "").strip()
        ):
            raise SovereignInstrumentReportValidationError(
                "benchmark_name is required when benchmark_spread_bp is supplied."
            )


@dataclass(frozen=True)
class InstrumentReportHistoricalMetric:
    current: float
    low: float
    high: float
    median: float
    percentile: float
    change_from_window_start: float
    distance_from_median: float
    observation_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(
            self.percentile
        ) <= 100.0:
            raise SovereignInstrumentReportValidationError(
                "Historical percentile must lie between 0 and 100."
            )

        if self.observation_count <= 0:
            raise SovereignInstrumentReportValidationError(
                "Historical observation_count must be positive."
            )


@dataclass(frozen=True)
class InstrumentReportHistoricalWindow:
    label: str
    requested_start_date: date
    actual_first_observation_date: date
    latest_observation_date: date
    price: InstrumentReportHistoricalMetric | None
    yield_percent: InstrumentReportHistoricalMetric | None
    benchmark_spread_bp: InstrumentReportHistoricalMetric | None


@dataclass(frozen=True)
class InstrumentReportRepoSnapshot:
    repo_days: int
    source_name: str
    data_status: str
    specific_repo_rate_percent: float | None = None
    gc_repo_rate_percent: float | None = None
    specialness_bp: float | None = None
    historical_percentile: float | None = None
    change_1d_bp: float | None = None
    change_1w_bp: float | None = None
    funding_advantage_vs_gc_eur: float | None = None
    repo_adjusted_carry_eur: float | None = None

    def __post_init__(self) -> None:
        if self.repo_days <= 0:
            raise SovereignInstrumentReportValidationError(
                "repo_days must be positive."
            )

        if not self.source_name.strip():
            raise SovereignInstrumentReportValidationError(
                "Repo source_name must not be blank."
            )

        if not self.data_status.strip():
            raise SovereignInstrumentReportValidationError(
                "Repo data_status must not be blank."
            )

        if (
            self.historical_percentile is not None
            and not 0.0 <= float(
                self.historical_percentile
            ) <= 100.0
        ):
            raise SovereignInstrumentReportValidationError(
                "Repo historical_percentile must lie between 0 and 100."
            )


@dataclass(frozen=True)
class InstrumentReportEvent:
    event_date: date
    category: str
    title: str
    relevance: str
    source_name: str

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise SovereignInstrumentReportValidationError(
                "Event category must not be blank."
            )

        if not self.title.strip():
            raise SovereignInstrumentReportValidationError(
                "Event title must not be blank."
            )

        if not self.relevance.strip():
            raise SovereignInstrumentReportValidationError(
                "Event relevance must not be blank."
            )

        if not self.source_name.strip():
            raise SovereignInstrumentReportValidationError(
                "Event source_name must not be blank."
            )


@dataclass(frozen=True)
class InstrumentReportNarrative:
    what_happened: str
    historical_context: str
    repo_context: str | None = None
    event_risk: str | None = None
    interpretation: str | None = None
    uncertainty: str | None = None

    def __post_init__(self) -> None:
        if not self.what_happened.strip():
            raise SovereignInstrumentReportValidationError(
                "Narrative what_happened must not be blank."
            )

        if not self.historical_context.strip():
            raise SovereignInstrumentReportValidationError(
                "Narrative historical_context must not be blank."
            )


@dataclass(frozen=True)
class InstrumentReportProvenance:
    label: str
    source_name: str
    data_status: str
    as_of: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise SovereignInstrumentReportValidationError(
                "Provenance label must not be blank."
            )

        if not self.source_name.strip():
            raise SovereignInstrumentReportValidationError(
                "Provenance source_name must not be blank."
            )

        if not self.data_status.strip():
            raise SovereignInstrumentReportValidationError(
                "Provenance data_status must not be blank."
            )

        if not self.as_of.strip():
            raise SovereignInstrumentReportValidationError(
                "Provenance as_of must not be blank."
            )


@dataclass(frozen=True)
class SovereignInstrumentReportModel:
    identity: InstrumentReportIdentity
    generated_at: datetime
    market: InstrumentReportMarketSnapshot
    historical_windows: tuple[InstrumentReportHistoricalWindow, ...]
    repo: InstrumentReportRepoSnapshot | None
    events: tuple[InstrumentReportEvent, ...]
    narrative: InstrumentReportNarrative
    provenance: tuple[InstrumentReportProvenance, ...]


def _report_metric(
    metric: HistoricalMetricContext | None,
) -> InstrumentReportHistoricalMetric | None:
    if metric is None:
        return None

    return InstrumentReportHistoricalMetric(
        current=metric.current,
        low=metric.low,
        high=metric.high,
        median=metric.median,
        percentile=metric.percentile,
        change_from_window_start=metric.change_from_window_start,
        distance_from_median=metric.distance_from_median,
        observation_count=metric.observation_count,
    )


def historical_windows_from_context(
    context: SovereignHistoricalContext,
) -> tuple[InstrumentReportHistoricalWindow, ...]:
    return tuple(
        InstrumentReportHistoricalWindow(
            label=window.window.label,
            requested_start_date=window.requested_start_date,
            actual_first_observation_date=window.actual_first_observation_date,
            latest_observation_date=window.latest_observation_date,
            price=_report_metric(
                window.price
            ),
            yield_percent=_report_metric(
                window.yield_percent
            ),
            benchmark_spread_bp=_report_metric(
                window.benchmark_spread_bp
            ),
        )
        for window in context.windows
    )


def build_sovereign_instrument_report_model(
    *,
    identity: InstrumentReportIdentity,
    market: InstrumentReportMarketSnapshot,
    historical_context: SovereignHistoricalContext,
    narrative: InstrumentReportNarrative,
    generated_at: datetime,
    repo: InstrumentReportRepoSnapshot | None = None,
    events: Iterable[InstrumentReportEvent] = (),
    provenance: Sequence[InstrumentReportProvenance] = (),
) -> SovereignInstrumentReportModel:
    """
    Assemble the single source of truth used by the instrument UI and PDF.

    The report model contains no market-data fetching and no narrative
    generation. All values must already come from sourced market observations,
    explicit desk/broker inputs, verified events or RepoLens-derived analytics.
    """
    if identity.isin != historical_context.isin:
        raise SovereignInstrumentReportValidationError(
            "Report identity ISIN does not match historical context ISIN."
        )

    if market.as_of_date != historical_context.latest_observation_date:
        raise SovereignInstrumentReportValidationError(
            "Market snapshot date must match the latest historical observation date."
        )

    ordered_events = tuple(
        sorted(
            events,
            key=lambda event: (
                event.event_date,
                event.category,
                event.title,
            ),
        )
    )

    return SovereignInstrumentReportModel(
        identity=identity,
        generated_at=generated_at,
        market=market,
        historical_windows=historical_windows_from_context(
            historical_context
        ),
        repo=repo,
        events=ordered_events,
        narrative=narrative,
        provenance=tuple(
            provenance
        ),
    )
