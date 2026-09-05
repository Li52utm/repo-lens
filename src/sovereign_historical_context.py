

from dataclasses import dataclass
from datetime import date
from enum import Enum
from math import isfinite
from statistics import median
from typing import Iterable, Sequence


class SovereignHistoricalContextError(Exception):
    """Base exception for sovereign historical-context analytics."""


class SovereignHistoricalContextValidationError(SovereignHistoricalContextError):
    """Raised when sovereign historical-context inputs are invalid."""


class HistoricalWindow(Enum):
    ONE_WEEK = ("1W", 0, 7)
    ONE_MONTH = ("1M", 1, 0)
    THREE_MONTHS = ("3M", 3, 0)
    SIX_MONTHS = ("6M", 6, 0)
    NINE_MONTHS = ("9M", 9, 0)
    ONE_YEAR = ("1Y", 12, 0)

    def __init__(
        self,
        label: str,
        months: int,
        days: int,
    ) -> None:
        self.label = label
        self.months = months
        self.days = days


DEFAULT_HISTORICAL_WINDOWS: tuple[HistoricalWindow, ...] = (
    HistoricalWindow.ONE_WEEK,
    HistoricalWindow.ONE_MONTH,
    HistoricalWindow.THREE_MONTHS,
    HistoricalWindow.SIX_MONTHS,
    HistoricalWindow.NINE_MONTHS,
    HistoricalWindow.ONE_YEAR,
)


@dataclass(frozen=True)
class SovereignHistoricalObservation:
    """
    One dated market observation for one sovereign security.

    Market fields are deliberately optional because a legitimate public or
    licensed source may provide yield without price, price without yield, or
    neither spread nor benchmark data. RepoLens must not manufacture missing
    fields.
    """

    isin: str
    observation_date: date
    source_name: str
    data_status: str
    price_per_100: float | None = None
    yield_percent: float | None = None
    benchmark_spread_bp: float | None = None
    benchmark_name: str | None = None

    def __post_init__(self) -> None:
        clean_isin = self.isin.strip().upper()

        if len(clean_isin) != 12 or not clean_isin.isalnum():
            raise SovereignHistoricalContextValidationError(
                "ISIN must be a 12-character alphanumeric identifier."
            )

        if not self.source_name.strip():
            raise SovereignHistoricalContextValidationError(
                "source_name must not be blank."
            )

        if not self.data_status.strip():
            raise SovereignHistoricalContextValidationError(
                "data_status must not be blank."
            )

        self._validate_optional_number(
            self.price_per_100,
            "price_per_100",
        )
        self._validate_optional_number(
            self.yield_percent,
            "yield_percent",
        )
        self._validate_optional_number(
            self.benchmark_spread_bp,
            "benchmark_spread_bp",
        )

        if self.price_per_100 is not None and self.price_per_100 <= 0.0:
            raise SovereignHistoricalContextValidationError(
                "price_per_100 must be positive when supplied."
            )

        if (
            self.benchmark_spread_bp is not None
            and not (self.benchmark_name or "").strip()
        ):
            raise SovereignHistoricalContextValidationError(
                "benchmark_name is required when benchmark_spread_bp is supplied."
            )

        object.__setattr__(
            self,
            "isin",
            clean_isin,
        )

    @staticmethod
    def _validate_optional_number(
        value: float | None,
        field_name: str,
    ) -> None:
        if value is not None and not isfinite(float(value)):
            raise SovereignHistoricalContextValidationError(
                f"{field_name} must be finite when supplied."
            )


@dataclass(frozen=True)
class HistoricalMetricContext:
    """
    Historical context for one market variable over one lookback window.

    Percentile is empirical and lies in [0, 100]. It is calculated using all
    valid observations in the selected window, including the current point.
    """

    current: float
    low: float
    high: float
    median: float
    percentile: float
    change_from_window_start: float
    distance_from_median: float
    observation_count: int
    first_observation_date: date
    latest_observation_date: date


@dataclass(frozen=True)
class HistoricalWindowContext:
    window: HistoricalWindow
    requested_start_date: date
    actual_first_observation_date: date
    latest_observation_date: date
    observation_count: int
    price: HistoricalMetricContext | None
    yield_percent: HistoricalMetricContext | None
    benchmark_spread_bp: HistoricalMetricContext | None


@dataclass(frozen=True)
class SovereignHistoricalContext:
    isin: str
    latest_observation_date: date
    latest_source_name: str
    latest_data_status: str
    benchmark_name: str | None
    windows: tuple[HistoricalWindowContext, ...]


def _subtract_months(
    value: date,
    months: int,
) -> date:
    if months < 0:
        raise SovereignHistoricalContextValidationError(
            "months must be non-negative."
        )

    total_months = value.year * 12 + (value.month - 1) - months
    year, zero_based_month = divmod(
        total_months,
        12,
    )
    month = zero_based_month + 1

    month_lengths = (
        31,
        29 if (
            year % 4 == 0
            and (
                year % 100 != 0
                or year % 400 == 0
            )
        ) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )

    day = min(
        value.day,
        month_lengths[month - 1],
    )

    return date(
        year,
        month,
        day,
    )


def historical_window_start_date(
    latest_date: date,
    window: HistoricalWindow,
) -> date:
    if window.months:
        return _subtract_months(
            latest_date,
            window.months,
        )

    return date.fromordinal(
        latest_date.toordinal() - window.days
    )


def _empirical_percentile(
    values: Sequence[float],
    current: float,
) -> float:
    if not values:
        raise SovereignHistoricalContextValidationError(
            "Cannot calculate percentile from an empty series."
        )

    less_than = sum(
        value < current
        for value in values
    )
    equal_to = sum(
        value == current
        for value in values
    )

    return (
        (
            less_than
            + 0.5 * equal_to
        )
        / len(values)
        * 100.0
    )


def _metric_context(
    dated_values: Sequence[tuple[date, float]],
) -> HistoricalMetricContext | None:
    if not dated_values:
        return None

    ordered = sorted(
        dated_values,
        key=lambda item: item[0],
    )

    values = [
        float(value)
        for _, value in ordered
    ]
    current = values[-1]
    centre = float(
        median(values)
    )

    return HistoricalMetricContext(
        current=current,
        low=min(values),
        high=max(values),
        median=centre,
        percentile=_empirical_percentile(
            values=values,
            current=current,
        ),
        change_from_window_start=(
            current
            - values[0]
        ),
        distance_from_median=(
            current
            - centre
        ),
        observation_count=len(values),
        first_observation_date=ordered[0][0],
        latest_observation_date=ordered[-1][0],
    )


def _validate_observation_universe(
    observations: Sequence[SovereignHistoricalObservation],
) -> tuple[SovereignHistoricalObservation, ...]:
    if not observations:
        raise SovereignHistoricalContextValidationError(
            "At least one historical observation is required."
        )

    ordered = tuple(
        sorted(
            observations,
            key=lambda item: item.observation_date,
        )
    )

    isins = {
        item.isin
        for item in ordered
    }

    if len(isins) != 1:
        raise SovereignHistoricalContextValidationError(
            "Historical context must be built for exactly one ISIN."
        )

    seen_dates: set[date] = set()

    for item in ordered:
        if item.observation_date in seen_dates:
            raise SovereignHistoricalContextValidationError(
                "Duplicate observation dates are not allowed for one ISIN."
            )
        seen_dates.add(
            item.observation_date
        )

    benchmark_names = {
        item.benchmark_name.strip()
        for item in ordered
        if item.benchmark_spread_bp is not None
        and item.benchmark_name is not None
        and item.benchmark_name.strip()
    }

    if len(benchmark_names) > 1:
        raise SovereignHistoricalContextValidationError(
            "All benchmark-spread observations must use the same benchmark."
        )

    return ordered


def _window_context(
    observations: Sequence[SovereignHistoricalObservation],
    latest_date: date,
    window: HistoricalWindow,
) -> HistoricalWindowContext:
    requested_start = historical_window_start_date(
        latest_date=latest_date,
        window=window,
    )

    selected = tuple(
        item
        for item in observations
        if requested_start <= item.observation_date <= latest_date
    )

    if not selected:
        raise SovereignHistoricalContextValidationError(
            f"No observations are available inside the {window.label} window."
        )

    price_values = tuple(
        (
            item.observation_date,
            float(item.price_per_100),
        )
        for item in selected
        if item.price_per_100 is not None
    )

    yield_values = tuple(
        (
            item.observation_date,
            float(item.yield_percent),
        )
        for item in selected
        if item.yield_percent is not None
    )

    spread_values = tuple(
        (
            item.observation_date,
            float(item.benchmark_spread_bp),
        )
        for item in selected
        if item.benchmark_spread_bp is not None
    )

    return HistoricalWindowContext(
        window=window,
        requested_start_date=requested_start,
        actual_first_observation_date=selected[0].observation_date,
        latest_observation_date=selected[-1].observation_date,
        observation_count=len(selected),
        price=_metric_context(
            price_values
        ),
        yield_percent=_metric_context(
            yield_values
        ),
        benchmark_spread_bp=_metric_context(
            spread_values
        ),
    )


def build_sovereign_historical_context(
    observations: Iterable[SovereignHistoricalObservation],
    windows: Sequence[HistoricalWindow] = DEFAULT_HISTORICAL_WINDOWS,
) -> SovereignHistoricalContext:
    """
    Build auditable historical market context for one sovereign instrument.

    The engine is provider-agnostic. It consumes already-sourced observations
    and calculates historical ranges, medians, empirical percentiles and
    changes. It does not fetch data, interpolate missing market observations or
    manufacture executable prices/yields.
    """
    ordered = _validate_observation_universe(
        tuple(
            observations
        )
    )

    if not windows:
        raise SovereignHistoricalContextValidationError(
            "At least one historical window is required."
        )

    if len(set(windows)) != len(windows):
        raise SovereignHistoricalContextValidationError(
            "Historical windows must be unique."
        )

    latest = ordered[-1]

    benchmark_names = {
        item.benchmark_name.strip()
        for item in ordered
        if item.benchmark_spread_bp is not None
        and item.benchmark_name is not None
        and item.benchmark_name.strip()
    }

    benchmark_name = (
        next(
            iter(
                benchmark_names
            )
        )
        if benchmark_names
        else None
    )

    contexts = tuple(
        _window_context(
            observations=ordered,
            latest_date=latest.observation_date,
            window=window,
        )
        for window in windows
    )

    return SovereignHistoricalContext(
        isin=latest.isin,
        latest_observation_date=latest.observation_date,
        latest_source_name=latest.source_name,
        latest_data_status=latest.data_status,
        benchmark_name=benchmark_name,
        windows=contexts,
    )