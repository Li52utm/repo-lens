from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.sovereign_historical_context import (
    DEFAULT_HISTORICAL_WINDOWS,
    HistoricalWindow,
    SovereignHistoricalContextValidationError,
    SovereignHistoricalObservation,
    build_sovereign_historical_context,
    historical_window_start_date,
)


TEST_ISIN = "IT0005706285"


def make_observation(
    *,
    day: date,
    price: float | None = None,
    yield_percent: float | None = None,
    spread_bp: float | None = None,
    benchmark_name: str | None = None,
    source_name: str = "Official public reference",
    data_status: str = "OFFICIAL",
) -> SovereignHistoricalObservation:
    return SovereignHistoricalObservation(
        isin=TEST_ISIN,
        observation_date=day,
        source_name=source_name,
        data_status=data_status,
        price_per_100=price,
        yield_percent=yield_percent,
        benchmark_spread_bp=spread_bp,
        benchmark_name=benchmark_name,
    )


def test_default_windows_match_broker_requested_horizons() -> None:
    assert tuple(
        window.label
        for window in DEFAULT_HISTORICAL_WINDOWS
    ) == (
        "1W",
        "1M",
        "3M",
        "6M",
        "9M",
        "1Y",
    )


def test_calendar_month_windows_use_calendar_dates() -> None:
    latest = date(
        2026,
        8,
        31,
    )

    assert historical_window_start_date(
        latest,
        HistoricalWindow.ONE_MONTH,
    ) == date(
        2026,
        7,
        31,
    )

    assert historical_window_start_date(
        latest,
        HistoricalWindow.SIX_MONTHS,
    ) == date(
        2026,
        2,
        28,
    )


def test_builds_yield_range_median_percentile_and_change() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    observations = [
        make_observation(
            day=latest - timedelta(days=6),
            yield_percent=3.20,
        ),
        make_observation(
            day=latest - timedelta(days=4),
            yield_percent=3.30,
        ),
        make_observation(
            day=latest - timedelta(days=2),
            yield_percent=3.40,
        ),
        make_observation(
            day=latest,
            yield_percent=3.50,
        ),
    ]

    context = build_sovereign_historical_context(
        observations,
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    metric = context.windows[0].yield_percent

    assert metric is not None
    assert metric.low == pytest.approx(
        3.20
    )
    assert metric.high == pytest.approx(
        3.50
    )
    assert metric.median == pytest.approx(
        3.35
    )
    assert metric.change_from_window_start == pytest.approx(
        0.30
    )
    assert metric.distance_from_median == pytest.approx(
        0.15
    )
    assert metric.percentile == pytest.approx(
        87.5
    )


def test_price_and_spread_are_calculated_independently() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    observations = [
        make_observation(
            day=latest - timedelta(days=5),
            price=98.0,
            spread_bp=105.0,
            benchmark_name="Germany 10Y",
        ),
        make_observation(
            day=latest,
            price=99.0,
            spread_bp=112.0,
            benchmark_name="Germany 10Y",
        ),
    ]

    context = build_sovereign_historical_context(
        observations,
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    window = context.windows[0]

    assert window.price is not None
    assert window.price.change_from_window_start == pytest.approx(
        1.0
    )

    assert window.benchmark_spread_bp is not None
    assert window.benchmark_spread_bp.change_from_window_start == pytest.approx(
        7.0
    )
    assert context.benchmark_name == "Germany 10Y"


def test_missing_metrics_are_not_manufactured() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    context = build_sovereign_historical_context(
        [
            make_observation(
                day=latest,
                yield_percent=3.50,
            )
        ],
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    window = context.windows[0]

    assert window.yield_percent is not None
    assert window.price is None
    assert window.benchmark_spread_bp is None


def test_latest_source_and_status_are_preserved() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    context = build_sovereign_historical_context(
        [
            make_observation(
                day=latest - timedelta(days=1),
                yield_percent=3.40,
                source_name="Source A",
                data_status="OFFICIAL",
            ),
            make_observation(
                day=latest,
                yield_percent=3.50,
                source_name="Source B",
                data_status="DELAYED",
            ),
        ],
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    assert context.latest_source_name == "Source B"
    assert context.latest_data_status == "DELAYED"


def test_duplicate_dates_are_rejected() -> None:
    day = date(
        2026,
        9,
        5,
    )

    with pytest.raises(
        SovereignHistoricalContextValidationError,
    ):
        build_sovereign_historical_context(
            [
                make_observation(
                    day=day,
                    yield_percent=3.40,
                ),
                make_observation(
                    day=day,
                    yield_percent=3.50,
                ),
            ],
            windows=(
                HistoricalWindow.ONE_WEEK,
            ),
        )


def test_multiple_isins_are_rejected() -> None:
    day = date(
        2026,
        9,
        5,
    )

    other = SovereignHistoricalObservation(
        isin="FR0014018YR0",
        observation_date=day,
        source_name="Official public reference",
        data_status="OFFICIAL",
        yield_percent=3.10,
    )

    with pytest.raises(
        SovereignHistoricalContextValidationError,
    ):
        build_sovereign_historical_context(
            [
                make_observation(
                    day=day - timedelta(days=1),
                    yield_percent=3.40,
                ),
                other,
            ],
            windows=(
                HistoricalWindow.ONE_WEEK,
            ),
        )


def test_benchmark_name_required_when_spread_is_present() -> None:
    with pytest.raises(
        SovereignHistoricalContextValidationError,
    ):
        make_observation(
            day=date(
                2026,
                9,
                5,
            ),
            spread_bp=110.0,
        )


def test_mixed_benchmarks_are_rejected() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    with pytest.raises(
        SovereignHistoricalContextValidationError,
    ):
        build_sovereign_historical_context(
            [
                make_observation(
                    day=latest - timedelta(days=1),
                    spread_bp=100.0,
                    benchmark_name="Germany 10Y",
                ),
                make_observation(
                    day=latest,
                    spread_bp=101.0,
                    benchmark_name="France 10Y",
                ),
            ],
            windows=(
                HistoricalWindow.ONE_WEEK,
            ),
        )


def test_duplicate_windows_are_rejected() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    with pytest.raises(
        SovereignHistoricalContextValidationError,
    ):
        build_sovereign_historical_context(
            [
                make_observation(
                    day=latest,
                    yield_percent=3.50,
                )
            ],
            windows=(
                HistoricalWindow.ONE_WEEK,
                HistoricalWindow.ONE_WEEK,
            ),
        )


def test_percentile_uses_midrank_for_equal_values() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    observations = [
        make_observation(
            day=latest - timedelta(days=3),
            yield_percent=3.00,
        ),
        make_observation(
            day=latest - timedelta(days=2),
            yield_percent=3.50,
        ),
        make_observation(
            day=latest - timedelta(days=1),
            yield_percent=3.50,
        ),
        make_observation(
            day=latest,
            yield_percent=3.50,
        ),
    ]

    context = build_sovereign_historical_context(
        observations,
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    metric = context.windows[0].yield_percent

    assert metric is not None
    assert metric.percentile == pytest.approx(
        62.5
    )


def test_negative_spreads_are_valid() -> None:
    latest = date(
        2026,
        9,
        5,
    )

    context = build_sovereign_historical_context(
        [
            make_observation(
                day=latest - timedelta(days=1),
                spread_bp=-5.0,
                benchmark_name="Germany 10Y",
            ),
            make_observation(
                day=latest,
                spread_bp=-3.0,
                benchmark_name="Germany 10Y",
            ),
        ],
        windows=(
            HistoricalWindow.ONE_WEEK,
        ),
    )

    spread = context.windows[0].benchmark_spread_bp

    assert spread is not None
    assert spread.current == pytest.approx(
        -3.0
    )
    assert spread.change_from_window_start == pytest.approx(
        2.0
    )