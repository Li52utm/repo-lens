from datetime import datetime, timezone

import pytest

from src.repo_market_state import (
    GCReference,
    RepoQuoteSourceType,
    SpecificRepoQuote,
    compare_specific_to_gc,
)
from src.repo_specialness_history import (
    RepoSpecialnessHistoryValidationError,
    SpecialnessObservation,
    analyse_specialness_history,
    observation_from_result,
)


def observation(
    *,
    day: int,
    specialness_bp: float,
) -> SpecialnessObservation:
    specific_rate = 2.00 - specialness_bp / 100.0

    return SpecialnessObservation(
        isin="DE000BU22148",
        currency="EUR",
        repo_days=1,
        quote_timestamp=datetime(
            2026,
            8,
            day,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        specific_repo_rate_percent=specific_rate,
        gc_repo_rate_percent=2.00,
        specialness_bp=specialness_bp,
    )


def test_history_analysis_builds_distributional_context() -> None:
    history = (
        observation(day=10, specialness_bp=10.0),
        observation(day=11, specialness_bp=20.0),
        observation(day=12, specialness_bp=30.0),
        observation(day=13, specialness_bp=40.0),
    )

    current = observation(
        day=14,
        specialness_bp=35.0,
    )

    result = analyse_specialness_history(
        historical_observations=history,
        current_observation=current,
    )

    assert result.isin == "DE000BU22148"
    assert result.currency == "EUR"
    assert result.repo_days == 1
    assert result.historical_observation_count == 4
    assert result.historical_mean_bp == pytest.approx(25.0)
    assert result.historical_median_bp == pytest.approx(25.0)
    assert result.historical_min_bp == pytest.approx(10.0)
    assert result.historical_max_bp == pytest.approx(40.0)
    assert result.historical_percentile == pytest.approx(75.0)
    assert result.change_vs_previous_bp == pytest.approx(-5.0)
    assert result.positive_specialness_share_percent == pytest.approx(100.0)
    assert result.z_score is not None
    assert result.z_score > 0.0


def test_current_observation_is_not_added_to_history_distribution() -> None:
    history = (
        observation(day=10, specialness_bp=10.0),
        observation(day=11, specialness_bp=20.0),
    )

    current = observation(
        day=12,
        specialness_bp=100.0,
    )

    result = analyse_specialness_history(
        historical_observations=history,
        current_observation=current,
    )

    assert result.historical_observation_count == 2
    assert result.historical_mean_bp == pytest.approx(15.0)
    assert result.historical_percentile == pytest.approx(100.0)


def test_zero_historical_volatility_returns_no_z_score() -> None:
    history = (
        observation(day=10, specialness_bp=25.0),
        observation(day=11, specialness_bp=25.0),
    )

    current = observation(
        day=12,
        specialness_bp=25.0,
    )

    result = analyse_specialness_history(
        historical_observations=history,
        current_observation=current,
    )

    assert result.historical_std_bp == pytest.approx(0.0)
    assert result.z_score is None


def test_negative_specialness_share_is_reported_objectively() -> None:
    history = (
        observation(day=10, specialness_bp=-10.0),
        observation(day=11, specialness_bp=20.0),
        observation(day=12, specialness_bp=-5.0),
        observation(day=13, specialness_bp=30.0),
    )

    current = observation(
        day=14,
        specialness_bp=15.0,
    )

    result = analyse_specialness_history(
        historical_observations=history,
        current_observation=current,
    )

    assert result.positive_specialness_share_percent == pytest.approx(50.0)


def test_mismatched_isin_is_rejected() -> None:
    history = (
        observation(day=10, specialness_bp=10.0),
    )

    current = SpecialnessObservation(
        isin="FR0014018YR0",
        currency="EUR",
        repo_days=1,
        quote_timestamp=datetime(
            2026,
            8,
            11,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        specific_repo_rate_percent=1.80,
        gc_repo_rate_percent=2.00,
        specialness_bp=20.0,
    )

    with pytest.raises(
        RepoSpecialnessHistoryValidationError,
        match="match on ISIN",
    ):
        analyse_specialness_history(
            historical_observations=history,
            current_observation=current,
        )


def test_mismatched_repo_term_is_rejected() -> None:
    history = (
        observation(day=10, specialness_bp=10.0),
    )

    current = SpecialnessObservation(
        isin="DE000BU22148",
        currency="EUR",
        repo_days=7,
        quote_timestamp=datetime(
            2026,
            8,
            11,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        specific_repo_rate_percent=1.80,
        gc_repo_rate_percent=2.00,
        specialness_bp=20.0,
    )

    with pytest.raises(
        RepoSpecialnessHistoryValidationError,
        match="repo_days",
    ):
        analyse_specialness_history(
            historical_observations=history,
            current_observation=current,
        )


def test_future_history_is_rejected() -> None:
    history = (
        observation(day=15, specialness_bp=10.0),
    )

    current = observation(
        day=14,
        specialness_bp=20.0,
    )

    with pytest.raises(
        RepoSpecialnessHistoryValidationError,
        match="must not be later",
    ):
        analyse_specialness_history(
            historical_observations=history,
            current_observation=current,
        )


def test_empty_history_is_rejected() -> None:
    current = observation(
        day=14,
        specialness_bp=20.0,
    )

    with pytest.raises(
        RepoSpecialnessHistoryValidationError,
        match="At least one historical observation",
    ):
        analyse_specialness_history(
            historical_observations=(),
            current_observation=current,
        )


def test_inconsistent_specialness_value_is_rejected() -> None:
    with pytest.raises(
        RepoSpecialnessHistoryValidationError,
        match="must equal GC minus specific",
    ):
        SpecialnessObservation(
            isin="DE000BU22148",
            currency="EUR",
            repo_days=1,
            quote_timestamp=datetime(
                2026,
                8,
                14,
                9,
                0,
                tzinfo=timezone.utc,
            ),
            specific_repo_rate_percent=1.90,
            gc_repo_rate_percent=2.00,
            specialness_bp=99.0,
        )


def test_structured_market_state_result_converts_to_history_observation() -> None:
    timestamp = datetime(
        2026,
        8,
        14,
        9,
        0,
        tzinfo=timezone.utc,
    )

    gc = GCReference(
        currency="EUR",
        repo_days=7,
        rate_percent=2.10,
        quote_timestamp=timestamp,
        source_name="Desk GC",
        source_type=RepoQuoteSourceType.DESK_INPUT,
    )

    specific = SpecificRepoQuote(
        isin="FR0014018YR0",
        currency="EUR",
        repo_days=7,
        rate_percent=1.80,
        quote_timestamp=timestamp,
        source_name="Broker",
        source_type=RepoQuoteSourceType.BROKER_INPUT,
    )

    market_state = compare_specific_to_gc(
        specific_quote=specific,
        gc_reference=gc,
    )

    converted = observation_from_result(
        market_state
    )

    assert converted.isin == "FR0014018YR0"
    assert converted.currency == "EUR"
    assert converted.repo_days == 7
    assert converted.specialness_bp == pytest.approx(30.0)