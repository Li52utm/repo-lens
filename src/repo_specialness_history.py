from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import fmean, median, pstdev
from typing import Iterable

from src.repo_market_state import RepoSpecialnessResult


class RepoSpecialnessHistoryError(RuntimeError):
    pass


class RepoSpecialnessHistoryValidationError(RepoSpecialnessHistoryError):
    pass


@dataclass(frozen=True)
class SpecialnessObservation:
    isin: str
    currency: str
    repo_days: int
    quote_timestamp: datetime
    specific_repo_rate_percent: float
    gc_repo_rate_percent: float
    specialness_bp: float

    def __post_init__(self) -> None:
        normalised_isin = self.isin.strip().upper()
        normalised_currency = self.currency.strip().upper()

        if len(normalised_isin) != 12 or not normalised_isin.isalnum():
            raise RepoSpecialnessHistoryValidationError(
                "isin must contain exactly 12 alphanumeric characters."
            )

        if len(normalised_currency) != 3 or not normalised_currency.isalpha():
            raise RepoSpecialnessHistoryValidationError(
                "currency must be a three-letter alphabetic code."
            )

        if self.repo_days <= 0:
            raise RepoSpecialnessHistoryValidationError(
                "repo_days must be positive."
            )

        for field_name, value in (
            ("specific_repo_rate_percent", self.specific_repo_rate_percent),
            ("gc_repo_rate_percent", self.gc_repo_rate_percent),
            ("specialness_bp", self.specialness_bp),
        ):
            if not isfinite(value):
                raise RepoSpecialnessHistoryValidationError(
                    f"{field_name} must be finite."
                )

        expected_specialness_bp = (
            self.gc_repo_rate_percent - self.specific_repo_rate_percent
        ) * 100.0

        if abs(self.specialness_bp - expected_specialness_bp) > 1e-9:
            raise RepoSpecialnessHistoryValidationError(
                "specialness_bp must equal GC minus specific repo in basis points."
            )


@dataclass(frozen=True)
class SpecialnessHistoryAnalysis:
    isin: str
    currency: str
    repo_days: int
    current_timestamp: datetime
    current_specialness_bp: float
    historical_observation_count: int
    historical_mean_bp: float
    historical_median_bp: float
    historical_min_bp: float
    historical_max_bp: float
    historical_std_bp: float
    historical_percentile: float
    z_score: float | None
    change_vs_previous_bp: float | None
    previous_timestamp: datetime | None
    positive_specialness_share_percent: float


def observation_from_result(
    result: RepoSpecialnessResult,
) -> SpecialnessObservation:
    return SpecialnessObservation(
        isin=result.isin,
        currency=result.currency,
        repo_days=result.repo_days,
        quote_timestamp=result.specific_quote_timestamp,
        specific_repo_rate_percent=result.specific_repo_rate_percent,
        gc_repo_rate_percent=result.gc_repo_rate_percent,
        specialness_bp=result.specialness_bp,
    )


def _normalised_identity(
    observation: SpecialnessObservation,
) -> tuple[str, str, int]:
    return (
        observation.isin.strip().upper(),
        observation.currency.strip().upper(),
        observation.repo_days,
    )


def _validate_history_identity(
    *,
    observations: tuple[SpecialnessObservation, ...],
    current: SpecialnessObservation,
) -> None:
    current_identity = _normalised_identity(current)

    for observation in observations:
        if _normalised_identity(observation) != current_identity:
            raise RepoSpecialnessHistoryValidationError(
                "Historical observations and current observation must match "
                "on ISIN, currency and repo_days."
            )

        if observation.quote_timestamp > current.quote_timestamp:
            raise RepoSpecialnessHistoryValidationError(
                "Historical observations must not be later than the current "
                "observation."
            )


def analyse_specialness_history(
    *,
    historical_observations: Iterable[SpecialnessObservation],
    current_observation: SpecialnessObservation,
) -> SpecialnessHistoryAnalysis:
    history = tuple(historical_observations)

    if not history:
        raise RepoSpecialnessHistoryValidationError(
            "At least one historical observation is required."
        )

    _validate_history_identity(
        observations=history,
        current=current_observation,
    )

    ordered_history = tuple(
        sorted(
            history,
            key=lambda observation: observation.quote_timestamp,
        )
    )

    values = [
        observation.specialness_bp
        for observation in ordered_history
    ]

    mean_bp = fmean(values)
    median_bp = median(values)
    std_bp = pstdev(values)

    percentile = (
        sum(
            value <= current_observation.specialness_bp
            for value in values
        )
        / len(values)
        * 100.0
    )

    z_score: float | None = None

    if std_bp > 0.0:
        z_score = (
            current_observation.specialness_bp - mean_bp
        ) / std_bp

    previous = ordered_history[-1]

    change_vs_previous_bp = (
        current_observation.specialness_bp - previous.specialness_bp
    )

    positive_share = (
        sum(value > 0.0 for value in values)
        / len(values)
        * 100.0
    )

    return SpecialnessHistoryAnalysis(
        isin=current_observation.isin.strip().upper(),
        currency=current_observation.currency.strip().upper(),
        repo_days=current_observation.repo_days,
        current_timestamp=current_observation.quote_timestamp,
        current_specialness_bp=current_observation.specialness_bp,
        historical_observation_count=len(values),
        historical_mean_bp=mean_bp,
        historical_median_bp=median_bp,
        historical_min_bp=min(values),
        historical_max_bp=max(values),
        historical_std_bp=std_bp,
        historical_percentile=percentile,
        z_score=z_score,
        change_vs_previous_bp=change_vs_previous_bp,
        previous_timestamp=previous.quote_timestamp,
        positive_specialness_share_percent=positive_share,
    )