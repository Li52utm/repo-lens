from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.sovereign_historical_context import (
    SovereignHistoricalContextValidationError,
    SovereignHistoricalObservation,
)


class BtpHistoryAdapterError(Exception):
    """Base exception for BTP historical-market data adapters."""


class BtpHistoryAdapterValidationError(BtpHistoryAdapterError):
    """Raised when BTP historical-market input is invalid."""


REQUIRED_COLUMNS = {
    "observation_date",
    "isin",
    "source_name",
    "data_status",
}

OPTIONAL_MARKET_COLUMNS = {
    "price_per_100",
    "yield_percent",
    "benchmark_spread_bp",
    "benchmark_name",
}


@dataclass(frozen=True)
class BtpHistoryDatasetMetadata:
    source_name: str
    data_status: str
    first_observation_date: date
    latest_observation_date: date
    observation_count: int
    isin_count: int


@dataclass(frozen=True)
class BtpHistoryDataset:
    observations: tuple[SovereignHistoricalObservation, ...]
    metadata: BtpHistoryDatasetMetadata

    def for_isin(
        self,
        isin: str,
    ) -> tuple[SovereignHistoricalObservation, ...]:
        clean_isin = isin.strip().upper()

        return tuple(
            observation
            for observation in self.observations
            if observation.isin == clean_isin
        )


def _optional_float(
    value: object,
) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(
        value,
        str,
    ):
        clean = value.strip()

        if not clean:
            return None

        value = clean.replace(
            ",",
            ".",
        )

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise BtpHistoryAdapterValidationError(
            f"Could not parse numeric market value: {value!r}"
        ) from error

    if pd.isna(
        result
    ):
        return None

    return result


def _optional_text(
    value: object,
) -> str | None:
    if pd.isna(value):
        return None

    clean = str(
        value
    ).strip()

    return clean or None


def _parse_date(
    value: object,
) -> date:
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(
        parsed
    ):
        raise BtpHistoryAdapterValidationError(
            f"Could not parse observation date: {value!r}"
        )

    return pd.Timestamp(
        parsed
    ).date()


def _validate_btp_isin(
    isin: str,
) -> str:
    clean = isin.strip().upper()

    if (
        len(clean) != 12
        or not clean.isalnum()
    ):
        raise BtpHistoryAdapterValidationError(
            "BTP ISIN must be a 12-character alphanumeric identifier."
        )

    if not clean.startswith(
        "IT"
    ):
        raise BtpHistoryAdapterValidationError(
            "BTP historical-market adapter only accepts Italian ISINs."
        )

    return clean


def _normalise_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(
        frame.columns
    )

    if missing:
        raise BtpHistoryAdapterValidationError(
            "BTP history input is missing required columns: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise BtpHistoryAdapterValidationError(
            "BTP history input contains no observations."
        )

    normalised = frame.copy()

    for column in OPTIONAL_MARKET_COLUMNS:
        if column not in normalised.columns:
            normalised[column] = None

    return normalised


def observations_from_frame(
    frame: pd.DataFrame,
) -> tuple[SovereignHistoricalObservation, ...]:
    """
    Convert a normalised public/licensed BTP history table into RepoLens
    historical observations.

    This adapter deliberately does not scrape, backfill, interpolate or infer
    missing market fields. Upstream data collection must retain the true source
    and status of every observation.
    """
    normalised = _normalise_frame(
        frame
    )

    observations: list[SovereignHistoricalObservation] = []

    for row_number, row in normalised.iterrows():
        source_name = _optional_text(
            row["source_name"]
        )
        data_status = _optional_text(
            row["data_status"]
        )

        if not source_name:
            raise BtpHistoryAdapterValidationError(
                f"Blank source_name at row {row_number}."
            )

        if not data_status:
            raise BtpHistoryAdapterValidationError(
                f"Blank data_status at row {row_number}."
            )

        price = _optional_float(
            row["price_per_100"]
        )
        yield_percent = _optional_float(
            row["yield_percent"]
        )
        spread = _optional_float(
            row["benchmark_spread_bp"]
        )
        benchmark_name = _optional_text(
            row["benchmark_name"]
        )

        if (
            price is None
            and yield_percent is None
            and spread is None
        ):
            raise BtpHistoryAdapterValidationError(
                "Each BTP history row must contain at least one market field: "
                "price_per_100, yield_percent or benchmark_spread_bp."
            )

        try:
            observation = SovereignHistoricalObservation(
                isin=_validate_btp_isin(
                    str(
                        row["isin"]
                    )
                ),
                observation_date=_parse_date(
                    row["observation_date"]
                ),
                source_name=source_name,
                data_status=data_status,
                price_per_100=price,
                yield_percent=yield_percent,
                benchmark_spread_bp=spread,
                benchmark_name=benchmark_name,
            )
        except SovereignHistoricalContextValidationError as error:
            raise BtpHistoryAdapterValidationError(
                f"Invalid BTP history row {row_number}: {error}"
            ) from error

        observations.append(
            observation
        )

    observations.sort(
        key=lambda item: (
            item.isin,
            item.observation_date,
        )
    )

    seen: set[tuple[str, date]] = set()

    for observation in observations:
        key = (
            observation.isin,
            observation.observation_date,
        )

        if key in seen:
            raise BtpHistoryAdapterValidationError(
                "Duplicate BTP observation for the same ISIN and date: "
                f"{observation.isin} {observation.observation_date.isoformat()}."
            )

        seen.add(
            key
        )

    return tuple(
        observations
    )


def _dataset_metadata(
    observations: Iterable[SovereignHistoricalObservation],
) -> BtpHistoryDatasetMetadata:
    observations = tuple(
        observations
    )

    if not observations:
        raise BtpHistoryAdapterValidationError(
            "Cannot build BTP dataset metadata without observations."
        )

    sources = {
        observation.source_name
        for observation in observations
    }

    statuses = {
        observation.data_status
        for observation in observations
    }

    source_name = (
        next(
            iter(
                sources
            )
        )
        if len(sources) == 1
        else "MULTIPLE SOURCES"
    )

    data_status = (
        next(
            iter(
                statuses
            )
        )
        if len(statuses) == 1
        else "MIXED"
    )

    dates = [
        observation.observation_date
        for observation in observations
    ]

    return BtpHistoryDatasetMetadata(
        source_name=source_name,
        data_status=data_status,
        first_observation_date=min(
            dates
        ),
        latest_observation_date=max(
            dates
        ),
        observation_count=len(
            observations
        ),
        isin_count=len(
            {
                observation.isin
                for observation in observations
            }
        ),
    )


def dataset_from_frame(
    frame: pd.DataFrame,
) -> BtpHistoryDataset:
    observations = observations_from_frame(
        frame
    )

    return BtpHistoryDataset(
        observations=observations,
        metadata=_dataset_metadata(
            observations
        ),
    )


def load_btp_history_csv(
    input_path: str | Path,
) -> BtpHistoryDataset:
    """
    Load a normalised BTP historical-market CSV.

    Expected required columns:
        observation_date, isin, source_name, data_status

    Optional market columns:
        price_per_100, yield_percent, benchmark_spread_bp, benchmark_name

    The loader is intentionally source-neutral. A separate downloader or data
    ingestion job should populate the CSV only from a legitimate public,
    delayed, licensed or explicit desk/broker source and retain provenance.
    """
    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"BTP historical-market file does not exist: {path}"
        )

    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )
    except pd.errors.ParserError as error:
        raise BtpHistoryAdapterValidationError(
            f"Could not parse BTP historical-market CSV: {path}"
        ) from error

    return dataset_from_frame(
        frame
    )
