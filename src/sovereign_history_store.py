

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.sovereign_historical_context import (
    SovereignHistoricalObservation,
)


class SovereignHistoryStoreError(Exception):
    """Base exception for persisted sovereign historical-market observations."""


class SovereignHistoryStoreValidationError(SovereignHistoryStoreError):
    """Raised when persisted sovereign history is invalid."""


HISTORY_COLUMNS = (
    "observation_date",
    "isin",
    "source_name",
    "data_status",
    "price_per_100",
    "yield_percent",
    "benchmark_spread_bp",
    "benchmark_name",
)


def _observation_key(
    observation: SovereignHistoricalObservation,
) -> tuple[str, date, str, str | None]:
    return (
        observation.isin,
        observation.observation_date,
        observation.source_name,
        observation.benchmark_name,
    )


def observations_to_frame(
    observations: Iterable[SovereignHistoricalObservation],
) -> pd.DataFrame:
    rows = []

    for observation in observations:
        row = asdict(
            observation
        )

        rows.append(
            {
                "observation_date": row["observation_date"].isoformat(),
                "isin": row["isin"],
                "source_name": row["source_name"],
                "data_status": row["data_status"],
                "price_per_100": row["price_per_100"],
                "yield_percent": row["yield_percent"],
                "benchmark_spread_bp": row["benchmark_spread_bp"],
                "benchmark_name": row["benchmark_name"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=HISTORY_COLUMNS,
    )


def _optional_float(
    value: object,
) -> float | None:
    if pd.isna(
        value
    ):
        return None

    return float(
        value
    )


def _optional_text(
    value: object,
) -> str | None:
    if pd.isna(
        value
    ):
        return None

    clean = str(
        value
    ).strip()

    return clean or None


def frame_to_observations(
    frame: pd.DataFrame,
) -> tuple[SovereignHistoricalObservation, ...]:
    missing = set(
        HISTORY_COLUMNS
    ) - set(
        frame.columns
    )

    if missing:
        raise SovereignHistoryStoreValidationError(
            "Persisted sovereign history is missing required columns: "
            f"{sorted(missing)}"
        )

    observations: list[SovereignHistoricalObservation] = []

    for row_number, row in frame.iterrows():
        parsed_date = pd.to_datetime(
            row["observation_date"],
            errors="coerce",
        )

        if pd.isna(
            parsed_date
        ):
            raise SovereignHistoryStoreValidationError(
                f"Invalid observation_date at row {row_number}."
            )

        source_name = str(
            row["source_name"]
        ).strip()
        data_status = str(
            row["data_status"]
        ).strip()

        if not source_name:
            raise SovereignHistoryStoreValidationError(
                f"Blank source_name at row {row_number}."
            )

        if not data_status:
            raise SovereignHistoryStoreValidationError(
                f"Blank data_status at row {row_number}."
            )

        try:
            observation = SovereignHistoricalObservation(
                isin=str(
                    row["isin"]
                ),
                observation_date=pd.Timestamp(
                    parsed_date
                ).date(),
                source_name=source_name,
                data_status=data_status,
                price_per_100=_optional_float(
                    row["price_per_100"]
                ),
                yield_percent=_optional_float(
                    row["yield_percent"]
                ),
                benchmark_spread_bp=_optional_float(
                    row["benchmark_spread_bp"]
                ),
                benchmark_name=_optional_text(
                    row["benchmark_name"]
                ),
            )
        except Exception as error:
            raise SovereignHistoryStoreValidationError(
                f"Invalid sovereign history row {row_number}: {error}"
            ) from error

        observations.append(
            observation
        )

    observations.sort(
        key=lambda item: (
            item.isin,
            item.observation_date,
            item.source_name,
            item.benchmark_name or "",
        )
    )

    seen: set[tuple[str, date, str, str | None]] = set()

    for observation in observations:
        key = _observation_key(
            observation
        )

        if key in seen:
            raise SovereignHistoryStoreValidationError(
                "Duplicate persisted sovereign observation: "
                f"{observation.isin} "
                f"{observation.observation_date.isoformat()} "
                f"{observation.source_name}."
            )

        seen.add(
            key
        )

    return tuple(
        observations
    )


class SovereignHistoryStore:
    """
    CSV-backed store for sourced sovereign historical-market observations.

    The store is intentionally provider-neutral. It persists source/status
    metadata and market fields exactly as supplied by upstream adapters.

    A row is uniquely identified by:
        ISIN + observation date + source name + benchmark name

    Existing observations are never silently overwritten. Callers must either
    append genuinely new observations or explicitly replace the backing file
    outside this store after a deliberate migration/reconciliation step.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(
            path
        )

    def load(
        self,
    ) -> tuple[SovereignHistoricalObservation, ...]:
        if not self.path.exists():
            return ()

        try:
            frame = pd.read_csv(
                self.path,
                encoding="utf-8-sig",
            )
        except pd.errors.ParserError as error:
            raise SovereignHistoryStoreValidationError(
                f"Could not parse sovereign history CSV: {self.path}"
            ) from error

        return frame_to_observations(
            frame
        )

    def append(
        self,
        observations: Iterable[SovereignHistoricalObservation],
    ) -> tuple[SovereignHistoricalObservation, ...]:
        incoming = tuple(
            observations
        )

        if not incoming:
            return self.load()

        existing = self.load()

        existing_keys = {
            _observation_key(
                observation
            )
            for observation in existing
        }

        incoming_keys: set[tuple[str, date, str, str | None]] = set()

        for observation in incoming:
            key = _observation_key(
                observation
            )

            if key in incoming_keys:
                raise SovereignHistoryStoreValidationError(
                    "Duplicate observation inside append batch: "
                    f"{observation.isin} "
                    f"{observation.observation_date.isoformat()} "
                    f"{observation.source_name}."
                )

            if key in existing_keys:
                raise SovereignHistoryStoreValidationError(
                    "Observation already exists in sovereign history store: "
                    f"{observation.isin} "
                    f"{observation.observation_date.isoformat()} "
                    f"{observation.source_name}."
                )

            incoming_keys.add(
                key
            )

        combined = tuple(
            sorted(
                existing + incoming,
                key=lambda item: (
                    item.isin,
                    item.observation_date,
                    item.source_name,
                    item.benchmark_name or "",
                ),
            )
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame = observations_to_frame(
            combined
        )

        frame.to_csv(
            self.path,
            index=False,
            encoding="utf-8-sig",
        )

        return combined

    def for_isin(
        self,
        isin: str,
    ) -> tuple[SovereignHistoricalObservation, ...]:
        clean_isin = isin.strip().upper()

        return tuple(
            observation
            for observation in self.load()
            if observation.isin == clean_isin
        )
