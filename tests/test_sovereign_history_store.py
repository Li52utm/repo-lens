from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.sovereign_historical_context import (
    SovereignHistoricalObservation,
)
from src.sovereign_history_store import (
    HISTORY_COLUMNS,
    SovereignHistoryStore,
    SovereignHistoryStoreValidationError,
    frame_to_observations,
    observations_to_frame,
)


BTP_ISIN = "IT0005706285"
OAT_ISIN = "FR0014018YR0"


def observation(
    *,
    isin: str = BTP_ISIN,
    day: date = date(2026, 9, 4),
    source_name: str = "Borsa Italiana MOT",
    data_status: str = "DELAYED",
    price: float | None = 97.55,
    yield_percent: float | None = 4.14,
    spread_bp: float | None = 110.0,
    benchmark_name: str | None = "Germany 10Y",
) -> SovereignHistoricalObservation:
    return SovereignHistoricalObservation(
        isin=isin,
        observation_date=day,
        source_name=source_name,
        data_status=data_status,
        price_per_100=price,
        yield_percent=yield_percent,
        benchmark_spread_bp=spread_bp,
        benchmark_name=benchmark_name,
    )


def test_round_trip_frame_preserves_observation() -> None:
    original = observation()

    frame = observations_to_frame(
        [
            original,
        ]
    )

    restored = frame_to_observations(
        frame
    )

    assert restored == (
        original,
    )


def test_empty_conversion_has_stable_schema() -> None:
    frame = observations_to_frame(
        []
    )

    assert tuple(
        frame.columns
    ) == HISTORY_COLUMNS
    assert frame.empty


def test_missing_file_loads_as_empty(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    assert store.load() == ()


def test_append_creates_parent_directories_and_file(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "nested"
        / "market"
        / "history.csv"
    )

    store = SovereignHistoryStore(
        path
    )

    saved = store.append(
        [
            observation(),
        ]
    )

    assert path.exists()
    assert len(
        saved
    ) == 1
    assert store.load() == saved


def test_append_preserves_multiple_instruments(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    store.append(
        [
            observation(),
            observation(
                isin=OAT_ISIN,
                day=date(
                    2026,
                    9,
                    3,
                ),
                source_name="Euronext",
                data_status="DELAYED",
                price=98.10,
                yield_percent=3.40,
                spread_bp=55.0,
            ),
        ]
    )

    assert len(
        store.load()
    ) == 2
    assert len(
        store.for_isin(
            BTP_ISIN
        )
    ) == 1
    assert len(
        store.for_isin(
            OAT_ISIN
        )
    ) == 1


def test_append_rejects_existing_observation(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    item = observation()

    store.append(
        [
            item,
        ]
    )

    with pytest.raises(
        SovereignHistoryStoreValidationError,
    ):
        store.append(
            [
                item,
            ]
        )


def test_append_rejects_duplicate_inside_batch(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    item = observation()

    with pytest.raises(
        SovereignHistoryStoreValidationError,
    ):
        store.append(
            [
                item,
                item,
            ]
        )


def test_same_isin_date_can_coexist_from_distinct_sources(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    saved = store.append(
        [
            observation(
                source_name="Source A",
            ),
            observation(
                source_name="Source B",
            ),
        ]
    )

    assert len(
        saved
    ) == 2


def test_same_isin_date_source_can_coexist_for_distinct_benchmark(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    saved = store.append(
        [
            observation(
                benchmark_name="Germany 10Y",
            ),
            observation(
                benchmark_name="Germany 7Y",
                spread_bp=95.0,
            ),
        ]
    )

    assert len(
        saved
    ) == 2


def test_for_isin_is_case_insensitive(
    tmp_path: Path,
) -> None:
    store = SovereignHistoryStore(
        tmp_path / "history.csv"
    )

    store.append(
        [
            observation(),
        ]
    )

    assert len(
        store.for_isin(
            BTP_ISIN.lower()
        )
    ) == 1


def test_corrupt_date_is_rejected() -> None:
    frame = observations_to_frame(
        [
            observation(),
        ]
    )

    frame.loc[
        frame.index[0],
        "observation_date",
    ] = "not-a-date"

    with pytest.raises(
        SovereignHistoryStoreValidationError,
    ):
        frame_to_observations(
            frame
        )


def test_missing_columns_are_rejected() -> None:
    frame = observations_to_frame(
        [
            observation(),
        ]
    ).drop(
        columns=[
            "data_status",
        ]
    )

    with pytest.raises(
        SovereignHistoryStoreValidationError,
    ):
        frame_to_observations(
            frame
        )


def test_duplicate_persisted_rows_are_rejected() -> None:
    frame = observations_to_frame(
        [
            observation(),
            observation(),
        ]
    )

    with pytest.raises(
        SovereignHistoryStoreValidationError,
    ):
        frame_to_observations(
            frame
        )


def test_utf8_sig_csv_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"

    store = SovereignHistoryStore(
        path
    )

    store.append(
        [
            observation(
                source_name="Mercato pubblico",
            ),
        ]
    )

    raw = path.read_bytes()

    assert raw.startswith(
        b"\xef\xbb\xbf"
    )

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    assert frame.iloc[0][
        "source_name"
    ] == "Mercato pubblico"
