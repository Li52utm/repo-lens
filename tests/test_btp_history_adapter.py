from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.btp_history_adapter import (
    BtpHistoryAdapterValidationError,
    dataset_from_frame,
    load_btp_history_csv,
    observations_from_frame,
)


TEST_ISIN = "IT0005706285"


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observation_date": "2026-09-03",
                "isin": TEST_ISIN,
                "source_name": "Borsa Italiana MOT",
                "data_status": "DELAYED",
                "price_per_100": 97.42,
                "yield_percent": 4.16,
                "benchmark_spread_bp": 112.0,
                "benchmark_name": "Germany 10Y",
            },
            {
                "observation_date": "2026-09-04",
                "isin": TEST_ISIN,
                "source_name": "Borsa Italiana MOT",
                "data_status": "DELAYED",
                "price_per_100": 97.55,
                "yield_percent": 4.14,
                "benchmark_spread_bp": 110.0,
                "benchmark_name": "Germany 10Y",
            },
        ]
    )


def test_normalises_frame_to_historical_observations() -> None:
    observations = observations_from_frame(
        base_frame()
    )

    assert len(
        observations
    ) == 2
    assert observations[0].isin == TEST_ISIN
    assert observations[0].price_per_100 == pytest.approx(
        97.42
    )
    assert observations[1].yield_percent == pytest.approx(
        4.14
    )


def test_comma_decimal_strings_are_supported() -> None:
    frame = pd.DataFrame(
        [
            {
                "observation_date": "2026-09-04",
                "isin": TEST_ISIN,
                "source_name": "Borsa Italiana MOT",
                "data_status": "DELAYED",
                "price_per_100": "97,42",
                "yield_percent": "4,14",
                "benchmark_spread_bp": "110,0",
                "benchmark_name": "Germany 10Y",
            }
        ]
    )

    observations = observations_from_frame(
        frame
    )

    assert observations[0].price_per_100 == pytest.approx(
        97.42
    )
    assert observations[0].yield_percent == pytest.approx(
        4.14
    )
    assert observations[0].benchmark_spread_bp == pytest.approx(
        110.0
    )


def test_optional_market_columns_can_be_absent() -> None:
    frame = pd.DataFrame(
        [
            {
                "observation_date": "2026-09-04",
                "isin": TEST_ISIN,
                "source_name": "Official source",
                "data_status": "OFFICIAL",
                "yield_percent": 4.14,
            }
        ]
    )

    observations = observations_from_frame(
        frame
    )

    assert observations[0].yield_percent == pytest.approx(
        4.14
    )
    assert observations[0].price_per_100 is None
    assert observations[0].benchmark_spread_bp is None


def test_row_without_any_market_field_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            {
                "observation_date": "2026-09-04",
                "isin": TEST_ISIN,
                "source_name": "Official source",
                "data_status": "OFFICIAL",
            }
        ]
    )

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_non_italian_isin_is_rejected() -> None:
    frame = base_frame().iloc[
        :1
    ].copy()

    frame.loc[
        frame.index[0],
        "isin",
    ] = "FR0014018YR0"

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_duplicate_isin_date_is_rejected() -> None:
    frame = pd.concat(
        [
            base_frame().iloc[
                :1
            ],
            base_frame().iloc[
                :1
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_blank_source_is_rejected() -> None:
    frame = base_frame().iloc[
        :1
    ].copy()

    frame.loc[
        frame.index[0],
        "source_name",
    ] = " "

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_blank_data_status_is_rejected() -> None:
    frame = base_frame().iloc[
        :1
    ].copy()

    frame.loc[
        frame.index[0],
        "data_status",
    ] = ""

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_spread_requires_benchmark_name() -> None:
    frame = base_frame().iloc[
        :1
    ].copy()

    frame.loc[
        frame.index[0],
        "benchmark_name",
    ] = None

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_dataset_metadata_preserves_uniform_source_and_status() -> None:
    dataset = dataset_from_frame(
        base_frame()
    )

    assert dataset.metadata.source_name == "Borsa Italiana MOT"
    assert dataset.metadata.data_status == "DELAYED"
    assert dataset.metadata.observation_count == 2
    assert dataset.metadata.isin_count == 1
    assert dataset.metadata.first_observation_date.isoformat() == "2026-09-03"
    assert dataset.metadata.latest_observation_date.isoformat() == "2026-09-04"


def test_dataset_metadata_marks_mixed_sources_and_statuses() -> None:
    frame = base_frame()

    frame.loc[
        frame.index[1],
        "source_name",
    ] = "Another legitimate source"

    frame.loc[
        frame.index[1],
        "data_status",
    ] = "OFFICIAL"

    dataset = dataset_from_frame(
        frame
    )

    assert dataset.metadata.source_name == "MULTIPLE SOURCES"
    assert dataset.metadata.data_status == "MIXED"


def test_dataset_can_filter_one_isin() -> None:
    frame = base_frame()

    second = frame.iloc[
        :1
    ].copy()

    second.loc[
        second.index[0],
        "observation_date",
    ] = "2026-09-02"

    second.loc[
        second.index[0],
        "isin",
    ] = "IT0005692410"

    dataset = dataset_from_frame(
        pd.concat(
            [
                frame,
                second,
            ],
            ignore_index=True,
        )
    )

    assert len(
        dataset.for_isin(
            TEST_ISIN
        )
    ) == 2

    assert len(
        dataset.for_isin(
            "IT0005692410"
        )
    ) == 1


def test_missing_required_columns_are_rejected() -> None:
    frame = base_frame().drop(
        columns=[
            "source_name",
        ]
    )

    with pytest.raises(
        BtpHistoryAdapterValidationError,
    ):
        observations_from_frame(
            frame
        )


def test_load_csv_uses_utf8_sig_and_returns_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "btp_history.csv"

    base_frame().to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    dataset = load_btp_history_csv(
        path
    )

    assert dataset.metadata.observation_count == 2
    assert dataset.observations[-1].price_per_100 == pytest.approx(
        97.55
    )


def test_missing_csv_raises_file_not_found(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        load_btp_history_csv(
            tmp_path / "missing.csv"
        )
