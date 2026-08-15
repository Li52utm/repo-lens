from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.sovereign_instrument_master import (
    BenchmarkStatus,
    InstrumentMasterValidationError,
    SovereignInstrumentMasterRecord,
    get_master_record,
    instrument_master_to_frame,
    load_instrument_master,
    primary_benchmark_for_country_tenor,
    records_for_country,
    validate_instrument_master,
)
from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
)


PROJECT_MASTER_PATH = Path(
    "data/reference/sovereign_instruments.csv"
)

EXPECTED_INSTRUMENT_COUNT = 16


def project_rows() -> list[
    dict[str, str]
]:
    """
    Read the project instrument-master rows.
    """
    with PROJECT_MASTER_PATH.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file
            )
        )


def write_rows(
    path: Path,
    rows: list[
        dict[str, str]
    ],
) -> None:
    """
    Write instrument-master rows to a temporary CSV.
    """
    if not rows:
        raise ValueError(
            "rows must not be empty."
        )

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def test_project_master_loads_full_universe() -> None:
    records = load_instrument_master()

    assert len(
        records
    ) == EXPECTED_INSTRUMENT_COUNT


def test_project_master_country_counts() -> None:
    records = load_instrument_master()

    german_records = records_for_country(
        country=SovereignCountry.GERMANY,
        records=records,
    )

    italian_records = records_for_country(
        country=SovereignCountry.ITALY,
        records=records,
    )

    french_records = records_for_country(
        country=SovereignCountry.FRANCE,
        records=records,
    )

    assert len(
        german_records
    ) == 7

    assert len(
        italian_records
    ) == 5

    assert len(
        french_records
    ) == 4


def test_project_master_contains_italian_three_year_area() -> None:
    records = load_instrument_master()

    record = get_master_record(
        isin="IT0005467482",
        records=records,
    )

    assert record.country == SovereignCountry.ITALY
    assert record.benchmark_tenor_years == 3
    assert record.instrument.coupon_frequency == 2
    assert record.instrument.annual_coupon_rate == pytest.approx(
        0.0045
    )


def test_project_master_contains_french_oat_curve() -> None:
    records = load_instrument_master()

    french_records = records_for_country(
        country=SovereignCountry.FRANCE,
        records=records,
    )

    assert {
        record.benchmark_tenor_years
        for record in french_records
    } == {
        2,
        5,
        10,
        30,
    }

    assert {
        record.instrument.isin
        for record in french_records
    } == {
        "FR001400XLW2",
        "FR001400Z2L7",
        "FR0014018YR0",
        "FR0014016CV2",
    }


def test_primary_benchmark_selection() -> None:
    records = load_instrument_master()

    instrument = primary_benchmark_for_country_tenor(
        country=SovereignCountry.GERMANY,
        benchmark_tenor_years=10,
        records=records,
    )

    assert isinstance(
        instrument,
        SovereignInstrument,
    )

    assert instrument.isin == "DE000BU2Z072"


def test_french_primary_benchmark_selection() -> None:
    records = load_instrument_master()

    instrument = primary_benchmark_for_country_tenor(
        country=SovereignCountry.FRANCE,
        benchmark_tenor_years=10,
        records=records,
    )

    assert isinstance(
        instrument,
        SovereignInstrument,
    )

    assert instrument.isin == "FR0014018YR0"


def test_master_records_convert_to_existing_bond_contract() -> None:
    records = load_instrument_master()

    record = get_master_record(
        isin="IT0005706285",
        records=records,
    )

    bond = record.instrument.to_fixed_rate_bond()

    assert bond.isin == "IT0005706285"
    assert bond.coupon_frequency == 2
    assert bond.face_value == pytest.approx(
        100.0
    )


def test_french_oat_converts_to_existing_bond_contract() -> None:
    records = load_instrument_master()

    record = get_master_record(
        isin="FR0014018YR0",
        records=records,
    )

    bond = record.instrument.to_fixed_rate_bond()

    assert bond.isin == "FR0014018YR0"
    assert bond.coupon_frequency == 1
    assert bond.face_value == pytest.approx(
        100.0
    )


def test_instrument_master_frame_contract() -> None:
    records = load_instrument_master()

    frame = instrument_master_to_frame(
        records
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert len(
        frame
    ) == EXPECTED_INSTRUMENT_COUNT

    assert {
        "isin",
        "display_name",
        "country",
        "maturity_date",
        "annual_coupon_percent",
        "benchmark_tenor_years",
        "original_maturity_years",
        "is_primary_benchmark",
        "benchmark_status",
        "source_name",
        "source_checked_date",
    }.issubset(
        frame.columns
    )

    assert set(
        frame[
            "country"
        ]
    ) == {
        SovereignCountry.GERMANY.value,
        SovereignCountry.ITALY.value,
        SovereignCountry.FRANCE.value,
    }


def test_missing_file_is_rejected(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="does not exist",
    ):
        load_instrument_master(
            missing_path
        )


def test_duplicate_isin_is_rejected(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    duplicate_row = rows[0].copy()

    rows.append(
        duplicate_row
    )

    csv_path = (
        tmp_path
        / "duplicate_isin.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="duplicate ISINs",
    ):
        load_instrument_master(
            csv_path
        )


def test_duplicate_primary_country_tenor_is_rejected(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    duplicate_primary = rows[2].copy()

    duplicate_primary[
        "isin"
    ] = "DE0000000001"

    duplicate_primary[
        "display_name"
    ] = "Test duplicate German 7Y benchmark"

    rows.append(
        duplicate_primary
    )

    csv_path = (
        tmp_path
        / "duplicate_primary.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="more than one primary benchmark",
    ):
        load_instrument_master(
            csv_path
        )


def test_invalid_boolean_is_rejected(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    rows[0][
        "is_primary_benchmark"
    ] = "yes"

    csv_path = (
        tmp_path
        / "invalid_boolean.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="must be true or false",
    ):
        load_instrument_master(
            csv_path
        )


def test_invalid_country_code_is_rejected(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    rows[0][
        "country_code"
    ] = "IT"

    csv_path = (
        tmp_path
        / "invalid_country_code.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="Germany must use country_code DE",
    ):
        load_instrument_master(
            csv_path
        )


def test_invalid_french_country_code_is_rejected(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    french_row = next(
        row
        for row in rows
        if row[
            "country"
        ] == SovereignCountry.FRANCE.value
    )

    french_row[
        "country_code"
    ] = "DE"

    csv_path = (
        tmp_path
        / "invalid_french_country_code.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="France must use country_code FR",
    ):
        load_instrument_master(
            csv_path
        )


def test_non_primary_cannot_use_primary_status(
    tmp_path: Path,
) -> None:
    rows = project_rows()

    rows[0][
        "is_primary_benchmark"
    ] = "false"

    rows[0][
        "benchmark_status"
    ] = BenchmarkStatus.PRIMARY_BENCHMARK.value

    csv_path = (
        tmp_path
        / "invalid_status.csv"
    )

    write_rows(
        path=csv_path,
        rows=rows,
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="non-primary instrument",
    ):
        load_instrument_master(
            csv_path
        )


def test_unknown_isin_is_rejected() -> None:
    records = load_instrument_master()

    with pytest.raises(
        InstrumentMasterValidationError,
        match="Unknown instrument-master ISIN",
    ):
        get_master_record(
            isin="XX0000000000",
            records=records,
        )


def test_missing_primary_country_is_rejected() -> None:
    records = load_instrument_master()

    modified_records = tuple(
        SovereignInstrumentMasterRecord(
            instrument=record.instrument,
            original_maturity_years=(
                record.original_maturity_years
            ),
            is_primary_benchmark=False,
            benchmark_status=BenchmarkStatus.REFERENCE_BOND,
        )
        if record.country
        == SovereignCountry.ITALY
        else record
        for record in records
    )

    with pytest.raises(
        InstrumentMasterValidationError,
        match="Every represented country",
    ):
        validate_instrument_master(
            modified_records
        )