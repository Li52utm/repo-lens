from __future__ import annotations

import pandas as pd
import pytest

from src.sovereign_instrument_catalog import (
    InstrumentCatalogValidationError,
    all_instruments,
    all_master_records,
    catalogue_to_frame,
    clear_instrument_catalog_cache,
    instrument_by_isin,
    instruments_for_country,
    instruments_for_security_type,
    instruments_for_tenor,
    master_record_by_isin,
    off_the_run_bonds,
    primary_benchmark,
    primary_benchmarks,
    reference_bonds,
)
from src.sovereign_instrument_master import (
    BenchmarkStatus,
    SovereignInstrumentMasterRecord,
    load_instrument_master,
)
from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
    SovereignSecurityType,
)


@pytest.fixture
def records() -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Load the project instrument master once per test.
    """
    return load_instrument_master()


def test_all_master_records_returns_twelve_records(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    result = all_master_records(
        records
    )

    assert len(
        result
    ) == 12


def test_all_instruments_returns_existing_contract(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instruments = all_instruments(
        records
    )

    assert len(
        instruments
    ) == 12

    assert all(
        isinstance(
            instrument,
            SovereignInstrument,
        )
        for instrument in instruments
    )


def test_instrument_lookup_by_isin(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instrument = instrument_by_isin(
        isin="it0005467482",
        records=records,
    )

    assert instrument.isin == "IT0005467482"
    assert instrument.country == SovereignCountry.ITALY
    assert instrument.benchmark_tenor_years == 3


def test_master_record_lookup_preserves_metadata(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    record = master_record_by_isin(
        isin="IT0005467482",
        records=records,
    )

    assert record.benchmark_status == (
        BenchmarkStatus.REFERENCE_BOND
    )

    assert record.is_primary_benchmark is False
    assert record.original_maturity_years == 7


def test_unknown_isin_is_rejected(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    with pytest.raises(
        InstrumentCatalogValidationError,
        match="Unknown sovereign instrument ISIN",
    ):
        instrument_by_isin(
            isin="XX0000000000",
            records=records,
        )


def test_country_filter(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    germany = instruments_for_country(
        country=SovereignCountry.GERMANY,
        records=records,
    )

    italy = instruments_for_country(
        country=SovereignCountry.ITALY,
        records=records,
    )

    assert len(
        germany
    ) == 7

    assert len(
        italy
    ) == 5

    assert all(
        instrument.country
        == SovereignCountry.GERMANY
        for instrument in germany
    )


def test_tenor_filter_across_countries(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    ten_year = instruments_for_tenor(
        benchmark_tenor_years=10,
        records=records,
    )

    assert {
        instrument.country
        for instrument in ten_year
    } == {
        SovereignCountry.GERMANY,
        SovereignCountry.ITALY,
    }


def test_three_year_filter_returns_italian_reference_bond(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instruments = instruments_for_tenor(
        benchmark_tenor_years=3,
        country=SovereignCountry.ITALY,
        records=records,
    )

    assert len(
        instruments
    ) == 1

    assert instruments[0].isin == "IT0005467482"


def test_security_type_filter(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    bunds = instruments_for_security_type(
        security_type=SovereignSecurityType.BUND,
        country=SovereignCountry.GERMANY,
        records=records,
    )

    assert len(
        bunds
    ) == 5


def test_primary_benchmark_lookup(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instrument = primary_benchmark(
        country=SovereignCountry.ITALY,
        benchmark_tenor_years=10,
        records=records,
    )

    assert instrument.isin == "IT0005706285"


def test_primary_benchmark_missing_for_three_year_sector(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    with pytest.raises(
        InstrumentCatalogValidationError,
        match="No primary benchmark exists",
    ):
        primary_benchmark(
            country=SovereignCountry.ITALY,
            benchmark_tenor_years=3,
            records=records,
        )


def test_primary_benchmark_list(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    german_benchmarks = primary_benchmarks(
        country=SovereignCountry.GERMANY,
        records=records,
    )

    italian_benchmarks = primary_benchmarks(
        country=SovereignCountry.ITALY,
        records=records,
    )

    assert len(
        german_benchmarks
    ) == 7

    assert len(
        italian_benchmarks
    ) == 4


def test_reference_bond_filter(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instruments = reference_bonds(
        country=SovereignCountry.ITALY,
        records=records,
    )

    assert len(
        instruments
    ) == 1

    assert instruments[0].isin == "IT0005467482"


def test_off_the_run_filter_is_empty_in_first_tranche(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    instruments = off_the_run_bonds(
        records=records
    )

    assert instruments == ()


def test_catalogue_frame_contract(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    frame = catalogue_to_frame(
        records
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert len(
        frame
    ) == 12

    assert {
        "isin",
        "display_name",
        "country",
        "security_type",
        "maturity_date",
        "annual_coupon_percent",
        "coupon_frequency_label",
        "benchmark_tenor_years",
        "is_primary_benchmark",
        "benchmark_status",
        "years_to_maturity_from_source_check",
    }.issubset(
        frame.columns
    )

    assert frame[
        "years_to_maturity_from_source_check"
    ].notna().all()


def test_invalid_tenor_is_rejected(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    with pytest.raises(
        InstrumentCatalogValidationError,
        match="must be positive",
    ):
        instruments_for_tenor(
            benchmark_tenor_years=0,
            records=records,
        )


def test_catalogue_cache_can_be_cleared() -> None:
    clear_instrument_catalog_cache()

    instruments = all_instruments()

    assert len(
        instruments
    ) == 12

    clear_instrument_catalog_cache()