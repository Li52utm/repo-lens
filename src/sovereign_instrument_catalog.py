from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.sovereign_instrument_master import (
    DEFAULT_INSTRUMENT_MASTER_PATH,
    BenchmarkStatus,
    InstrumentMasterValidationError,
    SovereignInstrumentMasterRecord,
    instrument_master_to_frame,
    load_instrument_master,
    primary_benchmark_for_country_tenor,
)
from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
    SovereignSecurityType,
)


class InstrumentCatalogError(RuntimeError):
    """
    Base exception for RepoLens sovereign instrument-catalogue operations.
    """


class InstrumentCatalogValidationError(
    InstrumentCatalogError
):
    """
    Raised when catalogue queries or catalogue data are invalid.
    """


@lru_cache(maxsize=4)
def cached_instrument_master(
    csv_path_text: str = str(
        DEFAULT_INSTRUMENT_MASTER_PATH
    ),
) -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Load and cache the validated sovereign instrument master.

    The path is stored as text so the function can be safely cached.
    """
    try:
        return load_instrument_master(
            Path(
                csv_path_text
            )
        )
    except InstrumentMasterValidationError as error:
        raise InstrumentCatalogValidationError(
            str(
                error
            )
        ) from error


def clear_instrument_catalog_cache() -> None:
    """
    Clear cached instrument-master records.

    This is useful after reference-data files are refreshed.
    """
    cached_instrument_master.cache_clear()


def active_records(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Return supplied records or the cached project instrument master.
    """
    return (
        cached_instrument_master()
        if records is None
        else records
    )


def all_master_records(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Return every validated sovereign instrument-master record.
    """
    selected_records = active_records(
        records
    )

    return tuple(
        sorted(
            selected_records,
            key=lambda record: (
                record.country.value,
                record.instrument.maturity_date,
                record.isin,
            ),
        )
    )


def all_instruments(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return every sovereign instrument in catalogue order.
    """
    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
    )


def instrument_by_isin(
    isin: str,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> SovereignInstrument:
    """
    Return one sovereign instrument by ISIN.
    """
    normalised_isin = isin.strip().upper()

    if not normalised_isin:
        raise InstrumentCatalogValidationError(
            "isin must not be empty."
        )

    matches = tuple(
        record.instrument
        for record in active_records(
            records
        )
        if record.isin == normalised_isin
    )

    if not matches:
        raise InstrumentCatalogValidationError(
            f"Unknown sovereign instrument ISIN: {normalised_isin}"
        )

    if len(
        matches
    ) > 1:
        raise InstrumentCatalogValidationError(
            f"Duplicate sovereign instrument ISIN: {normalised_isin}"
        )

    return matches[0]


def master_record_by_isin(
    isin: str,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> SovereignInstrumentMasterRecord:
    """
    Return one complete instrument-master record by ISIN.
    """
    normalised_isin = isin.strip().upper()

    if not normalised_isin:
        raise InstrumentCatalogValidationError(
            "isin must not be empty."
        )

    matches = tuple(
        record
        for record in active_records(
            records
        )
        if record.isin == normalised_isin
    )

    if not matches:
        raise InstrumentCatalogValidationError(
            f"Unknown sovereign instrument ISIN: {normalised_isin}"
        )

    if len(
        matches
    ) > 1:
        raise InstrumentCatalogValidationError(
            f"Duplicate sovereign instrument ISIN: {normalised_isin}"
        )

    return matches[0]


def instruments_for_country(
    country: SovereignCountry,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return all instruments for one sovereign country.
    """
    matches = tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if record.country == country
    )

    return matches


def instruments_for_tenor(
    benchmark_tenor_years: int,
    country: SovereignCountry | None = None,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return instruments assigned to one maturity sector.

    When country is omitted, matching instruments from every supported
    country are returned.
    """
    if benchmark_tenor_years <= 0:
        raise InstrumentCatalogValidationError(
            "benchmark_tenor_years must be positive."
        )

    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if (
            record.benchmark_tenor_years
            == benchmark_tenor_years
            and (
                country is None
                or record.country == country
            )
        )
    )


def instruments_for_security_type(
    security_type: SovereignSecurityType,
    country: SovereignCountry | None = None,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return instruments matching one sovereign security type.
    """
    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if (
            record.instrument.security_type
            == security_type
            and (
                country is None
                or record.country == country
            )
        )
    )


def primary_benchmark(
    country: SovereignCountry,
    benchmark_tenor_years: int,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> SovereignInstrument:
    """
    Return the explicitly designated primary benchmark instrument.
    """
    try:
        return primary_benchmark_for_country_tenor(
            country=country,
            benchmark_tenor_years=(
                benchmark_tenor_years
            ),
            records=active_records(
                records
            ),
        )
    except InstrumentMasterValidationError as error:
        raise InstrumentCatalogValidationError(
            str(
                error
            )
        ) from error


def primary_benchmarks(
    country: SovereignCountry | None = None,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return all explicitly designated primary benchmark instruments.
    """
    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if (
            record.is_primary_benchmark
            and (
                country is None
                or record.country == country
            )
        )
    )


def reference_bonds(
    country: SovereignCountry | None = None,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return instruments classified as reference bonds.
    """
    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if (
            record.benchmark_status
            == BenchmarkStatus.REFERENCE_BOND
            and (
                country is None
                or record.country == country
            )
        )
    )


def off_the_run_bonds(
    country: SovereignCountry | None = None,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Return instruments classified as off-the-run bonds.
    """
    return tuple(
        record.instrument
        for record in all_master_records(
            records
        )
        if (
            record.benchmark_status
            == BenchmarkStatus.OFF_THE_RUN
            and (
                country is None
                or record.country == country
            )
        )
    )


def catalogue_to_frame(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> pd.DataFrame:
    """
    Return a dashboard-ready catalogue DataFrame.
    """
    selected_records = all_master_records(
        records
    )

    try:
        frame = instrument_master_to_frame(
            selected_records
        )
    except InstrumentMasterValidationError as error:
        raise InstrumentCatalogValidationError(
            str(
                error
            )
        ) from error

    frame = frame.copy()

    frame["years_to_maturity_from_source_check"] = (
        pd.to_datetime(
            frame[
                "maturity_date"
            ]
        )
        - pd.to_datetime(
            frame[
                "source_checked_date"
            ]
        )
    ).dt.days / 365.25

    frame["coupon_frequency_label"] = (
        frame[
            "coupon_frequency"
        ]
        .map(
            {
                1: "Annual",
                2: "Semi-annual",
                4: "Quarterly",
            }
        )
        .fillna(
            "Other"
        )
    )

    return (
        frame
        .sort_values(
            [
                "country",
                "maturity_date",
                "isin",
            ]
        )
        .reset_index(
            drop=True
        )
    )