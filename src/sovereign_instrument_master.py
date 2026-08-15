from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

from src.sovereign_instruments import (
    InstrumentDataStatus,
    InstrumentRegistryValidationError,
    SovereignCountry,
    SovereignInstrument,
    SovereignSecurityType,
)


DEFAULT_INSTRUMENT_MASTER_PATH: Final[Path] = Path(
    "data/reference/sovereign_instruments.csv"
)


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "display_name",
    "country",
    "country_code",
    "issuer",
    "security_type",
    "issue_date",
    "maturity_date",
    "annual_coupon_rate",
    "coupon_frequency",
    "benchmark_tenor_years",
    "original_maturity_years",
    "currency",
    "face_value",
    "is_primary_benchmark",
    "benchmark_status",
    "source_name",
    "source_locator",
    "source_checked_date",
    "data_status",
    "is_nominal_fixed_rate",
)


class InstrumentMasterError(RuntimeError):
    """
    Base exception for RepoLens instrument-master operations.
    """


class InstrumentMasterValidationError(
    InstrumentMasterError
):
    """
    Raised when reference-data rows fail validation.
    """


class BenchmarkStatus(StrEnum):
    """
    Classification of an instrument within its maturity sector.
    """

    PRIMARY_BENCHMARK = "PRIMARY_BENCHMARK"
    REFERENCE_BOND = "REFERENCE_BOND"
    OFF_THE_RUN = "OFF_THE_RUN"


@dataclass(frozen=True)
class SovereignInstrumentMasterRecord:
    """
    Store a validated sovereign instrument and master-data metadata.

    The embedded SovereignInstrument preserves compatibility with the
    existing RepoLens pricing, snapshot, relative-value and portfolio
    engines.
    """

    instrument: SovereignInstrument
    original_maturity_years: int
    is_primary_benchmark: bool
    benchmark_status: BenchmarkStatus

    def __post_init__(self) -> None:
        if self.original_maturity_years <= 0:
            raise InstrumentMasterValidationError(
                "original_maturity_years must be positive."
            )

        if (
            self.is_primary_benchmark
            and self.benchmark_status
            != BenchmarkStatus.PRIMARY_BENCHMARK
        ):
            raise InstrumentMasterValidationError(
                "A primary benchmark must have benchmark_status "
                "PRIMARY_BENCHMARK."
            )

        if (
            not self.is_primary_benchmark
            and self.benchmark_status
            == BenchmarkStatus.PRIMARY_BENCHMARK
        ):
            raise InstrumentMasterValidationError(
                "A non-primary instrument cannot have benchmark_status "
                "PRIMARY_BENCHMARK."
            )

    @property
    def isin(self) -> str:
        """
        Return the instrument ISIN.
        """
        return self.instrument.isin

    @property
    def country(self) -> SovereignCountry:
        """
        Return the sovereign country.
        """
        return self.instrument.country

    @property
    def benchmark_tenor_years(self) -> int:
        """
        Return the assigned benchmark maturity sector.
        """
        return self.instrument.benchmark_tenor_years


def parse_required_text(
    row: dict[str, str],
    field_name: str,
    row_number: int,
) -> str:
    """
    Return one required non-empty text value.
    """
    value = str(
        row.get(
            field_name,
            "",
        )
    ).strip()

    if not value:
        raise InstrumentMasterValidationError(
            f"Row {row_number}: {field_name} must not be empty."
        )

    return value


def parse_date_value(
    row: dict[str, str],
    field_name: str,
    row_number: int,
) -> date:
    """
    Parse an ISO-format date from one CSV field.
    """
    value = parse_required_text(
        row=row,
        field_name=field_name,
        row_number=row_number,
    )

    try:
        return date.fromisoformat(
            value
        )
    except ValueError as error:
        raise InstrumentMasterValidationError(
            f"Row {row_number}: {field_name} must use YYYY-MM-DD format."
        ) from error


def parse_integer_value(
    row: dict[str, str],
    field_name: str,
    row_number: int,
) -> int:
    """
    Parse an integer from one CSV field.
    """
    value = parse_required_text(
        row=row,
        field_name=field_name,
        row_number=row_number,
    )

    try:
        return int(
            value
        )
    except ValueError as error:
        raise InstrumentMasterValidationError(
            f"Row {row_number}: {field_name} must be an integer."
        ) from error


def parse_float_value(
    row: dict[str, str],
    field_name: str,
    row_number: int,
) -> float:
    """
    Parse a floating-point number from one CSV field.
    """
    value = parse_required_text(
        row=row,
        field_name=field_name,
        row_number=row_number,
    )

    try:
        return float(
            value
        )
    except ValueError as error:
        raise InstrumentMasterValidationError(
            f"Row {row_number}: {field_name} must be numeric."
        ) from error


def parse_boolean_value(
    row: dict[str, str],
    field_name: str,
    row_number: int,
) -> bool:
    """
    Parse a strict true-or-false CSV value.
    """
    value = parse_required_text(
        row=row,
        field_name=field_name,
        row_number=row_number,
    ).lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise InstrumentMasterValidationError(
        f"Row {row_number}: {field_name} must be true or false."
    )


def parse_enum_value[
    EnumType: StrEnum
](
    enum_type: type[EnumType],
    value: str,
    field_name: str,
    row_number: int,
) -> EnumType:
    """
    Convert a CSV value into one supported string enumeration.
    """
    try:
        return enum_type(
            value
        )
    except ValueError as error:
        supported_values = ", ".join(
            member.value
            for member in enum_type
        )

        raise InstrumentMasterValidationError(
            f"Row {row_number}: unsupported {field_name} '{value}'. "
            f"Supported values are: {supported_values}."
        ) from error


def validate_country_code(
    country: SovereignCountry,
    country_code: str,
    row_number: int,
) -> None:
    """
    Validate the country and ISO-like code combination.
    """
    expected_codes = {
        SovereignCountry.GERMANY: "DE",
        SovereignCountry.ITALY: "IT",
        SovereignCountry.FRANCE: "FR",
    }

    expected_code = expected_codes[
        country
    ]

    if country_code != expected_code:
        raise InstrumentMasterValidationError(
            f"Row {row_number}: {country.value} must use country_code "
            f"{expected_code}."
        )


def record_from_csv_row(
    row: dict[str, str],
    row_number: int,
) -> SovereignInstrumentMasterRecord:
    """
    Convert one CSV row into a validated master record.
    """
    country = parse_enum_value(
        enum_type=SovereignCountry,
        value=parse_required_text(
            row=row,
            field_name="country",
            row_number=row_number,
        ),
        field_name="country",
        row_number=row_number,
    )

    country_code = parse_required_text(
        row=row,
        field_name="country_code",
        row_number=row_number,
    ).upper()

    validate_country_code(
        country=country,
        country_code=country_code,
        row_number=row_number,
    )

    security_type = parse_enum_value(
        enum_type=SovereignSecurityType,
        value=parse_required_text(
            row=row,
            field_name="security_type",
            row_number=row_number,
        ),
        field_name="security_type",
        row_number=row_number,
    )

    data_status = parse_enum_value(
        enum_type=InstrumentDataStatus,
        value=parse_required_text(
            row=row,
            field_name="data_status",
            row_number=row_number,
        ),
        field_name="data_status",
        row_number=row_number,
    )

    benchmark_status = parse_enum_value(
        enum_type=BenchmarkStatus,
        value=parse_required_text(
            row=row,
            field_name="benchmark_status",
            row_number=row_number,
        ),
        field_name="benchmark_status",
        row_number=row_number,
    )

    instrument = SovereignInstrument(
        isin=parse_required_text(
            row=row,
            field_name="isin",
            row_number=row_number,
        ).upper(),
        display_name=parse_required_text(
            row=row,
            field_name="display_name",
            row_number=row_number,
        ),
        country=country,
        country_code=country_code,
        issuer=parse_required_text(
            row=row,
            field_name="issuer",
            row_number=row_number,
        ),
        security_type=security_type,
        issue_date=parse_date_value(
            row=row,
            field_name="issue_date",
            row_number=row_number,
        ),
        maturity_date=parse_date_value(
            row=row,
            field_name="maturity_date",
            row_number=row_number,
        ),
        annual_coupon_rate=parse_float_value(
            row=row,
            field_name="annual_coupon_rate",
            row_number=row_number,
        ),
        coupon_frequency=parse_integer_value(
            row=row,
            field_name="coupon_frequency",
            row_number=row_number,
        ),
        benchmark_tenor_years=parse_integer_value(
            row=row,
            field_name="benchmark_tenor_years",
            row_number=row_number,
        ),
        currency=parse_required_text(
            row=row,
            field_name="currency",
            row_number=row_number,
        ).upper(),
        face_value=parse_float_value(
            row=row,
            field_name="face_value",
            row_number=row_number,
        ),
        source_name=parse_required_text(
            row=row,
            field_name="source_name",
            row_number=row_number,
        ),
        source_locator=parse_required_text(
            row=row,
            field_name="source_locator",
            row_number=row_number,
        ),
        source_checked_date=parse_date_value(
            row=row,
            field_name="source_checked_date",
            row_number=row_number,
        ),
        data_status=data_status,
        is_nominal_fixed_rate=parse_boolean_value(
            row=row,
            field_name="is_nominal_fixed_rate",
            row_number=row_number,
        ),
    )

    return SovereignInstrumentMasterRecord(
        instrument=instrument,
        original_maturity_years=parse_integer_value(
            row=row,
            field_name="original_maturity_years",
            row_number=row_number,
        ),
        is_primary_benchmark=parse_boolean_value(
            row=row,
            field_name="is_primary_benchmark",
            row_number=row_number,
        ),
        benchmark_status=benchmark_status,
    )


def validate_csv_columns(
    fieldnames: list[str] | None,
) -> None:
    """
    Validate that all required master-data columns are present.
    """
    if fieldnames is None:
        raise InstrumentMasterValidationError(
            "Instrument-master CSV does not contain a header row."
        )

    normalised_fieldnames = tuple(
        field_name.strip()
        for field_name in fieldnames
    )

    missing_columns = tuple(
        column
        for column in REQUIRED_COLUMNS
        if column not in normalised_fieldnames
    )

    if missing_columns:
        raise InstrumentMasterValidationError(
            "Instrument-master CSV is missing required columns: "
            f"{list(missing_columns)}."
        )


def validate_instrument_master(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ],
) -> None:
    """
    Validate cross-record uniqueness and benchmark assignments.

    Several bonds may occupy the same country-tenor sector, but only one
    may be designated as that sector's primary benchmark.
    """
    if not records:
        raise InstrumentMasterValidationError(
            "Instrument master must not be empty."
        )

    isins = [
        record.isin
        for record in records
    ]

    if len(
        isins
    ) != len(
        set(
            isins
        )
    ):
        raise InstrumentMasterValidationError(
            "Instrument master contains duplicate ISINs."
        )

    primary_pairs = [
        (
            record.country,
            record.benchmark_tenor_years,
        )
        for record in records
        if record.is_primary_benchmark
    ]

    if len(
        primary_pairs
    ) != len(
        set(
            primary_pairs
        )
    ):
        raise InstrumentMasterValidationError(
            "Instrument master contains more than one primary benchmark "
            "for the same country-tenor pair."
        )

    countries_with_records = {
        record.country
        for record in records
    }

    countries_with_primary_benchmarks = {
        record.country
        for record in records
        if record.is_primary_benchmark
    }

    missing_primary_countries = (
        countries_with_records
        - countries_with_primary_benchmarks
    )

    if missing_primary_countries:
        missing_names = sorted(
            country.value
            for country in missing_primary_countries
        )

        raise InstrumentMasterValidationError(
            "Every represented country must contain at least one primary "
            f"benchmark. Missing: {missing_names}."
        )


def load_instrument_master(
    csv_path: Path | str = DEFAULT_INSTRUMENT_MASTER_PATH,
) -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Load and validate the RepoLens sovereign instrument master.
    """
    path = Path(
        csv_path
    )

    if not path.exists():
        raise InstrumentMasterValidationError(
            f"Instrument-master CSV does not exist: {path}"
        )

    if not path.is_file():
        raise InstrumentMasterValidationError(
            f"Instrument-master path is not a file: {path}"
        )

    records: list[
        SovereignInstrumentMasterRecord
    ] = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        validate_csv_columns(
            reader.fieldnames
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                record = record_from_csv_row(
                    row=row,
                    row_number=row_number,
                )
            except InstrumentRegistryValidationError as error:
                raise InstrumentMasterValidationError(
                    f"Row {row_number}: {error}"
                ) from error

            records.append(
                record
            )

    loaded_records = tuple(
        records
    )

    validate_instrument_master(
        loaded_records
    )

    return loaded_records


def get_master_record(
    isin: str,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> SovereignInstrumentMasterRecord:
    """
    Return one master record by ISIN.
    """
    active_records = (
        load_instrument_master()
        if records is None
        else records
    )

    normalised_isin = isin.strip().upper()

    matches = tuple(
        record
        for record in active_records
        if record.isin == normalised_isin
    )

    if not matches:
        raise InstrumentMasterValidationError(
            f"Unknown instrument-master ISIN: {normalised_isin}"
        )

    if len(
        matches
    ) > 1:
        raise InstrumentMasterValidationError(
            f"Duplicate instrument-master ISIN: {normalised_isin}"
        )

    return matches[0]


def records_for_country(
    country: SovereignCountry,
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> tuple[
    SovereignInstrumentMasterRecord,
    ...,
]:
    """
    Return all master records for one country.
    """
    active_records = (
        load_instrument_master()
        if records is None
        else records
    )

    return tuple(
        sorted(
            (
                record
                for record in active_records
                if record.country == country
            ),
            key=lambda record: (
                record.instrument.maturity_date,
                record.isin,
            ),
        )
    )


def primary_benchmark_for_country_tenor(
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
    if benchmark_tenor_years <= 0:
        raise InstrumentMasterValidationError(
            "benchmark_tenor_years must be positive."
        )

    active_records = (
        load_instrument_master()
        if records is None
        else records
    )

    matches = tuple(
        record
        for record in active_records
        if (
            record.country == country
            and record.benchmark_tenor_years
            == benchmark_tenor_years
            and record.is_primary_benchmark
        )
    )

    if not matches:
        raise InstrumentMasterValidationError(
            "No primary benchmark exists for "
            f"{country.value} {benchmark_tenor_years}Y."
        )

    if len(
        matches
    ) > 1:
        raise InstrumentMasterValidationError(
            "More than one primary benchmark exists for "
            f"{country.value} {benchmark_tenor_years}Y."
        )

    return matches[0].instrument


def instrument_master_to_frame(
    records: tuple[
        SovereignInstrumentMasterRecord,
        ...,
    ] | None = None,
) -> pd.DataFrame:
    """
    Convert the validated instrument master into a tabular dataset.
    """
    active_records = (
        load_instrument_master()
        if records is None
        else records
    )

    validate_instrument_master(
        active_records
    )

    rows = [
        {
            "isin": record.instrument.isin,
            "display_name": record.instrument.display_name,
            "country": record.instrument.country.value,
            "country_code": record.instrument.country_code,
            "issuer": record.instrument.issuer,
            "security_type": record.instrument.security_type.value,
            "issue_date": record.instrument.issue_date,
            "maturity_date": record.instrument.maturity_date,
            "annual_coupon_rate": (
                record.instrument.annual_coupon_rate
            ),
            "annual_coupon_percent": (
                record.instrument.annual_coupon_rate
                * 100.0
            ),
            "coupon_frequency": (
                record.instrument.coupon_frequency
            ),
            "benchmark_tenor_years": (
                record.instrument.benchmark_tenor_years
            ),
            "original_maturity_years": (
                record.original_maturity_years
            ),
            "currency": record.instrument.currency,
            "face_value": record.instrument.face_value,
            "is_primary_benchmark": (
                record.is_primary_benchmark
            ),
            "benchmark_status": (
                record.benchmark_status.value
            ),
            "source_name": record.instrument.source_name,
            "source_locator": (
                record.instrument.source_locator
            ),
            "source_checked_date": (
                record.instrument.source_checked_date
            ),
            "data_status": (
                record.instrument.data_status.value
            ),
            "is_nominal_fixed_rate": (
                record.instrument.is_nominal_fixed_rate
            ),
        }
        for record in active_records
    ]

    return (
        pd.DataFrame(
            rows
        )
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