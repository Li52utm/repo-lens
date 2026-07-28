from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.bond_analytics import FixedRateBond
from src.sovereign_instruments import (
    GERMAN_INSTRUMENTS,
    INSTRUMENTS_BY_ISIN,
    ITALIAN_INSTRUMENTS,
    SOVEREIGN_INSTRUMENTS,
    InstrumentDataStatus,
    InstrumentRegistryValidationError,
    SovereignCountry,
    SovereignInstrument,
    SovereignSecurityType,
    get_instrument,
    instrument_for_country_tenor,
    instruments_for_country,
    registry_to_frame,
    validate_registry,
)


EXPECTED_ISINS = {
    "DE000BU22148",
    "DE000BU25075",
    "DE000BU2Z072",
    "DE000BU2D012",
    "IT0005692410",
    "IT0005707614",
    "IT0005706285",
    "IT0005668238",
}


def create_test_instrument(
    isin: str = "DE000TEST001",
    benchmark_tenor_years: int = 10,
) -> SovereignInstrument:
    """
    Create a deterministic valid registry instrument.
    """
    return SovereignInstrument(
        isin=isin,
        display_name="Test sovereign bond",
        country=SovereignCountry.GERMANY,
        country_code="DE",
        issuer="Federal Republic of Germany",
        security_type=SovereignSecurityType.BUND,
        issue_date=date(
            2026,
            1,
            1,
        ),
        maturity_date=date(
            2036,
            1,
            1,
        ),
        annual_coupon_rate=0.03,
        coupon_frequency=1,
        benchmark_tenor_years=benchmark_tenor_years,
        currency="EUR",
        face_value=100.0,
        source_name="Official test source",
        source_locator="https://example.test/bond",
        source_checked_date=date(
            2026,
            7,
            28,
        ),
        data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
    )


def test_registry_contains_eight_instruments() -> None:
    assert len(
        SOVEREIGN_INSTRUMENTS
    ) == 8

    assert len(
        GERMAN_INSTRUMENTS
    ) == 4

    assert len(
        ITALIAN_INSTRUMENTS
    ) == 4


def test_registry_contains_expected_isins() -> None:
    assert set(
        INSTRUMENTS_BY_ISIN
    ) == EXPECTED_ISINS


def test_each_country_has_expected_tenors() -> None:
    for country in (
        SovereignCountry.GERMANY,
        SovereignCountry.ITALY,
    ):
        country_instruments = instruments_for_country(
            country
        )

        assert {
            instrument.benchmark_tenor_years
            for instrument in country_instruments
        } == {
            2,
            5,
            10,
            30,
        }


def test_german_bonds_use_annual_coupons() -> None:
    assert all(
        instrument.coupon_frequency == 1
        for instrument in GERMAN_INSTRUMENTS
    )


def test_italian_bonds_use_semiannual_coupons() -> None:
    assert all(
        instrument.coupon_frequency == 2
        for instrument in ITALIAN_INSTRUMENTS
    )


def test_get_instrument_is_case_insensitive() -> None:
    instrument = get_instrument(
        "de000bu2z072"
    )

    assert instrument.isin == "DE000BU2Z072"

    assert instrument.country == SovereignCountry.GERMANY

    assert instrument.benchmark_tenor_years == 10


def test_unknown_instrument_is_rejected() -> None:
    with pytest.raises(
        InstrumentRegistryValidationError,
        match="Unknown sovereign instrument",
    ):
        get_instrument(
            "DE000UNKNOWN1"
        )


def test_country_tenor_lookup() -> None:
    instrument = instrument_for_country_tenor(
        country=SovereignCountry.ITALY,
        benchmark_tenor_years=10,
    )

    assert instrument.isin == "IT0005706285"

    assert instrument.annual_coupon_rate == pytest.approx(
        0.038
    )


def test_registry_entry_converts_to_bond_analytics_model() -> None:
    registry_instrument = get_instrument(
        "IT0005707614"
    )

    bond = registry_instrument.to_fixed_rate_bond()

    assert isinstance(
        bond,
        FixedRateBond,
    )

    assert bond.isin == registry_instrument.isin

    assert bond.maturity_date == registry_instrument.maturity_date

    assert bond.annual_coupon_rate == pytest.approx(
        registry_instrument.annual_coupon_rate
    )

    assert bond.coupon_frequency == 2


def test_registry_frame_contains_expected_contract() -> None:
    frame = registry_to_frame()

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert len(
        frame
    ) == 8

    assert {
        "isin",
        "display_name",
        "country",
        "security_type",
        "issue_date",
        "maturity_date",
        "annual_coupon_percent",
        "coupon_frequency",
        "benchmark_tenor_years",
        "source_name",
        "source_checked_date",
        "data_status",
    }.issubset(
        frame.columns
    )


def test_registry_has_unique_isins_and_country_tenors() -> None:
    validate_registry(
        SOVEREIGN_INSTRUMENTS
    )


def test_duplicate_isin_is_rejected() -> None:
    instrument = create_test_instrument()

    with pytest.raises(
        InstrumentRegistryValidationError,
        match="duplicate ISINs",
    ):
        validate_registry(
            (
                instrument,
                instrument,
            )
        )


def test_duplicate_country_tenor_is_rejected() -> None:
    first = create_test_instrument(
        isin="DE000TEST001",
        benchmark_tenor_years=10,
    )

    second = create_test_instrument(
        isin="DE000TEST002",
        benchmark_tenor_years=10,
    )

    with pytest.raises(
        InstrumentRegistryValidationError,
        match="country-tenor pairs",
    ):
        validate_registry(
            (
                first,
                second,
            )
        )


def test_invalid_isin_is_rejected() -> None:
    with pytest.raises(
        InstrumentRegistryValidationError,
        match="12 characters",
    ):
        create_test_instrument(
            isin="TOO-SHORT",
        )


def test_issue_date_must_precede_maturity() -> None:
    with pytest.raises(
        InstrumentRegistryValidationError,
        match="before maturity",
    ):
        SovereignInstrument(
            isin="DE000TEST003",
            display_name="Invalid dates",
            country=SovereignCountry.GERMANY,
            country_code="DE",
            issuer="Federal Republic of Germany",
            security_type=SovereignSecurityType.BUND,
            issue_date=date(
                2036,
                1,
                1,
            ),
            maturity_date=date(
                2036,
                1,
                1,
            ),
            annual_coupon_rate=0.03,
            coupon_frequency=1,
            benchmark_tenor_years=10,
            currency="EUR",
            face_value=100.0,
            source_name="Official source",
            source_locator="https://example.test",
            source_checked_date=date(
                2026,
                7,
                28,
            ),
            data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
        )


def test_non_euro_currency_is_rejected() -> None:
    with pytest.raises(
        InstrumentRegistryValidationError,
        match="EUR instruments only",
    ):
        SovereignInstrument(
            isin="DE000TEST004",
            display_name="Invalid currency",
            country=SovereignCountry.GERMANY,
            country_code="DE",
            issuer="Federal Republic of Germany",
            security_type=SovereignSecurityType.BUND,
            issue_date=date(
                2026,
                1,
                1,
            ),
            maturity_date=date(
                2036,
                1,
                1,
            ),
            annual_coupon_rate=0.03,
            coupon_frequency=1,
            benchmark_tenor_years=10,
            currency="GBP",
            face_value=100.0,
            source_name="Official source",
            source_locator="https://example.test",
            source_checked_date=date(
                2026,
                7,
                28,
            ),
            data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
        )