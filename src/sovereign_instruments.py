from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import pandas as pd

from src.bond_analytics import FixedRateBond


class InstrumentRegistryError(RuntimeError):
    """
    Base exception for the RepoLens sovereign instrument registry.
    """


class InstrumentRegistryValidationError(InstrumentRegistryError):
    """
    Raised when a sovereign instrument definition is invalid.
    """


class SovereignCountry(StrEnum):
    """
    Countries supported by the first RepoLens sovereign universe.
    """

    GERMANY = "Germany"
    ITALY = "Italy"
    FRANCE = "France"


class SovereignSecurityType(StrEnum):
    """
    Supported fixed-rate sovereign security types.
    """

    SCHATZ = "Schatz"
    BOBL = "Bobl"
    BUND = "Bund"
    BTP_SHORT_TERM = "BTP Short Term"
    BTP = "BTP"
    OAT = "OAT"


class InstrumentDataStatus(StrEnum):
    """
    Describe the origin and timeliness of instrument terms.
    """

    OFFICIAL_REFERENCE = "OFFICIAL_REFERENCE"


@dataclass(frozen=True)
class SovereignInstrument:
    """
    Define one approved fixed-rate sovereign bond.

    Coupon rates are stored as decimal annual rates.

    Example:
        0.03 means 3.00% per annum.
    """

    isin: str
    display_name: str
    country: SovereignCountry
    country_code: str
    issuer: str
    security_type: SovereignSecurityType
    issue_date: date
    maturity_date: date
    annual_coupon_rate: float
    coupon_frequency: int
    benchmark_tenor_years: int
    currency: str
    face_value: float
    source_name: str
    source_locator: str
    source_checked_date: date
    data_status: InstrumentDataStatus
    is_nominal_fixed_rate: bool = True

    def __post_init__(self) -> None:
        normalised_isin = self.isin.strip().upper()

        if len(normalised_isin) != 12:
            raise InstrumentRegistryValidationError(
                "isin must contain exactly 12 characters."
            )

        if not normalised_isin.isalnum():
            raise InstrumentRegistryValidationError(
                "isin must contain only letters and numbers."
            )

        if not self.display_name.strip():
            raise InstrumentRegistryValidationError(
                "display_name must not be empty."
            )

        if len(self.country_code.strip()) != 2:
            raise InstrumentRegistryValidationError(
                "country_code must contain exactly two characters."
            )

        if not self.issuer.strip():
            raise InstrumentRegistryValidationError(
                "issuer must not be empty."
            )

        if self.issue_date >= self.maturity_date:
            raise InstrumentRegistryValidationError(
                "issue_date must be before maturity_date."
            )

        if self.annual_coupon_rate < 0.0:
            raise InstrumentRegistryValidationError(
                "annual_coupon_rate must not be negative."
            )

        if self.coupon_frequency not in {
            1,
            2,
        }:
            raise InstrumentRegistryValidationError(
                "coupon_frequency must be 1 or 2."
            )

        if self.benchmark_tenor_years <= 0:
            raise InstrumentRegistryValidationError(
                "benchmark_tenor_years must be positive."
            )

        if self.currency != "EUR":
            raise InstrumentRegistryValidationError(
                "The first RepoLens registry supports EUR instruments only."
            )

        if self.face_value <= 0.0:
            raise InstrumentRegistryValidationError(
                "face_value must be positive."
            )

        if not self.source_name.strip():
            raise InstrumentRegistryValidationError(
                "source_name must not be empty."
            )

        if not self.source_locator.strip():
            raise InstrumentRegistryValidationError(
                "source_locator must not be empty."
            )

        if not self.is_nominal_fixed_rate:
            raise InstrumentRegistryValidationError(
                "The first registry supports nominal fixed-rate bonds only."
            )

    def to_fixed_rate_bond(self) -> FixedRateBond:
        """
        Convert the registry entry into the bond analytics contract.
        """
        return FixedRateBond(
            isin=self.isin,
            issuer=self.issuer,
            maturity_date=self.maturity_date,
            annual_coupon_rate=self.annual_coupon_rate,
            coupon_frequency=self.coupon_frequency,
            face_value=self.face_value,
            currency=self.currency,
        )


GERMANY_2Y_SCHATZ: Final[SovereignInstrument] = SovereignInstrument(
    isin="DE000BU22148",
    display_name="Germany 2.70% Schatz September 2028",
    country=SovereignCountry.GERMANY,
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=SovereignSecurityType.SCHATZ,
    issue_date=date(
        2026,
        7,
        16,
    ),
    maturity_date=date(
        2028,
        9,
        13,
    ),
    annual_coupon_rate=0.027,
    coupon_frequency=1,
    benchmark_tenor_years=2,
    currency="EUR",
    face_value=100.0,
    source_name="German Finance Agency",
    source_locator=(
        "https://www.deutsche-finanzagentur.de/"
        "fileadmin/user_upload/Institutionelle-investoren/"
        "auktionen/bund_fact_sheet.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


GERMANY_5Y_BOBL: Final[SovereignInstrument] = SovereignInstrument(
    isin="DE000BU25075",
    display_name="Germany 2.90% Bobl October 2031",
    country=SovereignCountry.GERMANY,
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=SovereignSecurityType.BOBL,
    issue_date=date(
        2026,
        7,
        23,
    ),
    maturity_date=date(
        2031,
        10,
        8,
    ),
    annual_coupon_rate=0.029,
    coupon_frequency=1,
    benchmark_tenor_years=5,
    currency="EUR",
    face_value=100.0,
    source_name="German Finance Agency",
    source_locator=(
        "https://www.deutsche-finanzagentur.de/"
        "fileadmin/user_upload/Institutionelle-investoren/"
        "auktionen/bund_fact_sheet.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


GERMANY_10Y_BUND: Final[SovereignInstrument] = SovereignInstrument(
    isin="DE000BU2Z072",
    display_name="Germany 3.00% Bund August 2036",
    country=SovereignCountry.GERMANY,
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=SovereignSecurityType.BUND,
    issue_date=date(
        2026,
        7,
        10,
    ),
    maturity_date=date(
        2036,
        8,
        15,
    ),
    annual_coupon_rate=0.03,
    coupon_frequency=1,
    benchmark_tenor_years=10,
    currency="EUR",
    face_value=100.0,
    source_name="German Finance Agency",
    source_locator=(
        "https://www.deutsche-finanzagentur.de/"
        "fileadmin/user_upload/Institutionelle-investoren/"
        "auktionen/bund_fact_sheet.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


GERMANY_30Y_BUND: Final[SovereignInstrument] = SovereignInstrument(
    isin="DE000BU2D012",
    display_name="Germany 2.90% Bund August 2056",
    country=SovereignCountry.GERMANY,
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=SovereignSecurityType.BUND,
    issue_date=date(
        2025,
        3,
        12,
    ),
    maturity_date=date(
        2056,
        8,
        15,
    ),
    annual_coupon_rate=0.029,
    coupon_frequency=1,
    benchmark_tenor_years=30,
    currency="EUR",
    face_value=100.0,
    source_name="German Finance Agency",
    source_locator=(
        "https://www.deutsche-finanzagentur.de/"
        "fileadmin/user_upload/Institutionelle-investoren/"
        "auktionen/bund_fact_sheet.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


ITALY_2Y_BTP_SHORT: Final[SovereignInstrument] = SovereignInstrument(
    isin="IT0005692410",
    display_name="Italy 2.20% BTP Short Term February 2028",
    country=SovereignCountry.ITALY,
    country_code="IT",
    issuer="Republic of Italy",
    security_type=SovereignSecurityType.BTP_SHORT_TERM,
    issue_date=date(
        2026,
        1,
        29,
    ),
    maturity_date=date(
        2028,
        2,
        28,
    ),
    annual_coupon_rate=0.022,
    coupon_frequency=2,
    benchmark_tenor_years=2,
    currency="EUR",
    face_value=100.0,
    source_name="Italian Department of the Treasury",
    source_locator=(
        "https://www.dt.mef.gov.it/export/sites/sitodt/"
        "modules/documenti_en/debito_pubblico/risultati_aste/"
        "risultati_aste_btp_short_term/"
        "BTP-Short-Term-Auction-Results-24.06.2026.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


ITALY_5Y_BTP: Final[SovereignInstrument] = SovereignInstrument(
    isin="IT0005707614",
    display_name="Italy 3.15% BTP June 2031",
    country=SovereignCountry.ITALY,
    country_code="IT",
    issuer="Republic of Italy",
    security_type=SovereignSecurityType.BTP,
    issue_date=date(
        2026,
        5,
        4,
    ),
    maturity_date=date(
        2031,
        6,
        1,
    ),
    annual_coupon_rate=0.0315,
    coupon_frequency=2,
    benchmark_tenor_years=5,
    currency="EUR",
    face_value=100.0,
    source_name="Italian Department of the Treasury",
    source_locator=(
        "https://www.dt.mef.gov.it/export/sites/sitodt/"
        "modules/documenti_en/debito_pubblico/risultati_aste/"
        "risultati_aste_btp_5_anni/"
        "BTP-5-Years-Auction-Results-26.06.2026.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


ITALY_10Y_BTP: Final[SovereignInstrument] = SovereignInstrument(
    isin="IT0005706285",
    display_name="Italy 3.80% BTP July 2036",
    country=SovereignCountry.ITALY,
    country_code="IT",
    issuer="Republic of Italy",
    security_type=SovereignSecurityType.BTP,
    issue_date=date(
        2026,
        4,
        22,
    ),
    maturity_date=date(
        2036,
        7,
        1,
    ),
    annual_coupon_rate=0.038,
    coupon_frequency=2,
    benchmark_tenor_years=10,
    currency="EUR",
    face_value=100.0,
    source_name="Italian Department of the Treasury",
    source_locator=(
        "https://www.dt.mef.gov.it/export/sites/sitodt/"
        "modules/documenti_en/debito_pubblico/comunicazioni_emissioni/"
        "medio_lungo_termine_comunicazioni/"
        "Medium-Long-Term-Announcement-25.05.2026.pdf"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


ITALY_30Y_BTP: Final[SovereignInstrument] = SovereignInstrument(
    isin="IT0005668238",
    display_name="Italy 4.65% BTP October 2055",
    country=SovereignCountry.ITALY,
    country_code="IT",
    issuer="Republic of Italy",
    security_type=SovereignSecurityType.BTP,
    issue_date=date(
        2025,
        9,
        5,
    ),
    maturity_date=date(
        2055,
        10,
        1,
    ),
    annual_coupon_rate=0.0465,
    coupon_frequency=2,
    benchmark_tenor_years=30,
    currency="EUR",
    face_value=100.0,
    source_name="Borsa Italiana",
    source_locator=(
        "https://www.borsaitaliana.it/borsa/search/"
        "scheda.html?code=IT0005668238&lang=en"
    ),
    source_checked_date=date(
        2026,
        7,
        28,
    ),
    data_status=InstrumentDataStatus.OFFICIAL_REFERENCE,
)


GERMAN_INSTRUMENTS: Final[
    tuple[SovereignInstrument, ...]
] = (
    GERMANY_2Y_SCHATZ,
    GERMANY_5Y_BOBL,
    GERMANY_10Y_BUND,
    GERMANY_30Y_BUND,
)


ITALIAN_INSTRUMENTS: Final[
    tuple[SovereignInstrument, ...]
] = (
    ITALY_2Y_BTP_SHORT,
    ITALY_5Y_BTP,
    ITALY_10Y_BTP,
    ITALY_30Y_BTP,
)


SOVEREIGN_INSTRUMENTS: Final[
    tuple[SovereignInstrument, ...]
] = (
    *GERMAN_INSTRUMENTS,
    *ITALIAN_INSTRUMENTS,
)


INSTRUMENTS_BY_ISIN: Final[
    dict[str, SovereignInstrument]
] = {
    instrument.isin: instrument
    for instrument in SOVEREIGN_INSTRUMENTS
}


def get_instrument(
    isin: str,
) -> SovereignInstrument:
    """
    Return one approved instrument by ISIN.
    """
    normalised_isin = isin.strip().upper()

    try:
        return INSTRUMENTS_BY_ISIN[
            normalised_isin
        ]
    except KeyError as error:
        raise InstrumentRegistryValidationError(
            f"Unknown sovereign instrument ISIN: {normalised_isin}"
        ) from error


def instruments_for_country(
    country: SovereignCountry,
) -> tuple[SovereignInstrument, ...]:
    """
    Return approved instruments for one country.
    """
    return tuple(
        instrument
        for instrument in SOVEREIGN_INSTRUMENTS
        if instrument.country == country
    )


def instrument_for_country_tenor(
    country: SovereignCountry,
    benchmark_tenor_years: int,
) -> SovereignInstrument:
    """
    Return the approved benchmark instrument for a country and tenor.
    """
    matches = tuple(
        instrument
        for instrument in SOVEREIGN_INSTRUMENTS
        if (
            instrument.country == country
            and instrument.benchmark_tenor_years
            == benchmark_tenor_years
        )
    )

    if not matches:
        raise InstrumentRegistryValidationError(
            "No approved instrument exists for "
            f"{country.value} {benchmark_tenor_years}Y."
        )

    if len(matches) > 1:
        raise InstrumentRegistryValidationError(
            "More than one approved instrument exists for "
            f"{country.value} {benchmark_tenor_years}Y."
        )

    return matches[0]


def validate_registry(
    instruments: tuple[
        SovereignInstrument,
        ...,
    ] = SOVEREIGN_INSTRUMENTS,
) -> None:
    """
    Validate uniqueness and benchmark coverage for a registry.
    """
    if not instruments:
        raise InstrumentRegistryValidationError(
            "The sovereign instrument registry must not be empty."
        )

    isins = [
        instrument.isin
        for instrument in instruments
    ]

    if len(
        isins
    ) != len(
        set(
            isins
        )
    ):
        raise InstrumentRegistryValidationError(
            "The sovereign instrument registry contains duplicate ISINs."
        )

    country_tenor_pairs = [
        (
            instrument.country,
            instrument.benchmark_tenor_years,
        )
        for instrument in instruments
    ]

    if len(
        country_tenor_pairs
    ) != len(
        set(
            country_tenor_pairs
        )
    ):
        raise InstrumentRegistryValidationError(
            "The sovereign instrument registry contains duplicate "
            "country-tenor pairs."
        )


def registry_to_frame(
    instruments: tuple[
        SovereignInstrument,
        ...,
    ] = SOVEREIGN_INSTRUMENTS,
) -> pd.DataFrame:
    """
    Convert the approved registry into a dashboard-friendly table.
    """
    validate_registry(
        instruments
    )

    rows = [
        {
            "isin": instrument.isin,
            "display_name": instrument.display_name,
            "country": instrument.country.value,
            "country_code": instrument.country_code,
            "issuer": instrument.issuer,
            "security_type": instrument.security_type.value,
            "issue_date": instrument.issue_date,
            "maturity_date": instrument.maturity_date,
            "annual_coupon_percent": (
                instrument.annual_coupon_rate
                * 100.0
            ),
            "coupon_frequency": instrument.coupon_frequency,
            "benchmark_tenor_years": (
                instrument.benchmark_tenor_years
            ),
            "currency": instrument.currency,
            "face_value": instrument.face_value,
            "source_name": instrument.source_name,
            "source_locator": instrument.source_locator,
            "source_checked_date": instrument.source_checked_date,
            "data_status": instrument.data_status.value,
        }
        for instrument in instruments
    ]

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "country",
            "benchmark_tenor_years",
        ]
    ).reset_index(
        drop=True
    )


validate_registry()