from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final


class MoneyMarketInstrumentError(RuntimeError):
    pass


class MoneyMarketInstrumentValidationError(MoneyMarketInstrumentError):
    pass


class MoneyMarketSecurityType(StrEnum):
    BUBILL = "Bubill"
    BTF = "BTF"
    BOT = "BOT"


class MoneyMarketDataStatus(StrEnum):
    OFFICIAL_REFERENCE = "OFFICIAL_REFERENCE"


class RemainingMaturityBucket(StrEnum):
    LT_1M = "<1M"
    M1_3 = "1-3M"
    M3_6 = "3-6M"
    M6_12 = "6-12M"
    Y1_2 = "1-2Y"
    GT_2Y = ">2Y"


@dataclass(frozen=True)
class SovereignDiscountSecurity:
    isin: str
    display_name: str
    country: str
    country_code: str
    issuer: str
    security_type: MoneyMarketSecurityType
    issue_date: date
    maturity_date: date
    currency: str
    redemption_value_per_100: float
    interest_day_count_basis: int
    source_name: str
    source_locator: str
    source_checked_date: date
    data_status: MoneyMarketDataStatus

    def __post_init__(self) -> None:
        normalised_isin = self.isin.strip().upper()

        if len(normalised_isin) != 12 or not normalised_isin.isalnum():
            raise MoneyMarketInstrumentValidationError(
                "isin must contain exactly 12 alphanumeric characters."
            )

        if not self.display_name.strip():
            raise MoneyMarketInstrumentValidationError(
                "display_name must not be empty."
            )

        if len(self.country_code.strip()) != 2:
            raise MoneyMarketInstrumentValidationError(
                "country_code must contain exactly two characters."
            )

        if not self.issuer.strip():
            raise MoneyMarketInstrumentValidationError(
                "issuer must not be empty."
            )

        if self.issue_date >= self.maturity_date:
            raise MoneyMarketInstrumentValidationError(
                "issue_date must be before maturity_date."
            )

        if self.currency != "EUR":
            raise MoneyMarketInstrumentValidationError(
                "The first RepoLens money-market universe supports EUR only."
            )

        if self.redemption_value_per_100 <= 0.0:
            raise MoneyMarketInstrumentValidationError(
                "redemption_value_per_100 must be positive."
            )

        if self.interest_day_count_basis not in {360, 365}:
            raise MoneyMarketInstrumentValidationError(
                "interest_day_count_basis must be 360 or 365."
            )

        if not self.source_name.strip() or not self.source_locator.strip():
            raise MoneyMarketInstrumentValidationError(
                "source_name and source_locator must not be empty."
            )

    @property
    def is_zero_coupon(self) -> bool:
        return True

    def days_to_maturity(self, as_of_date: date) -> int:
        days = (self.maturity_date - as_of_date).days
        if days < 0:
            raise MoneyMarketInstrumentValidationError(
                "as_of_date must not be after maturity_date."
            )
        return days

    def remaining_maturity_bucket(
        self,
        as_of_date: date,
    ) -> RemainingMaturityBucket:
        days = self.days_to_maturity(as_of_date)

        if days < 31:
            return RemainingMaturityBucket.LT_1M
        if days <= 92:
            return RemainingMaturityBucket.M1_3
        if days <= 183:
            return RemainingMaturityBucket.M3_6
        if days <= 366:
            return RemainingMaturityBucket.M6_12
        if days <= 731:
            return RemainingMaturityBucket.Y1_2
        return RemainingMaturityBucket.GT_2Y

    def market_value_eur(
        self,
        face_value_eur: float,
        price_per_100: float,
    ) -> float:
        if face_value_eur <= 0.0 or price_per_100 <= 0.0:
            raise MoneyMarketInstrumentValidationError(
                "face_value_eur and price_per_100 must be positive."
            )
        return face_value_eur * price_per_100 / 100.0

    def redemption_value_eur(
        self,
        face_value_eur: float,
    ) -> float:
        if face_value_eur <= 0.0:
            raise MoneyMarketInstrumentValidationError(
                "face_value_eur must be positive."
            )
        return face_value_eur * self.redemption_value_per_100 / 100.0

    def pull_to_par_eur(
        self,
        face_value_eur: float,
        price_per_100: float,
    ) -> float:
        return self.redemption_value_eur(face_value_eur) - self.market_value_eur(
            face_value_eur=face_value_eur,
            price_per_100=price_per_100,
        )


GERMANY_BUBILL_NOV_2026: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="DE000BU0E352",
    display_name="Germany Bubill November 2026",
    country="Germany",
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=MoneyMarketSecurityType.BUBILL,
    issue_date=date(2025, 11, 17),
    maturity_date=date(2026, 11, 18),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="German Finance Agency",
    source_locator="https://www.deutsche-finanzagentur.de/en/federal-securities/factsheet/isin/DE000BU0E352",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)

GERMANY_BUBILL_FEB_2027: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="DE000BU0E386",
    display_name="Germany Bubill February 2027",
    country="Germany",
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=MoneyMarketSecurityType.BUBILL,
    issue_date=date(2026, 2, 16),
    maturity_date=date(2027, 2, 17),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="German Finance Agency",
    source_locator="https://www.deutsche-finanzagentur.de/en/federal-securities/factsheet/isin/DE000BU0E386",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)

GERMANY_BUBILL_MAY_2027: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="DE000BU0E410",
    display_name="Germany Bubill May 2027",
    country="Germany",
    country_code="DE",
    issuer="Federal Republic of Germany",
    security_type=MoneyMarketSecurityType.BUBILL,
    issue_date=date(2026, 5, 11),
    maturity_date=date(2027, 5, 12),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="German Finance Agency",
    source_locator="https://www.deutsche-finanzagentur.de/en/federal-securities/factsheet/isin/DE000BU0E410",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)


# For BTFs, issue_date records the first AFT auction date visible in the
# official security history. BTFs can subsequently be reopened at later auctions.
FRANCE_BTF_OCT_2026: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="FR0129704088",
    display_name="France BTF October 2026",
    country="France",
    country_code="FR",
    issuer="French Republic",
    security_type=MoneyMarketSecurityType.BTF,
    issue_date=date(2026, 7, 20),
    maturity_date=date(2026, 10, 28),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="Agence France Trésor",
    source_locator="https://www.aft.gouv.fr/en/titre/fr0129704088",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)

FRANCE_BTF_FEB_2027: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="FR0129704146",
    display_name="France BTF February 2027",
    country="France",
    country_code="FR",
    issuer="French Republic",
    security_type=MoneyMarketSecurityType.BTF,
    issue_date=date(2026, 7, 27),
    maturity_date=date(2027, 2, 10),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="Agence France Trésor",
    source_locator="https://www.aft.gouv.fr/en/titre/fr0129704146",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)

FRANCE_BTF_JUL_2027: Final[SovereignDiscountSecurity] = SovereignDiscountSecurity(
    isin="FR0129704179",
    display_name="France BTF July 2027",
    country="France",
    country_code="FR",
    issuer="French Republic",
    security_type=MoneyMarketSecurityType.BTF,
    issue_date=date(2026, 7, 13),
    maturity_date=date(2027, 7, 14),
    currency="EUR",
    redemption_value_per_100=100.0,
    interest_day_count_basis=360,
    source_name="Agence France Trésor",
    source_locator="https://www.aft.gouv.fr/en/titre/fr0129704179",
    source_checked_date=date(2026, 8, 15),
    data_status=MoneyMarketDataStatus.OFFICIAL_REFERENCE,
)


GERMAN_BUBILLS: Final[tuple[SovereignDiscountSecurity, ...]] = (
    GERMANY_BUBILL_NOV_2026,
    GERMANY_BUBILL_FEB_2027,
    GERMANY_BUBILL_MAY_2027,
)

FRENCH_BTFS: Final[tuple[SovereignDiscountSecurity, ...]] = (
    FRANCE_BTF_OCT_2026,
    FRANCE_BTF_FEB_2027,
    FRANCE_BTF_JUL_2027,
)

MONEY_MARKET_INSTRUMENTS: Final[tuple[SovereignDiscountSecurity, ...]] = (
    *GERMAN_BUBILLS,
    *FRENCH_BTFS,
)


def get_money_market_instrument(isin: str) -> SovereignDiscountSecurity:
    normalised_isin = isin.strip().upper()
    for instrument in MONEY_MARKET_INSTRUMENTS:
        if instrument.isin == normalised_isin:
            return instrument
    raise MoneyMarketInstrumentValidationError(
        f"Unknown money-market instrument ISIN: {normalised_isin}"
    )


def validate_money_market_universe(
    instruments: tuple[SovereignDiscountSecurity, ...] = MONEY_MARKET_INSTRUMENTS,
) -> None:
    if not instruments:
        raise MoneyMarketInstrumentValidationError(
            "The money-market instrument universe must not be empty."
        )

    isins = [instrument.isin for instrument in instruments]

    if len(isins) != len(set(isins)):
        raise MoneyMarketInstrumentValidationError(
            "The money-market instrument universe contains duplicate ISINs."
        )


validate_money_market_universe()