

import csv
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from src.sovereign_instrument_catalog import all_instruments


EUROPEAN_UNIVERSE_PATH = Path(
    "data/reference/european_sovereign_universe.csv"
)

ITALY_UNIVERSE_PATH = Path(
    "data/reference/italy_nominal_btp_universe.csv"
)

UK_GILT_UNIVERSE_PATH = Path(
    "data/reference/uk_conventional_gilt_universe.csv"
)


class DeskUniverseError(RuntimeError):
    """Raised when a sovereign reference-universe file is malformed."""


class DeskCountry(Enum):
    AUSTRIA = "Austria"
    BELGIUM = "Belgium"
    EUROPEAN_UNION = "European Union"
    FRANCE = "France"
    GERMANY = "Germany"
    ITALY = "Italy"
    NETHERLANDS = "Netherlands"
    SPAIN = "Spain"
    UNITED_KINGDOM = "United Kingdom"


@dataclass(frozen=True)
class DeskSovereignInstrument:
    """
    Minimal desk-facing sovereign instrument representation.

    It deliberately exposes the attributes already consumed by the
    Sovereign Observations page while allowing the universe to be populated
    dynamically from reference files instead of a tiny hard-coded catalogue.
    """

    country: DeskCountry
    isin: str
    display_name: str
    maturity_date: date
    benchmark_tenor_years: int
    instrument_type: str
    currency: str
    coupon_percent: float | None
    source_name: str
    data_status: str

    @property
    def years_to_maturity(self) -> float:
        days = (
            self.maturity_date
            - date.today()
        ).days

        return max(
            days / 365.25,
            0.0,
        )


COUNTRY_NAME_MAP = {
    "AT": DeskCountry.AUSTRIA,
    "AUSTRIA": DeskCountry.AUSTRIA,
    "BE": DeskCountry.BELGIUM,
    "BELGIUM": DeskCountry.BELGIUM,
    "EU": DeskCountry.EUROPEAN_UNION,
    "EUROPEAN UNION": DeskCountry.EUROPEAN_UNION,
    "FR": DeskCountry.FRANCE,
    "FRANCE": DeskCountry.FRANCE,
    "DE": DeskCountry.GERMANY,
    "GERMANY": DeskCountry.GERMANY,
    "IT": DeskCountry.ITALY,
    "ITALY": DeskCountry.ITALY,
    "NL": DeskCountry.NETHERLANDS,
    "NETHERLANDS": DeskCountry.NETHERLANDS,
    "ES": DeskCountry.SPAIN,
    "SPAIN": DeskCountry.SPAIN,
    "GB": DeskCountry.UNITED_KINGDOM,
    "UK": DeskCountry.UNITED_KINGDOM,
    "UNITED KINGDOM": DeskCountry.UNITED_KINGDOM,
}


def _text(
    row: dict[str, str],
    *keys: str,
) -> str:
    for key in keys:
        value = (
            row.get(
                key
            )
            or ""
        ).strip()

        if value:
            return value

    return ""


def _float_or_none(
    value: str,
) -> float | None:
    text = (
        value
        .strip()
        .replace(
            ",",
            "",
        )
    )

    if not text:
        return None

    try:
        return float(
            text
        )
    except ValueError:
        return None


def _parse_iso_date(
    value: str,
) -> date:
    try:
        return date.fromisoformat(
            value.strip()[
                :10
            ]
        )
    except ValueError as error:
        raise DeskUniverseError(
            f"Invalid maturity date: {value!r}"
        ) from error


def _country_from_row(
    row: dict[str, str],
) -> DeskCountry:
    candidates = (
        _text(
            row,
            "country_name",
        ),
        _text(
            row,
            "country_code",
        ),
    )

    for candidate in candidates:
        mapped = COUNTRY_NAME_MAP.get(
            candidate.upper()
        )

        if mapped is not None:
            return mapped

    raise DeskUniverseError(
        "Unsupported or missing sovereign country in row: "
        f"{row}"
    )


def _sector_from_maturity(
    maturity: date,
) -> int:
    """
    Desk-friendly maturity bucket.

    These are buckets, not claims that each bond is an on-the-run benchmark.
    """
    years = max(
        (
            maturity
            - date.today()
        ).days
        / 365.25,
        0.0,
    )

    if years <= 1.5:
        return 1

    if years <= 3.5:
        return 2

    if years <= 6.0:
        return 5

    if years <= 8.5:
        return 7

    if years <= 12.5:
        return 10

    if years <= 17.5:
        return 15

    if years <= 25.0:
        return 20

    if years <= 40.0:
        return 30

    return 50


def _display_name(
    *,
    country: DeskCountry,
    instrument_type: str,
    coupon_percent: float | None,
    maturity: date,
    source_name: str,
    raw_name: str,
) -> str:
    if raw_name:
        return raw_name

    coupon_text = (
        f"{coupon_percent:.3f}% "
        if coupon_percent is not None
        else ""
    )

    maturity_text = maturity.strftime(
        "%b %Y"
    )

    type_text = (
        instrument_type.strip()
        or "Sovereign"
    )

    return (
        f"{country.value} "
        f"{coupon_text}{type_text} "
        f"{maturity_text}"
    ).strip()


def _read_reference_file(
    path: Path,
) -> tuple[
    DeskSovereignInstrument,
    ...,
]:
    if not path.exists():
        return ()

    instruments: list[
        DeskSovereignInstrument
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            isin = _text(
                row,
                "isin",
            ).upper()

            maturity_text = _text(
                row,
                "maturity_date",
            )

            if not isin or not maturity_text:
                continue

            try:
                country = _country_from_row(
                    row
                )

                maturity = _parse_iso_date(
                    maturity_text
                )
            except DeskUniverseError as error:
                raise DeskUniverseError(
                    f"{path} row {row_number}: {error}"
                ) from error

            coupon = _float_or_none(
                _text(
                    row,
                    "coupon_percent",
                    "coupon_display_percent",
                )
            )

            instrument_type = _text(
                row,
                "instrument_type",
            )

            raw_name = _text(
                row,
                "display_name",
                "description",
                "name",
                "instrument_name",
            )

            source_name = _text(
                row,
                "source_name",
            ) or path.name

            data_status = _text(
                row,
                "data_status",
            ) or "REFERENCE"

            currency = _text(
                row,
                "currency",
            ) or (
                "GBP"
                if country is DeskCountry.UNITED_KINGDOM
                else "EUR"
            )

            instruments.append(
                DeskSovereignInstrument(
                    country=country,
                    isin=isin,
                    display_name=_display_name(
                        country=country,
                        instrument_type=instrument_type,
                        coupon_percent=coupon,
                        maturity=maturity,
                        source_name=source_name,
                        raw_name=raw_name,
                    ),
                    maturity_date=maturity,
                    benchmark_tenor_years=_sector_from_maturity(
                        maturity
                    ),
                    instrument_type=instrument_type,
                    currency=currency,
                    coupon_percent=coupon,
                    source_name=source_name,
                    data_status=data_status,
                )
            )

    return tuple(
        instruments
    )


def _from_curated_catalogue() -> tuple[
    DeskSovereignInstrument,
    ...,
]:
    converted: list[
        DeskSovereignInstrument
    ] = []

    for instrument in all_instruments():
        country = COUNTRY_NAME_MAP.get(
            instrument.country.value.upper()
        )

        if country is None:
            continue

        converted.append(
            DeskSovereignInstrument(
                country=country,
                isin=instrument.isin,
                display_name=instrument.display_name,
                maturity_date=instrument.maturity_date,
                benchmark_tenor_years=instrument.benchmark_tenor_years,
                instrument_type="Sovereign",
                currency=(
                    "GBP"
                    if country is DeskCountry.UNITED_KINGDOM
                    else "EUR"
                ),
                coupon_percent=None,
                source_name="RepoLens curated catalogue",
                data_status="REFERENCE",
            )
        )

    return tuple(
        converted
    )


def load_desk_sovereign_universe(
    paths: Sequence[
        Path
    ] | None = None,
) -> tuple[
    DeskSovereignInstrument,
    ...,
]:
    """
    Load the best available reference universe.

    Dynamic reference files win over the legacy curated catalogue on duplicate
    ISINs. The catalogue remains only as a coverage fallback for countries not
    yet populated by a dynamic source.
    """
    if paths is None:
        paths = (
            EUROPEAN_UNIVERSE_PATH,
            ITALY_UNIVERSE_PATH,
            UK_GILT_UNIVERSE_PATH,
        )

    by_isin = {
        instrument.isin: instrument
        for instrument in _from_curated_catalogue()
    }

    for path in paths:
        for instrument in _read_reference_file(
            path
        ):
            by_isin[
                instrument.isin
            ] = instrument

    return tuple(
        sorted(
            by_isin.values(),
            key=lambda instrument: (
                instrument.country.value,
                instrument.maturity_date,
                instrument.isin,
            ),
        )
    )


def universe_country_names(
    instruments: Iterable[
        DeskSovereignInstrument
    ],
) -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(
            {
                instrument.country.value
                for instrument in instruments
            }
        )
    )


def universe_counts_by_country(
    instruments: Iterable[
        DeskSovereignInstrument
    ],
) -> dict[
    str,
    int,
]:
    counts: dict[
        str,
        int,
    ] = {}

    for instrument in instruments:
        country = instrument.country.value

        counts[
            country
        ] = (
            counts.get(
                country,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            counts.items()
        )
    )
