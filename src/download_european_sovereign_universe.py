from __future__ import annotations

import csv
import re
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Final


DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "data/reference/european_sovereign_universe.csv"
)

HTTP_TIMEOUT_SECONDS: Final[int] = 25
DATA_STATUS: Final[str] = "OFFICIAL_REFERENCE"

SOURCE_URLS: Final[dict[str, str]] = {
    "DE": (
        "https://www.deutsche-finanzagentur.de/"
        "bundeswertpapiere/handel/umlaufende-bundeswertpapiere"
    ),
    "FR": "https://www.aft.gouv.fr/en/encours-detaille-oat",
    "ES": (
        "https://tesoro.es/en/deuda-publica/valores-del-tesoro/"
        "valores-en-circulacion"
    ),
    "BE": "https://www.debtagency.be/en/productolostatistics",
}

SOURCE_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "DE": (
        SOURCE_URLS["DE"],
    ),
    "FR": (
        SOURCE_URLS["FR"],
        "https://www.aft.gouv.fr/fr/encours-detaille-oat",
    ),
    "ES": (
        SOURCE_URLS["ES"],
        (
            "https://tesoro.es/deuda-publica/valores-del-tesoro/"
            "valores-en-circulacion"
        ),
        (
            "https://www.tesoro.es/deuda-publica/valores-del-tesoro/"
            "valores-en-circulacion"
        ),
    ),
    "BE": (
        SOURCE_URLS["BE"],
    ),
}

SOURCE_NAMES: Final[dict[str, str]] = {
    "DE": "German Finance Agency",
    "FR": "Agence France Trésor",
    "ES": "Tesoro Público",
    "BE": "Belgian Debt Agency",
}


class SovereignUniverseError(RuntimeError):
    """Base exception for official sovereign-universe ingestion."""


class SovereignUniverseDownloadError(SovereignUniverseError):
    """Raised when an official issuer page cannot be downloaded."""


class SovereignUniverseParseError(SovereignUniverseError):
    """Raised when an issuer page cannot be parsed safely."""


@dataclass(frozen=True)
class SovereignUniverseInstrument:
    country_code: str
    country_name: str
    issuer: str
    instrument_type: str
    isin: str
    coupon_percent: float | None
    maturity_date: date
    outstanding_eur: float | None
    source_name: str
    source_url: str
    data_status: str = DATA_STATUS


class _TableTextParser(HTMLParser):
    """
    Convert HTML table content into row-oriented text while ignoring scripts/styles.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._in_row = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row_cells: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lower = tag.lower()

        if lower in {"script", "style"}:
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        if lower == "tr":
            self._in_row = True
            self._row_cells = []
            return

        if self._in_row and lower in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        lower = tag.lower()

        if lower in {"script", "style"}:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return

        if self._ignored_depth:
            return

        if self._in_row and lower in {"td", "th"} and self._in_cell:
            text = " ".join(
                " ".join(
                    self._cell_parts
                ).split()
            )
            self._row_cells.append(
                text
            )
            self._cell_parts = []
            self._in_cell = False
            return

        if lower == "tr" and self._in_row:
            if any(
                cell.strip()
                for cell in self._row_cells
            ):
                self.rows.append(
                    list(
                        self._row_cells
                    )
                )

            self._row_cells = []
            self._in_row = False
            self._in_cell = False

    def handle_data(
        self,
        data: str,
    ) -> None:
        if (
            self._ignored_depth
            or not self._in_cell
        ):
            return

        text = " ".join(
            data.split()
        )

        if text:
            self._cell_parts.append(
                text
            )


def table_rows(
    html: str,
) -> tuple[
    tuple[str, ...],
    ...,
]:
    parser = _TableTextParser()
    parser.feed(
        html
    )

    return tuple(
        tuple(
            row
        )
        for row in parser.rows
    )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if (
            tag.lower() in {"script", "style"}
            and self._ignored_depth > 0
        ):
            self._ignored_depth -= 1

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._ignored_depth:
            return

        text = " ".join(
            data.split()
        )

        if text:
            self.parts.append(
                text
            )


def visible_text(
    html: str,
) -> str:
    parser = _VisibleTextParser()
    parser.feed(
        html
    )

    return " ".join(
        parser.parts
    )


def _normalise_text(
    value: str,
) -> str:
    return " ".join(
        value.replace(
            "\xa0",
            " ",
        ).split()
    )


def _parse_number(
    value: str,
) -> float | None:
    """
    Parse either European or Anglo numeric formatting.

    Examples:
        28.677,5           -> 28677.5
        24,030,000,000.00  -> 24030000000.0
        16.485.000.000,00  -> 16485000000.0
        19.000 Mio. €      -> 19000.0
        2,70 %             -> 2.70
    """
    text = _normalise_text(
        value
    )

    if not text:
        return None

    match = re.search(
        r"[+-]?\d[\d.,]*",
        text,
    )

    if match is None:
        return None

    numeric = match.group(
        0
    )

    comma_count = numeric.count(
        ","
    )
    dot_count = numeric.count(
        "."
    )

    if (
        comma_count > 0
        and dot_count > 0
    ):
        last_comma = numeric.rfind(
            ","
        )
        last_dot = numeric.rfind(
            "."
        )

        if last_dot > last_comma:
            # Anglo style: 24,030,000,000.00
            normalised = numeric.replace(
                ",",
                "",
            )
        else:
            # European style: 16.485.000.000,00
            normalised = (
                numeric
                .replace(
                    ".",
                    "",
                )
                .replace(
                    ",",
                    ".",
                )
            )

    elif comma_count > 0:
        groups = numeric.split(
            ","
        )

        if (
            comma_count > 1
            and all(
                len(
                    group
                ) == 3
                for group in groups[
                    1:
                ]
            )
        ):
            normalised = numeric.replace(
                ",",
                "",
            )
        else:
            normalised = numeric.replace(
                ",",
                ".",
            )

    elif dot_count > 0:
        groups = numeric.split(
            "."
        )

        if (
            dot_count > 1
            and all(
                len(
                    group
                ) == 3
                for group in groups[
                    1:
                ]
            )
        ):
            normalised = numeric.replace(
                ".",
                "",
            )
        elif (
            dot_count == 1
            and len(
                groups[
                    1
                ]
            ) == 3
            and any(
                token.lower().startswith(
                    ("mio", "million")
                )
                for token in text.split()
            )
        ):
            # German issuer tables use e.g. "19.000 Mio. €".
            normalised = numeric.replace(
                ".",
                "",
            )
        else:
            normalised = numeric

    else:
        normalised = numeric

    return float(
        normalised
    )


def _parse_european_date(
    value: str,
) -> date | None:
    text = _normalise_text(
        value
    )

    for fmt in (
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            pass

    return None


def _parse_english_bond_date(
    value: str,
) -> date | None:
    text = _normalise_text(
        value
    )

    for fmt in (
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            pass

    return None


def _extract_isin(
    value: str,
    *,
    prefix: str,
) -> str | None:
    match = re.search(
        rf"\b({re.escape(prefix)}[A-Z0-9]{{10}})\b",
        value.upper(),
    )

    if match is None:
        return None

    return match.group(
        1
    )


def _country_name(
    country_code: str,
) -> str:
    return {
        "DE": "Germany",
        "FR": "France",
        "ES": "Spain",
        "BE": "Belgium",
    }[
        country_code
    ]


def _issuer_name(
    country_code: str,
) -> str:
    return {
        "DE": "Federal Republic of Germany",
        "FR": "French Republic",
        "ES": "Kingdom of Spain",
        "BE": "Kingdom of Belgium",
    }[
        country_code
    ]


def parse_germany(
    html: str,
) -> tuple[
    SovereignUniverseInstrument,
    ...,
]:
    instruments: list[
        SovereignUniverseInstrument
    ] = []

    for row in table_rows(
        html
    ):
        if len(
            row
        ) < 6:
            continue

        isin = _extract_isin(
            " ".join(
                row
            ),
            prefix="DE",
        )

        if isin is None:
            continue

        name = _normalise_text(
            row[
                0
            ]
        )
        maturity = _parse_european_date(
            row[
                1
            ]
        )

        if maturity is None:
            continue

        if "Bubill" in name:
            instrument_type = "BUBILL"
        elif "Schatz" in name:
            instrument_type = "SCHATZ"
        elif "Bobl" in name:
            instrument_type = "BOBL"
        elif "Bund" in name:
            instrument_type = "BUND"
        else:
            instrument_type = "GERMAN_GOVERNMENT"

        coupon = _parse_number(
            row[
                2
            ]
        )

        outstanding_millions = _parse_number(
            row[
                3
            ]
        )

        outstanding_eur = (
            outstanding_millions
            * 1_000_000.0
            if outstanding_millions is not None
            else None
        )

        instruments.append(
            SovereignUniverseInstrument(
                country_code="DE",
                country_name="Germany",
                issuer=_issuer_name(
                    "DE"
                ),
                instrument_type=instrument_type,
                isin=isin,
                coupon_percent=coupon,
                maturity_date=maturity,
                outstanding_eur=outstanding_eur,
                source_name=SOURCE_NAMES[
                    "DE"
                ],
                source_url=SOURCE_URLS[
                    "DE"
                ],
            )
        )

    if not instruments:
        raise SovereignUniverseParseError(
            "No German sovereign instruments were parsed."
        )

    return tuple(
        instruments
    )


def _france_instrument_from_fields(
    *,
    isin: str,
    description: str,
    outstanding_text: str,
) -> SovereignUniverseInstrument | None:
    bond_pattern = re.compile(
        (
            r"(?:GREEN\s+)?OAT\s+"
            r"(?:(?P<coupon>\d+(?:[.,]\d+)?)\s*%|zero\s+coupon)"
            r".*?"
            r"(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})"
        ),
        flags=re.IGNORECASE,
    )

    match = bond_pattern.search(
        description
    )

    if match is None:
        return None

    maturity = _parse_english_bond_date(
        match.group(
            "date"
        )
    )

    if maturity is None:
        return None

    coupon_text = match.group(
        "coupon"
    )

    coupon = (
        _parse_number(
            coupon_text
        )
        if coupon_text is not None
        else 0.0
    )

    outstanding = _parse_number(
        outstanding_text
    )

    return SovereignUniverseInstrument(
        country_code="FR",
        country_name="France",
        issuer=_issuer_name(
            "FR"
        ),
        instrument_type="OAT",
        isin=isin,
        coupon_percent=coupon,
        maturity_date=maturity,
        outstanding_eur=outstanding,
        source_name=SOURCE_NAMES[
            "FR"
        ],
        source_url=SOURCE_URLS[
            "FR"
        ],
    )


def parse_france(
    html: str,
) -> tuple[
    SovereignUniverseInstrument,
    ...,
]:
    instruments: dict[
        str,
        SovereignUniverseInstrument,
    ] = {}

    # First use actual HTML table rows when present.
    for row in table_rows(
        html
    ):
        if len(
            row
        ) < 3:
            continue

        isin = _extract_isin(
            " ".join(
                row
            ),
            prefix="FR",
        )

        if isin is None:
            continue

        instrument = _france_instrument_from_fields(
            isin=isin,
            description=_normalise_text(
                row[
                    1
                ]
            ),
            outstanding_text=row[
                2
            ],
        )

        if instrument is not None:
            instruments[
                isin
            ] = instrument

    # AFT's live page can be rendered without conventional <tr>/<td> markup.
    # Parse visible text as a second, independent route rather than returning
    # zero instruments silently.
    text = visible_text(
        html
    )

    text_pattern = re.compile(
        (
            r"(?P<isin>FR[A-Z0-9]{10})\s+"
            r"(?P<description>"
            r"(?:GREEN\s+)?OAT\s+"
            r"(?:(?:\d+(?:[.,]\d+)?)\s*%|zero\s+coupon)"
            r".*?"
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
            r")\s+"
            r"(?P<outstanding>\d[\d,.\s]*)"
        ),
        flags=re.IGNORECASE,
    )

    for match in text_pattern.finditer(
        text
    ):
        isin = match.group(
            "isin"
        ).upper()

        instrument = _france_instrument_from_fields(
            isin=isin,
            description=_normalise_text(
                match.group(
                    "description"
                )
            ),
            outstanding_text=match.group(
                "outstanding"
            ),
        )

        if instrument is not None:
            instruments[
                isin
            ] = instrument

    if not instruments:
        raise SovereignUniverseParseError(
            "No French OAT instruments were parsed from the live AFT page."
        )

    return tuple(
        sorted(
            instruments.values(),
            key=lambda item: (
                item.maturity_date,
                item.isin,
            ),
        )
    )


def _parse_spain_date(
    value: str,
) -> date | None:
    european = _parse_european_date(
        value
    )

    if european is not None:
        return european

    text = _normalise_text(
        value
    )

    try:
        return datetime.strptime(
            text,
            "%m/%d/%Y",
        ).date()
    except ValueError:
        return None


def _build_spanish_instrument(
    *,
    descriptor: str,
    maturity_text: str,
    outstanding_text: str,
) -> SovereignUniverseInstrument | None:
    isin = _extract_isin(
        descriptor,
        prefix="ES",
    )

    if (
        isin is None
        or isin.startswith(
            "ES0L"
        )
    ):
        return None

    maturity = _parse_spain_date(
        maturity_text
    )

    if maturity is None:
        return None

    outstanding_millions = _parse_number(
        outstanding_text
    )

    coupon_match = re.search(
        r"\b([BO])\s+(?:€i\s+)?(\d+(?:[.,]\d+)?)",
        descriptor,
        flags=re.IGNORECASE,
    )

    coupon = (
        _parse_number(
            coupon_match.group(
                2
            )
        )
        if coupon_match is not None
        else None
    )

    instrument_type = "SPGB"

    if coupon_match is not None:
        instrument_type = (
            "BONO"
            if coupon_match.group(
                1
            ).upper() == "B"
            else "OBLIGACION"
        )

    return SovereignUniverseInstrument(
        country_code="ES",
        country_name="Spain",
        issuer=_issuer_name(
            "ES"
        ),
        instrument_type=instrument_type,
        isin=isin,
        coupon_percent=coupon,
        maturity_date=maturity,
        outstanding_eur=(
            outstanding_millions
            * 1_000_000.0
            if outstanding_millions is not None
            else None
        ),
        source_name=SOURCE_NAMES[
            "ES"
        ],
        source_url=SOURCE_URLS[
            "ES"
        ],
    )


def parse_spain(
    html: str,
) -> tuple[
    SovereignUniverseInstrument,
    ...,
]:
    instruments: dict[
        str,
        SovereignUniverseInstrument,
    ] = {}

    for row in table_rows(
        html
    ):
        if len(
            row
        ) < 3:
            continue

        instrument = _build_spanish_instrument(
            descriptor=row[
                0
            ],
            maturity_text=row[
                1
            ],
            outstanding_text=row[
                2
            ],
        )

        if instrument is not None:
            instruments[
                instrument.isin
            ] = instrument

    # Fallback for Drupal/text-oriented rendering.
    text = visible_text(
        html
    )

    text_pattern = re.compile(
        (
            r"(?P<descriptor>"
            r"ES[A-Z0-9]{10}\s+[BO]\s+(?:€i\s+)?\d+(?:[.,]\d+)?"
            r"(?:\*{1,2})?"
            r")\s+"
            r"(?P<maturity>\d{2}/\d{2}/\d{4})\s+"
            r"(?P<outstanding>\d[\d.,]*)"
        ),
        flags=re.IGNORECASE,
    )

    for match in text_pattern.finditer(
        text
    ):
        instrument = _build_spanish_instrument(
            descriptor=match.group(
                "descriptor"
            ),
            maturity_text=match.group(
                "maturity"
            ),
            outstanding_text=match.group(
                "outstanding"
            ),
        )

        if instrument is not None:
            instruments[
                instrument.isin
            ] = instrument

    if not instruments:
        raise SovereignUniverseParseError(
            "No Spanish Bonos/Obligaciones were parsed from the live Tesoro page."
        )

    return tuple(
        sorted(
            instruments.values(),
            key=lambda item: (
                item.maturity_date,
                item.isin,
            ),
        )
    )


def parse_belgium(
    html: str,
) -> tuple[
    SovereignUniverseInstrument,
    ...,
]:
    instruments: list[
        SovereignUniverseInstrument
    ] = []

    for row in table_rows(
        html
    ):
        if len(
            row
        ) < 5:
            continue

        isin = _extract_isin(
            " ".join(
                row
            ),
            prefix="BE",
        )

        if isin is None:
            continue

        maturity = _parse_european_date(
            row[
                0
            ]
        )

        if maturity is None:
            continue

        coupon = _parse_number(
            row[
                1
            ]
        )

        outstanding = _parse_number(
            row[
                4
            ]
        )

        instruments.append(
            SovereignUniverseInstrument(
                country_code="BE",
                country_name="Belgium",
                issuer=_issuer_name(
                    "BE"
                ),
                instrument_type="OLO",
                isin=isin,
                coupon_percent=coupon,
                maturity_date=maturity,
                outstanding_eur=outstanding,
                source_name=SOURCE_NAMES[
                    "BE"
                ],
                source_url=SOURCE_URLS[
                    "BE"
                ],
            )
        )

    if not instruments:
        raise SovereignUniverseParseError(
            "No Belgian OLO instruments were parsed."
        )

    return tuple(
        instruments
    )


PARSERS = {
    "DE": parse_germany,
    "FR": parse_france,
    "ES": parse_spain,
    "BE": parse_belgium,
}


def _browser_headers(
    *,
    referer: str | None = None,
) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/152.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-GB,en;q=0.9,fr;q=0.8,es;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }

    if referer:
        headers[
            "Referer"
        ] = referer

    return headers


def _urllib_ssl_context() -> ssl.SSLContext:
    """
    Prefer certifi's CA bundle when available.

    This fixes environments where Python's bundled/OpenSSL trust store cannot
    validate an otherwise valid public issuer certificate. It does not disable
    certificate verification.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()

    return ssl.create_default_context(
        cafile=certifi.where()
    )


def _download_with_urllib(
    *,
    url: str,
    referer: str | None = None,
) -> str:
    request = urllib.request.Request(
        url,
        headers=_browser_headers(
            referer=referer
        ),
    )

    with urllib.request.urlopen(
        request,
        timeout=HTTP_TIMEOUT_SECONDS,
        context=_urllib_ssl_context(),
    ) as response:
        return response.read().decode(
            "utf-8",
            errors="replace",
        )


def _curl_origin(
    url: str,
) -> str:
    match = re.match(
        r"^(https?://[^/]+)",
        url,
        flags=re.IGNORECASE,
    )

    return (
        match.group(
            1
        )
        if match is not None
        else url
    )


def _download_with_windows_curl(
    *,
    url: str,
) -> str:
    """
    Windows fallback for issuer sites that reject Python/OpenSSL clients.

    curl.exe uses the Windows networking/certificate stack on normal Windows
    installations. A cookie-seeding request to the issuer homepage is made
    first because some government sites require a browser-like session before
    allowing access to a deep page.
    """
    origin = _curl_origin(
        url
    )

    with tempfile.TemporaryDirectory(
        prefix="repolens_sovereign_"
    ) as temp_directory:
        cookie_path = Path(
            temp_directory
        ) / "cookies.txt"

        common = [
            "curl.exe",
            "--location",
            "--compressed",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            str(
                HTTP_TIMEOUT_SECONDS
            ),
            "--user-agent",
            _browser_headers()[
                "User-Agent"
            ],
            "--header",
            (
                "Accept: "
                + _browser_headers()[
                    "Accept"
                ]
            ),
            "--header",
            "Accept-Language: en-GB,en;q=0.9,fr;q=0.8,es;q=0.7",
        ]

        # Seed normal session cookies. Failure here is non-fatal because some
        # sites do not use cookies at all.
        subprocess.run(
            [
                *common,
                "--cookie-jar",
                str(
                    cookie_path
                ),
                origin,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        command = [
            *common,
            "--cookie",
            str(
                cookie_path
            ),
            "--cookie-jar",
            str(
                cookie_path
            ),
            "--referer",
            (
                origin
                + "/"
            ),
            "--header",
            "Sec-Fetch-Dest: document",
            "--header",
            "Sec-Fetch-Mode: navigate",
            "--header",
            "Sec-Fetch-Site: same-origin",
            "--header",
            "Sec-Fetch-User: ?1",
            url,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if result.returncode != 0:
            detail = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"curl exit code {result.returncode}"
            )

            raise SovereignUniverseDownloadError(
                f"curl fallback failed for {url}: {detail}"
            )

        if not result.stdout.strip():
            raise SovereignUniverseDownloadError(
                f"curl fallback returned an empty response for {url}."
            )

        return result.stdout


def download_page(
    *,
    country_code: str,
) -> str:
    attempts: list[
        str
    ] = []

    candidates = SOURCE_CANDIDATES[
        country_code
    ]

    for url in candidates:
        origin = _curl_origin(
            url
        )

        try:
            return _download_with_urllib(
                url=url,
                referer=(
                    origin
                    + "/"
                ),
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionError,
            OSError,
            ssl.SSLError,
        ) as error:
            attempts.append(
                f"urllib {url}: {error}"
            )

        try:
            return _download_with_windows_curl(
                url=url
            )
        except (
            SovereignUniverseDownloadError,
            FileNotFoundError,
            subprocess.SubprocessError,
            OSError,
        ) as error:
            attempts.append(
                f"curl {url}: {error}"
            )

    raise SovereignUniverseDownloadError(
        f"Could not download {country_code} sovereign universe after "
        f"{len(attempts)} transport attempt(s): "
        + " | ".join(
            attempts
        )
    )


def download_all_universes() -> tuple[
    tuple[
        SovereignUniverseInstrument,
        ...,
    ],
    tuple[
        str,
        ...,
    ],
]:
    instruments: list[
        SovereignUniverseInstrument
    ] = []

    failures: list[
        str
    ] = []

    for country_code in (
        "DE",
        "FR",
        "ES",
        "BE",
    ):
        try:
            html = download_page(
                country_code=country_code
            )

            parsed = PARSERS[
                country_code
            ](
                html
            )

            instruments.extend(
                parsed
            )
        except SovereignUniverseError as error:
            failures.append(
                f"{country_code}: {error}"
            )

    if not instruments:
        raise SovereignUniverseError(
            "No official European sovereign instruments were downloaded. "
            + "; ".join(
                failures
            )
        )

    unique: dict[
        str,
        SovereignUniverseInstrument,
    ] = {}

    for instrument in instruments:
        unique[
            instrument.isin
        ] = instrument

    return (
        tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.country_code,
                    item.maturity_date,
                    item.isin,
                ),
            )
        ),
        tuple(
            failures
        ),
    )


def write_universe(
    *,
    instruments: tuple[
        SovereignUniverseInstrument,
        ...,
    ],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "country_code",
        "country_name",
        "issuer",
        "instrument_type",
        "isin",
        "coupon_percent",
        "maturity_date",
        "outstanding_eur",
        "source_name",
        "source_url",
        "data_status",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for instrument in instruments:
            writer.writerow(
                {
                    "country_code": instrument.country_code,
                    "country_name": instrument.country_name,
                    "issuer": instrument.issuer,
                    "instrument_type": instrument.instrument_type,
                    "isin": instrument.isin,
                    "coupon_percent": (
                        ""
                        if instrument.coupon_percent is None
                        else f"{instrument.coupon_percent:.6f}"
                    ),
                    "maturity_date": instrument.maturity_date.isoformat(),
                    "outstanding_eur": (
                        ""
                        if instrument.outstanding_eur is None
                        else f"{instrument.outstanding_eur:.2f}"
                    ),
                    "source_name": instrument.source_name,
                    "source_url": instrument.source_url,
                    "data_status": instrument.data_status,
                }
            )


def main() -> None:
    instruments, failures = download_all_universes()

    write_universe(
        instruments=instruments
    )

    print(
        f"Saved {len(instruments)} official sovereign instrument(s)."
    )

    country_counts: dict[
        str,
        int,
    ] = {}

    country_outstanding: dict[
        str,
        float,
    ] = {}

    for instrument in instruments:
        country_counts[
            instrument.country_code
        ] = (
            country_counts.get(
                instrument.country_code,
                0,
            )
            + 1
        )

        if instrument.outstanding_eur is not None:
            country_outstanding[
                instrument.country_code
            ] = (
                country_outstanding.get(
                    instrument.country_code,
                    0.0,
                )
                + instrument.outstanding_eur
            )

    for country_code in sorted(
        country_counts
    ):
        outstanding = country_outstanding.get(
            country_code
        )

        outstanding_text = (
            f"€{outstanding / 1_000_000_000:.1f}bn"
            if outstanding is not None
            else "N/A"
        )

        print(
            f"  {country_code} | "
            f"{country_counts[country_code]} instruments | "
            f"{outstanding_text} parsed outstanding"
        )

    if failures:
        print(
            "Issuer-source failures:"
        )

        for failure in failures:
            print(
                f"  {failure}"
            )


if __name__ == "__main__":
    main()