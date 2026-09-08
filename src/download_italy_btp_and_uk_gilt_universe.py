from __future__ import annotations

import csv
import random
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Final


ITALY_OUTPUT_PATH: Final[Path] = Path("data/reference/italy_nominal_btp_universe.csv")
UK_OUTPUT_PATH: Final[Path] = Path("data/reference/uk_conventional_gilt_universe.csv")
ITALY_MARKET_OUTPUT_PATH: Final[Path] = Path("data/market/italy_nominal_btp_market.csv")

BORSA_BTP_LIST_URL: Final[str] = (
    "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/"
    "lista.html?lang=it&page={page}"
)
BORSA_BTP_DETAIL_URL: Final[str] = (
    "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/"
    "scheda/{isin}-MOTX.html?lang=it"
)
DMO_GILTS_XML_URL: Final[str] = (
    "https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D1A"
)

HTTP_TIMEOUT_SECONDS: Final[int] = 25
ITALY_SOURCE_NAME: Final[str] = "Borsa Italiana MOT"
UK_SOURCE_NAME: Final[str] = "UK Debt Management Office"
PUBLIC_REFERENCE: Final[str] = "PUBLIC_REFERENCE"
OFFICIAL_REFERENCE: Final[str] = "OFFICIAL_REFERENCE"


class SovereignUniverseExpansionError(RuntimeError):
    pass


class SovereignUniverseExpansionDownloadError(SovereignUniverseExpansionError):
    pass


class SovereignUniverseExpansionParseError(SovereignUniverseExpansionError):
    pass


@dataclass(frozen=True)
class ItalyBtpObservation:
    isin: str
    description: str
    coupon_display_percent: float | None
    last_price: float | None
    maturity_date: date
    annual_coupon_percent: float | None = None
    periodic_coupon_percent: float | None = None
    coupon_frequency: str | None = None
    accrual_start_date: date | None = None
    first_coupon_date: date | None = None
    day_count_convention: str | None = None
    reference_price: float | None = None
    gross_yield_percent: float | None = None
    reference_date: date | None = None
    market_source_name: str | None = None
    source_name: str = ITALY_SOURCE_NAME
    data_status: str = PUBLIC_REFERENCE

    @property
    def is_strip(self) -> bool:
        upper = self.description.upper()
        return "STRIP" in upper or upper.startswith("BTPSTRIP")

    @property
    def is_inflation_linked(self) -> bool:
        upper = self.description.upper()
        return "BTPI" in upper or "BTP€I" in upper or "ITALIA" in upper

    @property
    def is_nominal_btp(self) -> bool:
        return (
            self.isin.startswith("IT")
            and self.description.upper().startswith("BTP")
            and not self.is_strip
            and not self.is_inflation_linked
        )

    @property
    def has_exact_reference_yield(self) -> bool:
        return (
            self.gross_yield_percent is not None
            and self.reference_date is not None
        )

    @property
    def has_pricing_terms(self) -> bool:
        return (
            self.annual_coupon_percent is not None
            and self.coupon_frequency is not None
            and self.day_count_convention is not None
            and self.maturity_date is not None
        )


@dataclass(frozen=True)
class ConventionalGilt:
    isin: str
    name: str
    coupon_percent: float | None
    first_issue_date: date | None
    maturity_date: date
    nominal_amount_outstanding_gbp: float | None
    source_name: str = UK_SOURCE_NAME
    data_status: str = OFFICIAL_REFERENCE


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored = 0
        self._in_row = False
        self._in_cell = False
        self._row: list[str] = []
        self._cell: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        lower = tag.lower()
        if lower in {"script", "style"}:
            self._ignored += 1
            return
        if self._ignored:
            return
        if lower == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and lower in {"td", "th"}:
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"script", "style"}:
            if self._ignored > 0:
                self._ignored -= 1
            return
        if self._ignored:
            return
        if self._in_row and self._in_cell and lower in {"td", "th"}:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = []
            self._in_cell = False
            return
        if self._in_row and lower == "tr":
            if any(cell.strip() for cell in self._row):
                self.rows.append(list(self._row))
            self._row = []
            self._in_row = False
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._ignored or not self._in_cell:
            return
        text = " ".join(data.split())
        if text:
            self._cell.append(text)


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/152.0.0.0 Safari/537.36 RepoLens/1.0"
        ),
        "Accept-Language": "en-GB,en;q=0.9,it;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _verified_ssl_context() -> ssl.SSLContext:
    """
    Build a verified TLS context.

    Prefer certifi's CA bundle when available. We do not disable certificate
    verification under any circumstances.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()

    return ssl.create_default_context(
        cafile=certifi.where()
    )


def _curl_download_text(
    url: str,
) -> str:
    """
    Windows curl fallback.

    curl.exe on Windows uses the Windows certificate store, which is useful
    when Python's OpenSSL trust store cannot validate a locally trusted
    certificate chain. Certificate verification remains enabled.
    """
    curl_path = shutil.which(
        "curl.exe"
    ) or shutil.which(
        "curl"
    )

    if curl_path is None:
        raise SovereignUniverseExpansionDownloadError(
            "curl is not available for transport fallback."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".download",
        delete=False,
    ) as temp_handle:
        temp_path = Path(
            temp_handle.name
        )

    try:
        completed = subprocess.run(
            [
                curl_path,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                str(
                    HTTP_TIMEOUT_SECONDS
                ),
                "--max-time",
                str(
                    HTTP_TIMEOUT_SECONDS
                    + 10
                ),
                "--retry",
                "3",
                "--retry-delay",
                "1",
                "--retry-all-errors",
                "--user-agent",
                _headers()[
                    "User-Agent"
                ],
                "--header",
                "Accept-Language: en-GB,en;q=0.9,it;q=0.8",
                "--output",
                str(
                    temp_path
                ),
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"curl exit code {completed.returncode}"
            )

            raise SovereignUniverseExpansionDownloadError(
                f"curl fallback failed for {url}: {message}"
            )

        payload = temp_path.read_bytes()

        if not payload:
            raise SovereignUniverseExpansionDownloadError(
                f"curl fallback returned an empty response for {url}."
            )

        return payload.decode(
            "utf-8",
            errors="replace",
        )
    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass


def _download_text(
    url: str,
    *,
    retries: int = 4,
    base_delay_seconds: float = 0.8,
) -> str:
    """
    Download one source using verified TLS with bounded retries.

    Transport order:
    1. urllib using a verified certifi/default SSL context.
    2. Windows/system curl using the operating-system certificate store.

    This handles both Borsa's intermittent connection resets and Windows
    environments where Python's OpenSSL trust chain differs from Schannel.
    """
    last_error: Exception | None = None
    ssl_context = _verified_ssl_context()

    for attempt in range(
        1,
        retries
        + 1,
    ):
        request = urllib.request.Request(
            url,
            headers=_headers(),
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=HTTP_TIMEOUT_SECONDS,
                context=ssl_context,
            ) as response:
                return response.read().decode(
                    "utf-8",
                    errors="replace",
                )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionResetError,
            ConnectionError,
            ssl.SSLError,
            OSError,
        ) as error:
            last_error = error

            if attempt >= retries:
                break

            delay = (
                base_delay_seconds
                * attempt
                + random.uniform(
                    0.10,
                    0.35,
                )
            )

            time.sleep(
                delay
            )

    try:
        print(
            "  transport | urllib failed; trying verified curl fallback"
        )

        return _curl_download_text(
            url
        )
    except SovereignUniverseExpansionDownloadError as curl_error:
        raise SovereignUniverseExpansionDownloadError(
            f"Could not download {url}. "
            f"urllib error: {last_error}. "
            f"curl error: {curl_error}"
        ) from curl_error


def _parse_decimal(value: str) -> float | None:
    text = value.replace("\xa0", " ").strip()
    if not text:
        return None
    match = re.search(r"[+-]?\d[\d.,]*", text)
    if match is None:
        return None
    numeric = match.group(0)
    if "," in numeric and "." in numeric:
        if numeric.rfind(",") > numeric.rfind("."):
            numeric = numeric.replace(".", "").replace(",", ".")
        else:
            numeric = numeric.replace(",", "")
    elif "," in numeric:
        groups = numeric.split(",")
        if len(groups) > 2 and all(len(group) == 3 for group in groups[1:]):
            numeric = numeric.replace(",", "")
        else:
            numeric = numeric.replace(",", ".")
    return float(numeric)


def _parse_date(value: str) -> date | None:
    text = " ".join(value.split())

    if not text:
        return None

    # DMO XML commonly uses ISO timestamps such as
    # 2026-09-07T00:00:00.
    try:
        return datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        ).date()
    except ValueError:
        pass

    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            pass

    return None


def parse_borsa_btp_page(html: str) -> tuple[ItalyBtpObservation, ...]:
    parser = _TableParser()
    parser.feed(html)
    observations: list[ItalyBtpObservation] = []

    for row in parser.rows:
        if len(row) < 5:
            continue
        joined = " ".join(row)
        isin_match = re.search(r"\bIT[A-Z0-9]{10}\b", joined.upper())
        if isin_match is None:
            continue
        maturity = _parse_date(row[4])
        if maturity is None:
            continue
        observations.append(
            ItalyBtpObservation(
                isin=isin_match.group(0),
                description=row[1].strip(),
                last_price=_parse_decimal(row[2]),
                coupon_display_percent=_parse_decimal(row[3]),
                maturity_date=maturity,
            )
        )

    if not observations:
        raise SovereignUniverseExpansionParseError(
            "No BTP rows were parsed from the Borsa Italiana page."
        )

    unique = {item.isin: item for item in observations}
    return tuple(sorted(unique.values(), key=lambda item: (item.maturity_date, item.isin)))




def _normalise_label(
    value: str,
) -> str:
    return (
        " ".join(
            value.split()
        )
        .strip()
        .casefold()
    )


def parse_borsa_btp_detail_page(
    html: str,
    *,
    isin: str,
) -> ItalyBtpObservation:
    """
    Parse one Borsa Italiana BTP detail page.

    The detail page is materially better than the list page for modelling:
    it exposes annual coupon, periodic coupon, coupon frequency, day-count
    basis, accrual-start date, reference price/date and gross yield.

    Borsa states that effective-yield data and calculations are supplied by
    Skipper Informatica Srl, so RepoLens keeps a separate market_source_name
    for those values rather than pretending the calculation is an executable
    Borsa quote.
    """
    parser = _TableParser()
    parser.feed(
        html
    )

    fields: dict[
        str,
        str,
    ] = {}

    for row in parser.rows:
        if len(
            row
        ) < 2:
            continue

        label = _normalise_label(
            row[
                0
            ]
        )

        value = " ".join(
            row[
                1:
            ]
        ).strip()

        if (
            label
            and value
            and label not in fields
        ):
            fields[
                label
            ] = value

    if not fields:
        raise SovereignUniverseExpansionParseError(
            f"No detail fields were parsed for {isin}."
        )

    parsed_isin = (
        fields.get(
            "codice isin"
        )
        or isin
    ).strip().upper()

    if parsed_isin != isin.strip().upper():
        raise SovereignUniverseExpansionParseError(
            f"Borsa detail-page ISIN mismatch: expected {isin}, "
            f"received {parsed_isin}."
        )

    description = (
        fields.get(
            "denominazione"
        )
        or isin
    )

    maturity = _parse_date(
        fields.get(
            "scadenza",
            "",
        )
    )

    if maturity is None:
        raise SovereignUniverseExpansionParseError(
            f"Borsa detail page for {isin} does not contain a valid maturity."
        )

    market_source = (
        "Borsa Italiana / Skipper Informatica"
    )

    return ItalyBtpObservation(
        isin=parsed_isin,
        description=description,
        coupon_display_percent=(
            _parse_decimal(
                fields.get(
                    "tasso cedola periodale",
                    "",
                )
            )
        ),
        last_price=None,
        maturity_date=maturity,
        annual_coupon_percent=(
            _parse_decimal(
                fields.get(
                    "tasso cedola su base annua",
                    "",
                )
            )
        ),
        periodic_coupon_percent=(
            _parse_decimal(
                fields.get(
                    "tasso cedola periodale",
                    "",
                )
            )
        ),
        coupon_frequency=(
            fields.get(
                "periodicità cedola"
            )
        ),
        accrual_start_date=(
            _parse_date(
                fields.get(
                    "data godimento",
                    "",
                )
            )
        ),
        first_coupon_date=(
            _parse_date(
                fields.get(
                    "data stacco prima cedola",
                    "",
                )
            )
        ),
        day_count_convention=(
            fields.get(
                "base di calcolo"
            )
        ),
        reference_price=(
            _parse_decimal(
                fields.get(
                    "prezzo di riferimento",
                    "",
                )
            )
        ),
        gross_yield_percent=(
            _parse_decimal(
                fields.get(
                    "rendimento effettivo a scadenza lordo",
                    "",
                )
            )
        ),
        reference_date=(
            _parse_date(
                fields.get(
                    "data di riferimento",
                    "",
                )
            )
        ),
        market_source_name=market_source,
    )


def enrich_nominal_btp_details(
    observations: tuple[
        ItalyBtpObservation,
        ...,
    ],
    *,
    request_delay_seconds: float = 0.35,
) -> tuple[
    ItalyBtpObservation,
    ...,
]:
    """
    Enrich nominal BTP list rows with exact detail-page terms and yields.

    A failed detail page does not destroy the already-collected reference
    universe. RepoLens keeps the base row and leaves unsupported fields blank.
    """
    enriched: list[
        ItalyBtpObservation
    ] = []

    nominal_total = sum(
        1
        for item in observations
        if item.is_nominal_btp
    )

    nominal_done = 0

    for item in observations:
        if not item.is_nominal_btp:
            enriched.append(
                item
            )
            continue

        nominal_done += 1

        detail_url = BORSA_BTP_DETAIL_URL.format(
            isin=item.isin
        )

        try:
            detail_html = _download_text(
                detail_url,
                retries=3,
                base_delay_seconds=0.6,
            )

            detail = parse_borsa_btp_detail_page(
                detail_html,
                isin=item.isin,
            )
        except (
            SovereignUniverseExpansionDownloadError,
            SovereignUniverseExpansionParseError,
        ) as error:
            print(
                f"  IT | detail {nominal_done}/{nominal_total} "
                f"{item.isin}: unavailable ({error}); keeping list-page row."
            )

            enriched.append(
                item
            )

        else:
            merged = replace(
                item,
                description=(
                    detail.description
                    or item.description
                ),
                annual_coupon_percent=(
                    detail.annual_coupon_percent
                ),
                periodic_coupon_percent=(
                    detail.periodic_coupon_percent
                ),
                coupon_frequency=(
                    detail.coupon_frequency
                ),
                accrual_start_date=(
                    detail.accrual_start_date
                ),
                first_coupon_date=(
                    detail.first_coupon_date
                ),
                day_count_convention=(
                    detail.day_count_convention
                ),
                reference_price=(
                    detail.reference_price
                ),
                gross_yield_percent=(
                    detail.gross_yield_percent
                ),
                reference_date=(
                    detail.reference_date
                ),
                market_source_name=(
                    detail.market_source_name
                ),
            )

            enriched.append(
                merged
            )

            print(
                f"  IT | detail {nominal_done}/{nominal_total} "
                f"{item.isin}: "
                f"{'yield' if merged.has_exact_reference_yield else 'no yield'}"
            )

        if request_delay_seconds > 0.0:
            time.sleep(
                request_delay_seconds
            )

    return tuple(
        enriched
    )


def download_full_btp_universe(max_pages: int = 25) -> tuple[ItalyBtpObservation, ...]:
    all_rows: dict[str, ItalyBtpObservation] = {}
    previous_page_isins: set[str] | None = None

    consecutive_failed_pages = 0

    for page in range(
        1,
        max_pages
        + 1,
    ):
        page_url = BORSA_BTP_LIST_URL.format(
            page=page
        )

        try:
            html = _download_text(
                page_url
            )
        except SovereignUniverseExpansionDownloadError as error:
            if page == 1 and not all_rows:
                raise

            consecutive_failed_pages += 1

            print(
                f"  IT | page {page} unavailable after retries; "
                f"keeping {len(all_rows)} instruments collected so far."
            )

            if consecutive_failed_pages >= 2:
                print(
                    "  IT | stopping pagination after two consecutive "
                    "failed pages to avoid hammering the source."
                )
                break

            time.sleep(
                1.5
            )
            continue

        consecutive_failed_pages = 0

        try:
            rows = parse_borsa_btp_page(
                html
            )
        except SovereignUniverseExpansionParseError:
            if page == 1 and not all_rows:
                raise

            break

        page_isins = {
            row.isin
            for row in rows
        }

        if (
            previous_page_isins is not None
            and page_isins == previous_page_isins
        ):
            break

        new_count = 0

        for row in rows:
            if row.isin not in all_rows:
                new_count += 1

            all_rows[
                row.isin
            ] = row

        print(
            f"  IT | page {page}: "
            f"{len(rows)} rows, {new_count} new, "
            f"{len(all_rows)} cumulative"
        )

        if new_count == 0:
            break

        previous_page_isins = page_isins

        # Be polite to the public exchange endpoint rather than firing pages
        # back-to-back. This also materially reduces WinError 10054 resets.
        time.sleep(
            0.9
        )

    if not all_rows:
        raise SovereignUniverseExpansionParseError(
            "No BTP instruments were collected across the paginated list."
        )

    return tuple(sorted(all_rows.values(), key=lambda item: (item.maturity_date, item.isin)))


def _local_name(
    tag: str,
) -> str:
    return tag.split(
        "}"
    )[
        -1
    ].upper()


def _normalise_xml_key(
    key: str,
) -> str:
    return (
        _local_name(
            key
        )
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


def _row_mapping(
    element: ET.Element,
) -> dict[
    str,
    str,
]:
    """
    Flatten one DMO XML row.

    The DMO D1A feed exposes its gilt records using fields such as
    INSTRUMENT_TYPE, ISIN_CODE, INSTRUMENT_NAME, REDEMPTION_DATE,
    FIRST_ISSUE_DATE and CLOSE_OF_BUSINESS_DATE. Depending on the XML
    representation, values may be attributes or child elements, so RepoLens
    supports both forms.
    """
    result: dict[
        str,
        str,
    ] = {}

    for key, value in element.attrib.items():
        text = (
            value
            or ""
        ).strip()

        if text:
            result[
                _normalise_xml_key(
                    key
                )
            ] = text

    for child in element:
        key = _normalise_xml_key(
            child.tag
        )

        text = (
            child.text
            or ""
        ).strip()

        if text:
            result[
                key
            ] = text

        for attr_key, attr_value in child.attrib.items():
            attr_text = (
                attr_value
                or ""
            ).strip()

            if attr_text:
                result[
                    _normalise_xml_key(
                        attr_key
                    )
                ] = attr_text

    return result


def _first_present(
    row: dict[
        str,
        str,
    ],
    *keys: str,
) -> str | None:
    for key in keys:
        value = row.get(
            key
        )

        if (
            value is not None
            and value.strip()
        ):
            return value.strip()

    return None


def _coupon_from_gilt_name(
    name: str,
) -> float | None:
    """
    Parse decimal and common UK fractional coupon notation from gilt names.
    """
    text = name.strip()

    decimal_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        text,
    )

    if decimal_match is not None:
        return float(
            decimal_match.group(
                1
            )
        )

    fraction_values = {
        "⅛": 0.125,
        "¼": 0.25,
        "⅜": 0.375,
        "½": 0.5,
        "⅝": 0.625,
        "¾": 0.75,
        "⅞": 0.875,
    }

    for symbol, fraction in fraction_values.items():
        unicode_match = re.search(
            rf"(\d+)?\s*{re.escape(symbol)}\s*%",
            text,
        )

        if unicode_match is not None:
            whole = (
                float(
                    unicode_match.group(
                        1
                    )
                )
                if unicode_match.group(
                    1
                )
                else 0.0
            )

            return whole + fraction

    ascii_fraction_match = re.search(
        r"(?:(\d+)\s+)?(\d+)/(\d+)\s*%",
        text,
    )

    if ascii_fraction_match is not None:
        whole = (
            float(
                ascii_fraction_match.group(
                    1
                )
            )
            if ascii_fraction_match.group(
                1
            )
            else 0.0
        )

        numerator = float(
            ascii_fraction_match.group(
                2
            )
        )

        denominator = float(
            ascii_fraction_match.group(
                3
            )
        )

        if denominator:
            return (
                whole
                + numerator
                / denominator
            )

    return None


def parse_dmo_gilts_xml(
    xml_text: str,
) -> tuple[
    ConventionalGilt,
    ...,
]:
    try:
        root = ET.fromstring(
            xml_text
        )
    except ET.ParseError as error:
        raise SovereignUniverseExpansionParseError(
            f"Could not parse DMO gilt XML: {error}"
        ) from error

    gilts: dict[
        str,
        ConventionalGilt,
    ] = {}

    recognised_rows = 0

    for element in root.iter():
        element_name = _local_name(
            element.tag
        )

        row = _row_mapping(
            element
        )

        # The official D1A XML uses View_GILTS_IN_ISSUE records. Keep the
        # parser tolerant of future wrappers while requiring the actual DMO
        # field names before treating an element as a gilt row.
        if (
            element_name != "VIEW_GILTS_IN_ISSUE"
            and "ISIN_CODE" not in row
        ):
            continue

        isin = _first_present(
            row,
            "ISIN_CODE",
            "ISIN",
        )

        name = _first_present(
            row,
            "INSTRUMENT_NAME",
            "GILT_NAME",
            "SECURITY_NAME",
            "NAME",
        )

        maturity_text = _first_present(
            row,
            "REDEMPTION_DATE",
            "MATURITY_DATE",
        )

        if (
            isin is None
            or name is None
            or maturity_text is None
        ):
            continue

        isin = isin.upper().strip()

        if re.fullmatch(
            r"GB[A-Z0-9]{10}",
            isin,
        ) is None:
            continue

        recognised_rows += 1

        instrument_type = (
            _first_present(
                row,
                "INSTRUMENT_TYPE",
            )
            or ""
        ).upper()

        upper_name = name.upper()

        # D1A can contain conventional and index-linked names depending on
        # report evolution. The first-pass curve universe is conventional
        # gilts only.
        if (
            "INDEX" in upper_name
            or "INDEX-LINKED" in upper_name
            or "INDEX LINKED" in upper_name
            or "INDEX" in instrument_type
        ):
            continue

        maturity = _parse_date(
            maturity_text
        )

        if maturity is None:
            continue

        coupon_text = _first_present(
            row,
            "COUPON",
            "COUPON_RATE",
            "INTEREST_RATE",
        )

        coupon_percent = (
            _parse_decimal(
                coupon_text
            )
            if coupon_text is not None
            else _coupon_from_gilt_name(
                name
            )
        )

        first_issue_text = _first_present(
            row,
            "FIRST_ISSUE_DATE",
            "ISSUE_DATE",
        )

        first_issue_date = (
            _parse_date(
                first_issue_text
            )
            if first_issue_text is not None
            else None
        )

        outstanding_text = _first_present(
            row,
            "NOMINAL_AMOUNT_IN_ISSUE",
            "NOMINAL_AMOUNT_OUTSTANDING",
            "TOTAL_AMOUNT_IN_ISSUE",
            "AMOUNT_IN_ISSUE",
            "CURRENT_AMOUNT_IN_ISSUE",
            "NOMINAL_AMOUNT",
            "AMOUNT_OUTSTANDING",
        )

        nominal_amount = (
            _parse_decimal(
                outstanding_text
            )
            if outstanding_text is not None
            else None
        )

        gilts[
            isin
        ] = ConventionalGilt(
            isin=isin,
            name=name,
            coupon_percent=coupon_percent,
            first_issue_date=first_issue_date,
            maturity_date=maturity,
            nominal_amount_outstanding_gbp=nominal_amount,
        )

    if not gilts:
        raise SovereignUniverseExpansionParseError(
            "DMO XML downloaded successfully but no conventional gilts were "
            "parsed. "
            f"Recognised D1A-style rows before filtering: {recognised_rows}."
        )

    return tuple(
        sorted(
            gilts.values(),
            key=lambda item: (
                item.maturity_date,
                item.isin,
            ),
        )
    )


def download_conventional_gilts() -> tuple[ConventionalGilt, ...]:
    return parse_dmo_gilts_xml(_download_text(DMO_GILTS_XML_URL))


def write_italy_outputs(
    observations: tuple[ItalyBtpObservation, ...],
    reference_path: Path = ITALY_OUTPUT_PATH,
    market_path: Path = ITALY_MARKET_OUTPUT_PATH,
) -> tuple[int, int]:
    nominal = tuple(
        item
        for item in observations
        if item.is_nominal_btp
    )

    reference_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with reference_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country_code",
                "country_name",
                "issuer",
                "instrument_type",
                "isin",
                "description",
                "annual_coupon_percent",
                "periodic_coupon_percent",
                "coupon_frequency",
                "accrual_start_date",
                "first_coupon_date",
                "maturity_date",
                "day_count_convention",
                "currency",
                "source_name",
                "data_status",
            ],
        )

        writer.writeheader()

        for item in nominal:
            writer.writerow(
                {
                    "country_code": "IT",
                    "country_name": "Italy",
                    "issuer": "Republic of Italy",
                    "instrument_type": "BTP",
                    "isin": item.isin,
                    "description": item.description,
                    "annual_coupon_percent": (
                        ""
                        if item.annual_coupon_percent is None
                        else f"{item.annual_coupon_percent:.6f}"
                    ),
                    "periodic_coupon_percent": (
                        ""
                        if item.periodic_coupon_percent is None
                        else f"{item.periodic_coupon_percent:.6f}"
                    ),
                    "coupon_frequency": (
                        item.coupon_frequency
                        or ""
                    ),
                    "accrual_start_date": (
                        ""
                        if item.accrual_start_date is None
                        else item.accrual_start_date.isoformat()
                    ),
                    "first_coupon_date": (
                        ""
                        if item.first_coupon_date is None
                        else item.first_coupon_date.isoformat()
                    ),
                    "maturity_date": item.maturity_date.isoformat(),
                    "day_count_convention": (
                        item.day_count_convention
                        or ""
                    ),
                    "currency": "EUR",
                    "source_name": item.source_name,
                    "data_status": item.data_status,
                }
            )

    with market_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country_code",
                "isin",
                "description",
                "maturity_date",
                "last_price",
                "reference_price",
                "yield_percent",
                "observation_date",
                "source_name",
                "data_status",
            ],
        )

        writer.writeheader()

        for item in nominal:
            writer.writerow(
                {
                    "country_code": "IT",
                    "isin": item.isin,
                    "description": item.description,
                    "maturity_date": item.maturity_date.isoformat(),
                    "last_price": (
                        ""
                        if item.last_price is None
                        else f"{item.last_price:.6f}"
                    ),
                    "reference_price": (
                        ""
                        if item.reference_price is None
                        else f"{item.reference_price:.6f}"
                    ),
                    "yield_percent": (
                        ""
                        if item.gross_yield_percent is None
                        else f"{item.gross_yield_percent:.6f}"
                    ),
                    "observation_date": (
                        ""
                        if item.reference_date is None
                        else item.reference_date.isoformat()
                    ),
                    "source_name": (
                        item.market_source_name
                        or item.source_name
                    ),
                    "data_status": item.data_status,
                }
            )

    return (
        len(
            nominal
        ),
        len(
            observations
        ),
    )


def write_uk_output(
    gilts: tuple[ConventionalGilt, ...],
    output_path: Path = UK_OUTPUT_PATH,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country_code",
                "country_name",
                "issuer",
                "instrument_type",
                "isin",
                "name",
                "coupon_percent",
                "first_issue_date",
                "maturity_date",
                "nominal_amount_outstanding_gbp",
                "currency",
                "source_name",
                "data_status",
            ],
        )
        writer.writeheader()
        for gilt in gilts:
            writer.writerow(
                {
                    "country_code": "GB",
                    "country_name": "United Kingdom",
                    "issuer": "United Kingdom",
                    "instrument_type": "GILT",
                    "isin": gilt.isin,
                    "name": gilt.name,
                    "coupon_percent": (
                        "" if gilt.coupon_percent is None else f"{gilt.coupon_percent:.6f}"
                    ),
                    "first_issue_date": (
                        "" if gilt.first_issue_date is None else gilt.first_issue_date.isoformat()
                    ),
                    "maturity_date": gilt.maturity_date.isoformat(),
                    "nominal_amount_outstanding_gbp": (
                        ""
                        if gilt.nominal_amount_outstanding_gbp is None
                        else f"{gilt.nominal_amount_outstanding_gbp:.2f}"
                    ),
                    "currency": "GBP",
                    "source_name": gilt.source_name,
                    "data_status": gilt.data_status,
                }
            )

    return len(gilts)


def main() -> None:
    btp_rows = download_full_btp_universe()

    btp_rows = enrich_nominal_btp_details(
        btp_rows
    )

    nominal_count, total_borsa_count = write_italy_outputs(
        btp_rows
    )

    exact_yield_count = sum(
        1
        for item in btp_rows
        if (
            item.is_nominal_btp
            and item.has_exact_reference_yield
        )
    )

    gilts = download_conventional_gilts()
    gilt_count = write_uk_output(gilts)

    print("Expanded sovereign universe successfully.")
    print(
        f"  IT | {nominal_count} nominal BTPs | "
        f"{total_borsa_count} total Borsa BTP-category rows scanned"
    )
    print(
        f"  IT | {exact_yield_count} nominal BTPs with exact reference yields"
    )
    print(
        f"  GB | {gilt_count} conventional gilts | official UK DMO XML"
    )
    print(
        f"  IT | {total_borsa_count - nominal_count} non-nominal/strip/"
        "inflation-linked rows excluded from first-pass nominal curve work"
    )


if __name__ == "__main__":
    main()