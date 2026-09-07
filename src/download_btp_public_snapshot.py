from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from re import sub
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.sovereign_historical_context import SovereignHistoricalObservation
from src.sovereign_history_store import (
    SovereignHistoryStore,
    SovereignHistoryStoreValidationError,
)
from src.sovereign_instruments import (
    SOVEREIGN_INSTRUMENTS,
    SovereignCountry,
    SovereignInstrument,
)

DEFAULT_OUTPUT_PATH = Path("data/market/sovereign_history.csv")
BORSA_BTP_URL_TEMPLATE = (
    "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/"
    "scheda/{isin}-MOTX.html?lang=it"
)
HTTP_TIMEOUT_SECONDS = 20
PUBLIC_SOURCE_NAME = "Borsa Italiana MOT / Skipper Informatica"
PUBLIC_DATA_STATUS = "DELAYED_PUBLIC_REFERENCE"


class BtpPublicSnapshotError(Exception):
    """Base exception for public BTP snapshot ingestion."""


class BtpPublicSnapshotParseError(BtpPublicSnapshotError):
    """Raised when required public BTP fields cannot be parsed."""


class BtpPublicSnapshotDownloadError(BtpPublicSnapshotError):
    """Raised when a public BTP page cannot be downloaded."""


@dataclass(frozen=True)
class BtpPublicSnapshot:
    isin: str
    observation_date: date
    reference_price_per_100: float
    gross_yield_percent: float
    source_name: str = PUBLIC_SOURCE_NAME
    data_status: str = PUBLIC_DATA_STATUS

    def to_historical_observation(self) -> SovereignHistoricalObservation:
        return SovereignHistoricalObservation(
            isin=self.isin,
            observation_date=self.observation_date,
            source_name=self.source_name,
            data_status=self.data_status,
            price_per_100=self.reference_price_per_100,
            yield_percent=self.gross_yield_percent,
            benchmark_spread_bp=None,
            benchmark_name=None,
        )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return sub(r"\s+", " ", unescape(" ".join(self._parts))).strip()


def html_to_visible_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return parser.text()


def _italian_decimal(value: str) -> float:
    clean = value.strip().replace(".", "").replace(",", ".")
    try:
        return float(clean)
    except ValueError as error:
        raise BtpPublicSnapshotParseError(
            f"Could not parse Italian decimal value: {value!r}"
        ) from error


def _parse_reference_date(value: str) -> date:
    pieces = value.strip().split("/")
    if len(pieces) != 3:
        raise BtpPublicSnapshotParseError(
            f"Could not parse BTP reference date: {value!r}"
        )

    try:
        day, month, year = (int(piece) for piece in pieces)
        if year < 100:
            year += 2000
        return date(year, month, day)
    except ValueError as error:
        raise BtpPublicSnapshotParseError(
            f"Could not parse BTP reference date: {value!r}"
        ) from error


def _extract_value_after_label(text: str, label: str) -> str:
    marker = f"{label} "
    start = text.find(marker)
    if start < 0:
        raise BtpPublicSnapshotParseError(
            f"Could not find required BTP field: {label}"
        )

    remainder = text[start + len(marker):].lstrip()
    if not remainder:
        raise BtpPublicSnapshotParseError(f"BTP field has no value: {label}")

    return remainder.split(" ", 1)[0]


def parse_btp_public_snapshot(*, isin: str, html: str) -> BtpPublicSnapshot:
    """
    Parse the public reference price/date and gross YTM from one BTP page.

    The parser deliberately uses the 'Prezzo di riferimento' and
    'Data di riferimento' fields from the same yield section as the gross YTM,
    rather than combining differently dated fields.
    """
    clean_isin = isin.strip().upper()
    if (
        len(clean_isin) != 12
        or not clean_isin.isalnum()
        or not clean_isin.startswith("IT")
    ):
        raise BtpPublicSnapshotParseError(
            "BTP public snapshot requires a valid Italian ISIN."
        )

    text = html_to_visible_text(html)
    page_isin = _extract_value_after_label(text, "Codice Isin").upper()
    if page_isin != clean_isin:
        raise BtpPublicSnapshotParseError(
            f"Downloaded page ISIN {page_isin} does not match {clean_isin}."
        )

    gross_yield = _italian_decimal(
        _extract_value_after_label(
            text,
            "Rendimento effettivo a scadenza lordo",
        )
    )
    reference_price = _italian_decimal(
        _extract_value_after_label(text, "Prezzo di riferimento")
    )
    reference_date = _parse_reference_date(
        _extract_value_after_label(text, "Data di riferimento")
    )

    if reference_price <= 0.0:
        raise BtpPublicSnapshotParseError(
            "BTP reference price must be positive."
        )

    return BtpPublicSnapshot(
        isin=clean_isin,
        observation_date=reference_date,
        reference_price_per_100=reference_price,
        gross_yield_percent=gross_yield,
    )


def download_btp_page(isin: str) -> str:
    clean_isin = isin.strip().upper()
    url = BORSA_BTP_URL_TEMPLATE.format(isin=clean_isin)
    request = Request(
        url,
        headers={
            "User-Agent": (
                "RepoLens/0.1 public-market-reference-ingestion "
                "(research analytics)"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        },
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise BtpPublicSnapshotDownloadError(
            f"Could not download Borsa Italiana BTP page for {clean_isin}: "
            f"{error}"
        ) from error


def fetch_btp_public_snapshot(isin: str) -> BtpPublicSnapshot:
    return parse_btp_public_snapshot(
        isin=isin,
        html=download_btp_page(isin),
    )


def catalogue_btp_instruments() -> tuple[SovereignInstrument, ...]:
    return tuple(
        instrument
        for instrument in SOVEREIGN_INSTRUMENTS
        if instrument.country == SovereignCountry.ITALY
    )


def ingest_btp_public_snapshots(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    instruments: Iterable[SovereignInstrument] | None = None,
) -> tuple[tuple[BtpPublicSnapshot, ...], tuple[tuple[str, str], ...]]:
    """
    Fetch and persist one current public reference observation per BTP ISIN.

    Existing same-date observations from this source are skipped. Failures are
    returned per ISIN so a single unavailable page does not block the universe.
    """
    selected = tuple(
        instruments if instruments is not None else catalogue_btp_instruments()
    )
    store = SovereignHistoryStore(output_path)
    existing = {
        (
            observation.isin,
            observation.observation_date,
            observation.source_name,
        )
        for observation in store.load()
    }

    saved: list[BtpPublicSnapshot] = []
    failures: list[tuple[str, str]] = []

    for instrument in selected:
        try:
            snapshot = fetch_btp_public_snapshot(instrument.isin)
            key = (
                snapshot.isin,
                snapshot.observation_date,
                snapshot.source_name,
            )
            if key in existing:
                continue

            store.append([snapshot.to_historical_observation()])
            existing.add(key)
            saved.append(snapshot)
        except (
            BtpPublicSnapshotError,
            SovereignHistoryStoreValidationError,
        ) as error:
            failures.append((instrument.isin, str(error)))

    return tuple(saved), tuple(failures)


def main() -> None:
    saved, failures = ingest_btp_public_snapshots()

    print(f"Saved {len(saved)} new BTP public reference observation(s).")
    for snapshot in saved:
        print(
            f"  {snapshot.isin} | {snapshot.observation_date.isoformat()} | "
            f"price {snapshot.reference_price_per_100:.4f} | "
            f"yield {snapshot.gross_yield_percent:.3f}%"
        )

    if failures:
        print(f"{len(failures)} BTP instrument(s) could not be refreshed:")
        for isin, message in failures:
            print(f"  {isin}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
