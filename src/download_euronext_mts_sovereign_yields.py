from __future__ import annotations

import csv
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Final


DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "data/market/european_sovereign_benchmark_yields.csv"
)

EURONEXT_MTS_URL: Final[str] = (
    "https://live.euronext.com/en/products/indices/euronext-mts"
)

SOURCE_NAME: Final[str] = "Euronext MTS Indices"
DATA_STATUS: Final[str] = "PUBLIC_REFERENCE"
HTTP_TIMEOUT_SECONDS: Final[int] = 20


class EuronextMtsYieldError(RuntimeError):
    """Base exception for Euronext MTS sovereign-yield ingestion."""


class EuronextMtsYieldDownloadError(EuronextMtsYieldError):
    """Raised when the public Euronext MTS page cannot be downloaded."""


class EuronextMtsYieldParseError(EuronextMtsYieldError):
    """Raised when the public Euronext MTS page cannot be parsed safely."""


@dataclass(frozen=True)
class SovereignBenchmarkYieldObservation:
    country_code: str
    country_name: str
    curve_name: str
    tenor_years: int
    observation_date: date
    yield_percent: float
    index_name: str
    index_isin: str
    symbol: str
    source_name: str = SOURCE_NAME
    data_status: str = DATA_STATUS


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


def _normalise_number(
    value: str,
) -> float:
    cleaned = (
        value
        .strip()
        .replace(
            ",",
            "",
        )
    )

    return float(
        cleaned
    )


def _parse_date(
    value: str,
) -> date:
    return datetime.strptime(
        value.strip(),
        "%d %b %Y",
    ).date()


def _country_mapping(
    index_name: str,
) -> tuple[
    str,
    str,
    str,
] | None:
    upper = index_name.upper()

    mappings = (
        ("BTP", "IT", "Italy", "BTP"),
        ("BUND", "DE", "Germany", "Bund"),
        ("OAT", "FR", "France", "OAT"),
        ("SPGB", "ES", "Spain", "SPGB"),
        ("OLO", "BE", "Belgium", "OLO"),
        ("RAGB", "AT", "Austria", "RAGB"),
        ("DSL", "NL", "Netherlands", "DSL"),
        ("NXG", "EU", "European Union", "NGEU"),
    )

    for token, code, country, curve in mappings:
        if (
            f"MTS {token} " in upper
            or upper.startswith(
                f"EURONEXT MTS {token} "
            )
        ):
            return (
                code,
                country,
                curve,
            )

    if upper.startswith(
        "TEC"
    ) or " MTS TEC" in upper:
        return (
            "FR",
            "France",
            "TEC",
        )

    return None


def _extract_tenor_years(
    index_name: str,
) -> int | None:
    match = re.search(
        r"(\d{1,2})Y\s+YIELD",
        index_name,
        flags=re.IGNORECASE,
    )

    if match is not None:
        return int(
            match.group(
                1
            )
        )

    tec_match = re.fullmatch(
        r"TEC(\d{1,2})",
        index_name.strip(),
        flags=re.IGNORECASE,
    )

    if tec_match is not None:
        return int(
            tec_match.group(
                1
            )
        )

    return None


def parse_euronext_mts_yields(
    html: str,
) -> tuple[
    SovereignBenchmarkYieldObservation,
    ...,
]:
    text = visible_text(
        html
    )

    row_pattern = re.compile(
        (
            r"(?P<name>"
            r"(?:Euronext\s+MTS\s+(?:BTP|BUND|DSL|OAT|OLO|RAGB|SPGB|NXG)"
            r"\s+\d{1,2}Y\s+Yield)"
            r"|(?:TEC(?:1|2|3|5|7|10|15|20|25|30))"
            r")"
            r"\s+"
            r"(?P<isin>[A-Z]{2}[A-Z0-9]{10})"
            r"\s+"
            r"(?P<symbol>[A-Z0-9]+)"
            r"\s+"
            r"EUR\s+"
            r"(?P<yield>[+-]?\d+(?:[.,]\d+)?)"
            r"\s+"
            r"[+-]?\d+(?:[.,]\d+)?%"
            r"\s+"
            r"(?P<date>\d{2}\s+[A-Z][a-z]{2}\s+\d{4})"
        ),
        flags=re.IGNORECASE,
    )

    observations: list[
        SovereignBenchmarkYieldObservation
    ] = []

    seen: set[
        tuple[str, int, date]
    ] = set()

    for match in row_pattern.finditer(
        text
    ):
        index_name = " ".join(
            match.group(
                "name"
            ).split()
        )

        if "11AM" in index_name.upper():
            continue

        mapping = _country_mapping(
            index_name
        )

        if mapping is None:
            continue

        tenor_years = _extract_tenor_years(
            index_name
        )

        if tenor_years is None:
            continue

        country_code, country_name, curve_name = mapping

        observation_date = _parse_date(
            match.group(
                "date"
            )
        )

        key = (
            country_code,
            tenor_years,
            observation_date,
        )

        if key in seen:
            # Prefer the first canonical country curve found.
            # This avoids silently duplicating France OAT 10Y with TEC10.
            continue

        seen.add(
            key
        )

        observations.append(
            SovereignBenchmarkYieldObservation(
                country_code=country_code,
                country_name=country_name,
                curve_name=curve_name,
                tenor_years=tenor_years,
                observation_date=observation_date,
                yield_percent=_normalise_number(
                    match.group(
                        "yield"
                    )
                ),
                index_name=index_name,
                index_isin=match.group(
                    "isin"
                ).upper(),
                symbol=match.group(
                    "symbol"
                ).upper(),
            )
        )

    if not observations:
        raise EuronextMtsYieldParseError(
            "No recognised Euronext MTS sovereign-yield observations were parsed."
        )

    return tuple(
        sorted(
            observations,
            key=lambda item: (
                item.country_name,
                item.tenor_years,
                item.index_name,
            ),
        )
    )


def download_euronext_mts_page() -> str:
    request = urllib.request.Request(
        EURONEXT_MTS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 RepoLens/1.0"
            ),
            "Accept-Language": "en-GB,en;q=0.9",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            return response.read().decode(
                "utf-8",
                errors="replace",
            )
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        OSError,
    ) as error:
        raise EuronextMtsYieldDownloadError(
            "Could not download the Euronext MTS sovereign-yield page: "
            f"{error}"
        ) from error


def _existing_keys(
    output_path: Path,
) -> set[
    tuple[str, str, str]
]:
    if not output_path.exists():
        return set()

    keys: set[
        tuple[str, str, str]
    ] = set()

    with output_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            keys.add(
                (
                    row[
                        "country_code"
                    ],
                    row[
                        "tenor_years"
                    ],
                    row[
                        "observation_date"
                    ],
                )
            )

    return keys


def append_observations(
    *,
    observations: tuple[
        SovereignBenchmarkYieldObservation,
        ...,
    ],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = _existing_keys(
        output_path
    )

    fieldnames = [
        "country_code",
        "country_name",
        "curve_name",
        "tenor_years",
        "observation_date",
        "yield_percent",
        "index_name",
        "index_isin",
        "symbol",
        "source_name",
        "data_status",
    ]

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    appended = 0

    with output_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        for observation in observations:
            key = (
                observation.country_code,
                str(
                    observation.tenor_years
                ),
                observation.observation_date.isoformat(),
            )

            if key in existing:
                continue

            writer.writerow(
                {
                    "country_code": observation.country_code,
                    "country_name": observation.country_name,
                    "curve_name": observation.curve_name,
                    "tenor_years": observation.tenor_years,
                    "observation_date": observation.observation_date.isoformat(),
                    "yield_percent": f"{observation.yield_percent:.6f}",
                    "index_name": observation.index_name,
                    "index_isin": observation.index_isin,
                    "symbol": observation.symbol,
                    "source_name": observation.source_name,
                    "data_status": observation.data_status,
                }
            )

            existing.add(
                key
            )
            appended += 1

    return appended


def ingest_european_sovereign_benchmark_yields(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[
    tuple[SovereignBenchmarkYieldObservation, ...],
    int,
]:
    html = download_euronext_mts_page()

    observations = parse_euronext_mts_yields(
        html
    )

    appended = append_observations(
        observations=observations,
        output_path=output_path,
    )

    return (
        observations,
        appended,
    )


def main() -> None:
    observations, appended = (
        ingest_european_sovereign_benchmark_yields()
    )

    print(
        f"Saved {appended} new European sovereign benchmark observation(s)."
    )

    for observation in observations:
        print(
            f"  {observation.country_code} | "
            f"{observation.tenor_years:>2}Y | "
            f"{observation.yield_percent:.3f}% | "
            f"{observation.observation_date.isoformat()} | "
            f"{observation.curve_name} | "
            f"{observation.symbol}"
        )


if __name__ == "__main__":
    main()