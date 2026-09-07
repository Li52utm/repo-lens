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


DEFAULT_SPREAD_OUTPUT_PATH: Final[Path] = Path(
    "data/market/sovereign_spreads.csv"
)

TELEBORSA_SPREAD_URL: Final[str] = (
    "https://borsaitaliana.teleborsa.it/pages/spread/overview.aspx"
)

SOURCE_NAME: Final[str] = "Teleborsa / Borsa Italiana spread overview"
DATA_STATUS: Final[str] = "PUBLIC_REFERENCE"
HTTP_TIMEOUT_SECONDS: Final[int] = 20

SPREAD_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("BTP-BUND", "BTP", "BUND"),
    ("BONO-BUND", "BONO", "BUND"),
    ("OAT-BUND", "OAT", "BUND"),
)


class SovereignSpreadError(RuntimeError):
    """Base exception for sovereign-spread ingestion."""


class SovereignSpreadDownloadError(SovereignSpreadError):
    """Raised when the public spread page cannot be downloaded."""


class SovereignSpreadParseError(SovereignSpreadError):
    """Raised when the public spread page cannot be parsed safely."""


@dataclass(frozen=True)
class SovereignSpreadSnapshot:
    spread_name: str
    observation_date: date
    observation_time: str
    spread_bp: float
    change_percent: float
    left_yield_percent: float
    right_yield_percent: float
    left_label: str
    right_label: str
    source_name: str = SOURCE_NAME
    data_status: str = DATA_STATUS

    @property
    def implied_previous_spread_bp(self) -> float | None:
        denominator = 1.0 + self.change_percent / 100.0

        if denominator <= 0.0:
            return None

        return self.spread_bp / denominator

    @property
    def implied_change_bp(self) -> float | None:
        previous = self.implied_previous_spread_bp

        if previous is None:
            return None

        return self.spread_bp - previous


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


def _normalise_decimal(
    value: str,
) -> float:
    cleaned = (
        value
        .strip()
        .replace(
            ".",
            "",
        )
        .replace(
            ",",
            ".",
        )
        .replace(
            "+",
            "",
        )
    )

    return float(
        cleaned
    )


def _spread_heading_pattern(
    spread_name: str,
) -> re.Pattern[str]:
    """
    Build a safe heading matcher for names such as BTP-BUND.

    The public page can render the separator as either "VS" or "-".
    Each leg is escaped independently so the separator remains regex syntax.
    """
    parts = [
        part.strip()
        for part in spread_name.split(
            "-"
        )
        if part.strip()
    ]

    if len(
        parts
    ) != 2:
        raise SovereignSpreadParseError(
            f"Unsupported spread name: {spread_name}."
        )

    left, right = parts

    return re.compile(
        (
            rf"Spread\s+{re.escape(left)}"
            rf"\s*(?:vs|-)\s*"
            rf"{re.escape(right)}"
        ),
        flags=re.IGNORECASE,
    )


def _extract_block(
    text: str,
    *,
    spread_name: str,
) -> str:
    pattern = _spread_heading_pattern(
        spread_name
    )

    match = pattern.search(
        text
    )

    if match is None:
        raise SovereignSpreadParseError(
            f"Could not find spread block for {spread_name}."
        )

    next_positions: list[int] = []

    for other_name, _, _ in SPREAD_SPECS:
        if other_name == spread_name:
            continue

        other_pattern = _spread_heading_pattern(
            other_name
        )

        other_match = other_pattern.search(
            text,
            pos=match.end(),
        )

        if other_match is not None:
            next_positions.append(
                other_match.start()
            )

    end = min(
        next_positions
    ) if next_positions else len(
        text
    )

    return text[
        match.start():end
    ]


def _search_number(
    block: str,
    pattern: str,
    *,
    field_name: str,
) -> float:
    match = re.search(
        pattern,
        block,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise SovereignSpreadParseError(
            f"Could not parse {field_name}."
        )

    return _normalise_decimal(
        match.group(
            1
        )
    )


def _search_timestamp(
    block: str,
) -> tuple[date, str]:
    match = re.search(
        r"(?:Aggiornamento|Ultimo\s+aggiornamento)\s*"
        r"(\d{2}/\d{2}/\d{4})"
        r"(?:\s+(\d{1,2}[.:]\d{2}))?",
        block,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise SovereignSpreadParseError(
            "Could not parse spread observation timestamp."
        )

    observation_date = datetime.strptime(
        match.group(
            1
        ),
        "%d/%m/%Y",
    ).date()

    observation_time = (
        match.group(
            2
        )
        or ""
    ).replace(
        ".",
        ":",
    )

    return (
        observation_date,
        observation_time,
    )


def parse_spread_snapshot(
    *,
    text: str,
    spread_name: str,
    left_label: str,
    right_label: str,
) -> SovereignSpreadSnapshot:
    block = _extract_block(
        text,
        spread_name=spread_name,
    )

    spread_bp = _search_number(
        block,
        r"Attuale\s*([+-]?\d+(?:[.,]\d+)?)\s*(?:Punti|punti)",
        field_name=f"{spread_name} current spread",
    )

    change_percent = _search_number(
        block,
        r"Variazione(?:%)?\s*([+-]?\d+(?:[.,]\d+)?)\s*%",
        field_name=f"{spread_name} percentage change",
    )

    left_yield = _search_number(
        block,
        rf"Rendimento\s+{re.escape(left_label)}\s+10\s+Anni"
        r"\s*([+-]?\d+(?:[.,]\d+)?)\s*%",
        field_name=f"{left_label} 10Y yield",
    )

    right_yield = _search_number(
        block,
        rf"Rendimento\s+{re.escape(right_label)}\s+10\s+Anni"
        r"\s*([+-]?\d+(?:[.,]\d+)?)\s*%",
        field_name=f"{right_label} 10Y yield",
    )

    observation_date, observation_time = _search_timestamp(
        block
    )

    return SovereignSpreadSnapshot(
        spread_name=spread_name,
        observation_date=observation_date,
        observation_time=observation_time,
        spread_bp=spread_bp,
        change_percent=change_percent,
        left_yield_percent=left_yield,
        right_yield_percent=right_yield,
        left_label=left_label,
        right_label=right_label,
    )


def parse_all_spreads(
    html: str,
) -> tuple[
    SovereignSpreadSnapshot,
    ...,
]:
    text = visible_text(
        html
    )

    snapshots = tuple(
        parse_spread_snapshot(
            text=text,
            spread_name=spread_name,
            left_label=left_label,
            right_label=right_label,
        )
        for (
            spread_name,
            left_label,
            right_label,
        ) in SPREAD_SPECS
    )

    return snapshots


def download_spread_page() -> str:
    request = urllib.request.Request(
        TELEBORSA_SPREAD_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 RepoLens/1.0"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
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
        raise SovereignSpreadDownloadError(
            "Could not download the public sovereign-spread page: "
            f"{error}"
        ) from error


def _existing_keys(
    output_path: Path,
) -> set[
    tuple[str, str, str, str]
]:
    if not output_path.exists():
        return set()

    keys: set[
        tuple[str, str, str, str]
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
                        "spread_name"
                    ],
                    row[
                        "observation_date"
                    ],
                    row.get(
                        "observation_time",
                        "",
                    ),
                    row[
                        "source_name"
                    ],
                )
            )

    return keys


def append_snapshots(
    *,
    snapshots: tuple[
        SovereignSpreadSnapshot,
        ...,
    ],
    output_path: Path = DEFAULT_SPREAD_OUTPUT_PATH,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = _existing_keys(
        output_path
    )

    fieldnames = [
        "spread_name",
        "observation_date",
        "observation_time",
        "spread_bp",
        "change_percent",
        "left_yield_percent",
        "right_yield_percent",
        "left_label",
        "right_label",
        "source_name",
        "data_status",
    ]

    should_write_header = (
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

        if should_write_header:
            writer.writeheader()

        for snapshot in snapshots:
            key = (
                snapshot.spread_name,
                snapshot.observation_date.isoformat(),
                snapshot.observation_time,
                snapshot.source_name,
            )

            if key in existing:
                continue

            writer.writerow(
                {
                    "spread_name": snapshot.spread_name,
                    "observation_date": snapshot.observation_date.isoformat(),
                    "observation_time": snapshot.observation_time,
                    "spread_bp": f"{snapshot.spread_bp:.6f}",
                    "change_percent": f"{snapshot.change_percent:.6f}",
                    "left_yield_percent": f"{snapshot.left_yield_percent:.6f}",
                    "right_yield_percent": f"{snapshot.right_yield_percent:.6f}",
                    "left_label": snapshot.left_label,
                    "right_label": snapshot.right_label,
                    "source_name": snapshot.source_name,
                    "data_status": snapshot.data_status,
                }
            )

            existing.add(
                key
            )
            appended += 1

    return appended


def ingest_public_sovereign_spreads(
    output_path: Path = DEFAULT_SPREAD_OUTPUT_PATH,
) -> tuple[
    tuple[SovereignSpreadSnapshot, ...],
    int,
]:
    html = download_spread_page()
    snapshots = parse_all_spreads(
        html
    )

    appended = append_snapshots(
        snapshots=snapshots,
        output_path=output_path,
    )

    return (
        snapshots,
        appended,
    )


def main() -> None:
    snapshots, appended = ingest_public_sovereign_spreads()

    print(
        f"Saved {appended} new sovereign spread observation(s)."
    )

    for snapshot in snapshots:
        implied_change = snapshot.implied_change_bp

        change_text = (
            f"{implied_change:+.2f} bp implied"
            if implied_change is not None
            else "N/A"
        )

        print(
            f"  {snapshot.spread_name} | "
            f"{snapshot.observation_date.isoformat()} "
            f"{snapshot.observation_time or '--:--'} | "
            f"{snapshot.spread_bp:.1f} bp | "
            f"{snapshot.left_yield_percent:.2f}% vs "
            f"{snapshot.right_yield_percent:.2f}% | "
            f"{snapshot.change_percent:+.2f}% ({change_text})"
        )


if __name__ == "__main__":
    main()