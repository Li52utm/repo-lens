from __future__ import annotations

from datetime import date

import pytest

from src.download_btp_public_snapshot import (
    BtpPublicSnapshotParseError,
    PUBLIC_DATA_STATUS,
    PUBLIC_SOURCE_NAME,
    html_to_visible_text,
    parse_btp_public_snapshot,
)


ISIN = "IT0005706285"


def sample_html(
    *,
    isin: str = ISIN,
    gross_yield: str = "4,14",
    reference_price: str = "97,55",
    reference_date: str = "04/09/2026",
) -> str:
    return f"""
    <!doctype html>
    <html>
        <head>
            <style>.hidden {{ display: none; }}</style>
            <script>const fake = "Prezzo di riferimento 999";</script>
        </head>
        <body>
            <h3>Rendimenti Effettivi</h3>
            <div>Rendimento effettivo a scadenza lordo</div>
            <div>{gross_yield}</div>
            <div>Prezzo di riferimento</div>
            <div>{reference_price}</div>
            <div>Data di riferimento</div>
            <div>{reference_date}</div>
            <div>Codice Isin</div>
            <div>{isin}</div>
        </body>
    </html>
    """


def test_visible_text_ignores_script_and_style_content() -> None:
    text = html_to_visible_text(sample_html())
    assert "const fake" not in text
    assert ".hidden" not in text
    assert "Rendimento effettivo a scadenza lordo" in text


def test_parses_reference_price_yield_and_date() -> None:
    snapshot = parse_btp_public_snapshot(isin=ISIN, html=sample_html())

    assert snapshot.isin == ISIN
    assert snapshot.observation_date == date(2026, 9, 4)
    assert snapshot.reference_price_per_100 == pytest.approx(97.55)
    assert snapshot.gross_yield_percent == pytest.approx(4.14)
    assert snapshot.source_name == PUBLIC_SOURCE_NAME
    assert snapshot.data_status == PUBLIC_DATA_STATUS


def test_two_digit_reference_year_is_supported() -> None:
    snapshot = parse_btp_public_snapshot(
        isin=ISIN,
        html=sample_html(reference_date="04/09/26"),
    )
    assert snapshot.observation_date == date(2026, 9, 4)


def test_thousands_separator_and_comma_decimal_are_supported() -> None:
    snapshot = parse_btp_public_snapshot(
        isin=ISIN,
        html=sample_html(reference_price="1.097,55"),
    )
    assert snapshot.reference_price_per_100 == pytest.approx(1097.55)


def test_requested_isin_must_match_page_isin() -> None:
    with pytest.raises(BtpPublicSnapshotParseError):
        parse_btp_public_snapshot(
            isin=ISIN,
            html=sample_html(isin="IT0005692410"),
        )


def test_missing_yield_is_rejected() -> None:
    html = sample_html().replace(
        "Rendimento effettivo a scadenza lordo",
        "Gross yield missing",
    )
    with pytest.raises(BtpPublicSnapshotParseError):
        parse_btp_public_snapshot(isin=ISIN, html=html)


def test_missing_reference_price_is_rejected() -> None:
    html = sample_html().replace(
        "Prezzo di riferimento",
        "Reference price missing",
    )
    with pytest.raises(BtpPublicSnapshotParseError):
        parse_btp_public_snapshot(isin=ISIN, html=html)


def test_missing_reference_date_is_rejected() -> None:
    html = sample_html().replace(
        "Data di riferimento",
        "Reference date missing",
    )
    with pytest.raises(BtpPublicSnapshotParseError):
        parse_btp_public_snapshot(isin=ISIN, html=html)


def test_non_italian_requested_isin_is_rejected() -> None:
    with pytest.raises(BtpPublicSnapshotParseError):
        parse_btp_public_snapshot(
            isin="FR0014018YR0",
            html=sample_html(),
        )


def test_snapshot_converts_to_historical_observation() -> None:
    observation = parse_btp_public_snapshot(
        isin=ISIN,
        html=sample_html(),
    ).to_historical_observation()

    assert observation.isin == ISIN
    assert observation.observation_date == date(2026, 9, 4)
    assert observation.price_per_100 == pytest.approx(97.55)
    assert observation.yield_percent == pytest.approx(4.14)
    assert observation.benchmark_spread_bp is None
    assert observation.benchmark_name is None