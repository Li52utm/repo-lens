from __future__ import annotations

from pathlib import Path

from src.sovereign_desk_universe import (
    DeskCountry,
    load_desk_sovereign_universe,
    universe_counts_by_country,
)


def test_dynamic_reference_files_override_and_expand_catalogue(
    tmp_path: Path,
) -> None:
    italy = tmp_path / "italy.csv"
    italy.write_text(
        "\n".join(
            [
                (
                    "country_code,country_name,issuer,instrument_type,isin,"
                    "description,coupon_display_percent,maturity_date,currency,"
                    "source_name,data_status"
                ),
                (
                    "IT,Italy,Republic of Italy,BTP,IT0000000001,"
                    "Btp Test One,3.250000,2031-06-01,EUR,"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
                (
                    "IT,Italy,Republic of Italy,BTP,IT0000000002,"
                    "Btp Test Two,4.000000,2037-02-01,EUR,"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    uk = tmp_path / "uk.csv"
    uk.write_text(
        "\n".join(
            [
                (
                    "country_code,country_name,issuer,instrument_type,isin,"
                    "name,coupon_percent,first_issue_date,maturity_date,"
                    "nominal_amount_outstanding_gbp,currency,source_name,data_status"
                ),
                (
                    "GB,United Kingdom,United Kingdom,GILT,GB0000000001,"
                    "4.125% Treasury Gilt 2031,4.125000,2025-01-01,2031-01-29,"
                    "35000.00,GBP,UK Debt Management Office,OFFICIAL_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    instruments = load_desk_sovereign_universe(
        paths=(
            italy,
            uk,
        )
    )

    italy_rows = [
        instrument
        for instrument in instruments
        if instrument.country is DeskCountry.ITALY
        and instrument.isin.startswith("IT000000")
    ]

    uk_rows = [
        instrument
        for instrument in instruments
        if instrument.country is DeskCountry.UNITED_KINGDOM
    ]

    assert len(
        italy_rows
    ) == 2

    assert len(
        uk_rows
    ) == 1

    assert uk_rows[
        0
    ].currency == "GBP"

    assert uk_rows[
        0
    ].source_name == "UK Debt Management Office"

    assert uk_rows[
        0
    ].benchmark_tenor_years in {
        2,
        5,
        7,
        10,
    }


def test_counts_are_desk_friendly(
    tmp_path: Path,
) -> None:
    italy = tmp_path / "italy.csv"
    italy.write_text(
        "\n".join(
            [
                (
                    "country_code,country_name,issuer,instrument_type,isin,"
                    "description,coupon_display_percent,maturity_date,currency,"
                    "source_name,data_status"
                ),
                (
                    "IT,Italy,Republic of Italy,BTP,IT0000000001,"
                    "Btp One,3.250000,2031-06-01,EUR,"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
                (
                    "IT,Italy,Republic of Italy,BTP,IT0000000002,"
                    "Btp Two,4.000000,2037-02-01,EUR,"
                    "Borsa Italiana MOT,PUBLIC_REFERENCE"
                ),
            ]
        ),
        encoding="utf-8",
    )

    counts = universe_counts_by_country(
        load_desk_sovereign_universe(
            paths=(
                italy,
            )
        )
    )

    assert counts[
        "Italy"
    ] >= 2
