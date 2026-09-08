from __future__ import annotations

from datetime import date

from src.download_italy_btp_and_uk_gilt_universe import (
    parse_borsa_btp_detail_page,
    parse_borsa_btp_page,
    parse_dmo_gilts_xml,
)


BORSA_HTML = """
<table>
<tr><th>Isin</th><th>Descrizione</th><th>Ultimo</th><th>Cedola</th><th>Scadenza</th></tr>
<tr><td>IT0001086567</td><td>Btp-1nv26 7,25%</td><td>100,787</td><td>3,625</td><td>01/11/2026</td></tr>
<tr><td>IT0003745541</td><td>Btpi-15st35 2,35%</td><td>105,47</td><td>1,175</td><td>15/09/2035</td></tr>
<tr><td>IT0003268882</td><td>Btpstripital Zc Aug27 Eur</td><td>97,612</td><td></td><td>01/08/2027</td></tr>
<tr><td>IT0003934657</td><td>Btp-1fb37 4%</td><td>98,96</td><td>2,00</td><td>01/02/2037</td></tr>
</table>
"""


BORSA_DETAIL_HTML = """
<table>
<tr><td>Prezzo ufficiale</td><td>98,3658</td></tr>
<tr><td>Data Pr Ufficiale</td><td>04/09/26</td></tr>
<tr><td>Rendimento effettivo a scadenza lordo</td><td>3,54</td></tr>
<tr><td>Rendimento effettivo a scadenza netto</td><td>3,12</td></tr>
<tr><td>Prezzo di riferimento</td><td>98,44</td></tr>
<tr><td>Data di riferimento</td><td>04/09/2026</td></tr>
<tr><td>Codice Isin</td><td>IT0005707614</td></tr>
<tr><td>Denominazione</td><td>Btp Fx 3.15% Jun31 Eur</td></tr>
<tr><td>Data Godimento</td><td>04/05/26</td></tr>
<tr><td>Data Stacco prima Cedola</td><td>04/05/26</td></tr>
<tr><td>Scadenza</td><td>01/06/31</td></tr>
<tr><td>Periodicità cedola</td><td>Semestrale</td></tr>
<tr><td>Base di Calcolo</td><td>ACT/ACT (ICMA)</td></tr>
<tr><td>Tasso Cedola Periodale</td><td>1,575</td></tr>
<tr><td>Tasso Cedola su base Annua</td><td>3,15</td></tr>
</table>
"""


DMO_XML = """
<NewDataSet>
  <View_GILTS_IN_ISSUE
      INSTRUMENT_TYPE="Conventional"
      ISIN_CODE="GB00BL6C7720"
      INSTRUMENT_NAME="4.125% Treasury Gilt 2027"
      REDEMPTION_DATE="2027-01-29T00:00:00"
      FIRST_ISSUE_DATE="2022-10-18T00:00:00"
      CLOSE_OF_BUSINESS_DATE="2026-09-07T00:00:00"
      NOMINAL_AMOUNT_IN_ISSUE="35000.0" />
  <View_GILTS_IN_ISSUE
      INSTRUMENT_TYPE="Index-linked"
      ISIN_CODE="GB00BPSNBB36"
      INSTRUMENT_NAME="1.250% Index-linked Treasury Gilt 2054"
      REDEMPTION_DATE="2054-11-22T00:00:00"
      FIRST_ISSUE_DATE="2024-03-01T00:00:00"
      CLOSE_OF_BUSINESS_DATE="2026-09-07T00:00:00"
      NOMINAL_AMOUNT_IN_ISSUE="12000.0" />
  <View_GILTS_IN_ISSUE>
    <INSTRUMENT_TYPE>Conventional</INSTRUMENT_TYPE>
    <ISIN_CODE>GB00BPSNBF73</ISIN_CODE>
    <INSTRUMENT_NAME>4⅜% Treasury Gilt 2054</INSTRUMENT_NAME>
    <REDEMPTION_DATE>2054-07-31T00:00:00</REDEMPTION_DATE>
    <FIRST_ISSUE_DATE>2024-01-24T00:00:00</FIRST_ISSUE_DATE>
    <NOMINAL_AMOUNT_IN_ISSUE>42000.0</NOMINAL_AMOUNT_IN_ISSUE>
  </View_GILTS_IN_ISSUE>
</NewDataSet>
"""


def test_parse_borsa_btp_page_and_classification() -> None:
    rows = parse_borsa_btp_page(BORSA_HTML)
    assert len(rows) == 4

    by_isin = {row.isin: row for row in rows}

    assert by_isin["IT0001086567"].last_price == 100.787
    assert by_isin["IT0001086567"].coupon_display_percent == 3.625
    assert by_isin["IT0001086567"].maturity_date == date(2026, 11, 1)
    assert by_isin["IT0001086567"].is_nominal_btp

    assert not by_isin["IT0003745541"].is_nominal_btp
    assert not by_isin["IT0003268882"].is_nominal_btp
    assert by_isin["IT0003934657"].is_nominal_btp


def test_parse_dmo_attribute_and_child_element_rows() -> None:
    gilts = parse_dmo_gilts_xml(DMO_XML)

    assert len(gilts) == 2
    assert {gilt.isin for gilt in gilts} == {
        "GB00BL6C7720",
        "GB00BPSNBF73",
    }

    short = next(gilt for gilt in gilts if gilt.isin == "GB00BL6C7720")
    assert short.coupon_percent == 4.125
    assert short.maturity_date == date(2027, 1, 29)
    assert short.first_issue_date == date(2022, 10, 18)
    assert short.nominal_amount_outstanding_gbp == 35000.0

    long = next(gilt for gilt in gilts if gilt.isin == "GB00BPSNBF73")
    assert long.coupon_percent == 4.375
    assert long.maturity_date == date(2054, 7, 31)


def test_sources_and_statuses() -> None:
    italy = parse_borsa_btp_page(BORSA_HTML)[0]
    uk = parse_dmo_gilts_xml(DMO_XML)[0]

    assert italy.data_status == "PUBLIC_REFERENCE"
    assert italy.source_name == "Borsa Italiana MOT"
    assert uk.data_status == "OFFICIAL_REFERENCE"
    assert uk.source_name == "UK Debt Management Office"

def test_parse_borsa_detail_page_extracts_curve_grade_fields() -> None:
    detail = parse_borsa_btp_detail_page(
        BORSA_DETAIL_HTML,
        isin="IT0005707614",
    )

    assert detail.isin == "IT0005707614"
    assert detail.description == "Btp Fx 3.15% Jun31 Eur"
    assert detail.annual_coupon_percent == 3.15
    assert detail.periodic_coupon_percent == 1.575
    assert detail.coupon_frequency == "Semestrale"
    assert detail.day_count_convention == "ACT/ACT (ICMA)"
    assert detail.accrual_start_date == date(
        2026,
        5,
        4,
    )
    assert detail.first_coupon_date == date(
        2026,
        5,
        4,
    )
    assert detail.maturity_date == date(
        2031,
        6,
        1,
    )
    assert detail.reference_price == 98.44
    assert detail.gross_yield_percent == 3.54
    assert detail.reference_date == date(
        2026,
        9,
        4,
    )
    assert detail.has_exact_reference_yield
    assert detail.has_pricing_terms
    assert detail.market_source_name == (
        "Borsa Italiana / Skipper Informatica"
    )


def test_detail_page_rejects_isin_mismatch() -> None:
    import pytest

    from src.download_italy_btp_and_uk_gilt_universe import (
        SovereignUniverseExpansionParseError,
    )

    with pytest.raises(
        SovereignUniverseExpansionParseError,
        match="ISIN mismatch",
    ):
        parse_borsa_btp_detail_page(
            BORSA_DETAIL_HTML,
            isin="IT0000000000",
        )