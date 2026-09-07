from __future__ import annotations

from datetime import date

from src.download_european_sovereign_universe import (
    parse_belgium,
    parse_france,
    parse_germany,
    parse_spain,
)


GERMANY_HTML = """
<table>
<tr><th>Anleihe</th><th>Fälligkeit</th><th>Kupon</th><th>Umlaufend</th><th>Letzte Emission</th><th>ISIN</th></tr>
<tr><td>2024 Schatz</td><td>17.09.2026</td><td>2,70 %</td><td>19.000 Mio. €</td><td>24.09.2024</td><td>DE000BU22064</td></tr>
<tr><td>2017 (2027) Bund</td><td>15.02.2027</td><td>0,25 %</td><td>30.500 Mio. €</td><td>16.04.2020</td><td>DE0001102416</td></tr>
</table>
"""


FRANCE_HTML = """
<table>
<tr><th>Code ISIN</th><th>Libellé</th><th>Encours (€)</th></tr>
<tr><td>FR001400FYQ4</td><td>OAT 2.50% 24 September 2026</td><td>24,030,000,000.00</td></tr>
<tr><td>FR0014003513</td><td>OAT 0.00% 25 February 2027</td><td>38,448,000,000.00</td></tr>
</table>
"""


SPAIN_HTML = """
<table>
<tr><th>Código ISIN</th><th>Fecha de vencimiento</th><th>Saldo en circulación</th><th>Coeficiente</th><th>Valor facial</th></tr>
<tr><td>ES00000128H5 O 1,30</td><td>31/10/2026</td><td>28.677,5</td><td></td><td></td></tr>
<tr><td>ES0000012J15 B 0,00</td><td>31/01/2027</td><td>22.790,2</td><td></td><td></td></tr>
<tr><td>ES0L02610092</td><td>09/10/2026</td><td>9.984,2</td><td></td><td></td></tr>
</table>
"""


BELGIUM_HTML = """
<table>
<tr><th>Maturity Date</th><th>Coupon</th><th>ISIN Code</th><th>Nr</th><th>Net outstanding (EUR)</th></tr>
<tr><td>22/06/2027</td><td>0.80</td><td>BE0000341504</td><td>81</td><td>16,485,000,000.00</td></tr>
<tr><td>28/03/2028</td><td>5.50</td><td>BE0000291972</td><td>31</td><td>19,337,939,135.27</td></tr>
</table>
"""


def test_parse_germany() -> None:
    items = parse_germany(
        GERMANY_HTML
    )

    assert len(
        items
    ) == 2

    assert items[
        0
    ].coupon_percent == 2.70

    assert items[
        0
    ].outstanding_eur == 19_000_000_000.0


def test_parse_france() -> None:
    items = parse_france(
        FRANCE_HTML
    )

    assert len(
        items
    ) == 2

    assert items[
        0
    ].isin == "FR001400FYQ4"

    assert items[
        0
    ].maturity_date == date(
        2026,
        9,
        24,
    )

    assert items[
        0
    ].outstanding_eur == 24_030_000_000.0


def test_parse_spain_and_exclude_bills() -> None:
    items = parse_spain(
        SPAIN_HTML
    )

    assert len(
        items
    ) == 2

    assert {
        item.instrument_type
        for item in items
    } == {
        "BONO",
        "OBLIGACION",
    }

    assert items[
        0
    ].outstanding_eur == 28_677_500_000.0


def test_parse_belgium() -> None:
    items = parse_belgium(
        BELGIUM_HTML
    )

    assert len(
        items
    ) == 2

    assert items[
        1
    ].coupon_percent == 5.50

    assert items[
        1
    ].outstanding_eur == 19_337_939_135.27