from __future__ import annotations

from datetime import date

from src.download_euronext_mts_sovereign_yields import (
    parse_euronext_mts_yields,
)


HTML = """
<html>
<body>
<table>
<tr><td>Euronext MTS BTP 10Y Yield</td><td>NLIX00006972</td><td>MIT10</td><td>EUR 4.15</td><td>0.00%</td><td>03 Sep 2026</td><td>+17.23%</td></tr>
<tr><td>Euronext MTS BTP 10Y Yield 11am</td><td>NLIX00007194</td><td>MIT1A</td><td>EUR 4.18</td><td>0.00%</td><td>03 Sep 2026</td><td>+19.09%</td></tr>
<tr><td>Euronext MTS BTP 2Y Yield</td><td>NLIX00006956</td><td>MIT2Y</td><td>EUR 3.16</td><td>0.00%</td><td>03 Sep 2026</td><td>+40.44%</td></tr>
<tr><td>Euronext MTS BTP 5Y Yield</td><td>NLIX00006964</td><td>MIT5Y</td><td>EUR 3.55</td><td>0.00%</td><td>03 Sep 2026</td><td>+25.89%</td></tr>
<tr><td>Euronext MTS BUND 10Y Yield</td><td>NLIX00006998</td><td>MDE10</td><td>EUR 3.35</td><td>0.00%</td><td>03 Sep 2026</td><td>+15.52%</td></tr>
<tr><td>Euronext MTS OAT 10Y Yield</td><td>NLIX00006980</td><td>MFR10</td><td>EUR 4.18</td><td>0.00%</td><td>03 Sep 2026</td><td>+17.09%</td></tr>
<tr><td>Euronext MTS SPGB 10Y Yield</td><td>NLIX00007038</td><td>MES10</td><td>EUR 3.77</td><td>0.00%</td><td>03 Sep 2026</td><td>+14.24%</td></tr>
<tr><td>Euronext MTS OLO 10Y Yield</td><td>NLIX00007020</td><td>MBE10</td><td>EUR 3.90</td><td>0.00%</td><td>03 Sep 2026</td><td>+14.71%</td></tr>
<tr><td>Euronext MTS RAGB 10Y Yield</td><td>NLIX00007012</td><td>MAT10</td><td>EUR 3.59</td><td>0.00%</td><td>03 Sep 2026</td><td>+12.89%</td></tr>
<tr><td>Euronext MTS DSL 10Y Yield</td><td>NLIX00007046</td><td>MNL10</td><td>EUR 3.44</td><td>0.00%</td><td>03 Sep 2026</td><td>+14.67%</td></tr>
<tr><td>Euronext MTS NXG 10Y Yield</td><td>NLIX00007004</td><td>MEU10</td><td>EUR 3.72</td><td>0.00%</td><td>03 Sep 2026</td><td>+16.25%</td></tr>
<tr><td>TEC1</td><td>FRIX00007925</td><td>TEC1Y</td><td>EUR 2.96</td><td>0.00%</td><td>03 Sep 2026</td><td>+38.32%</td></tr>
<tr><td>TEC30</td><td>FRIX00008014</td><td>TC30Y</td><td>EUR 4.95</td><td>0.00%</td><td>03 Sep 2026</td><td>+10.99%</td></tr>
</table>
</body>
</html>
"""


def test_parse_euronext_mts_yields() -> None:
    observations = parse_euronext_mts_yields(
        HTML
    )

    by_key = {
        (
            observation.country_code,
            observation.tenor_years,
        ): observation
        for observation in observations
    }

    assert by_key[
        (
            "IT",
            2,
        )
    ].yield_percent == 3.16

    assert by_key[
        (
            "IT",
            5,
        )
    ].yield_percent == 3.55

    assert by_key[
        (
            "IT",
            10,
        )
    ].yield_percent == 4.15

    assert by_key[
        (
            "DE",
            10,
        )
    ].yield_percent == 3.35

    assert by_key[
        (
            "ES",
            10,
        )
    ].yield_percent == 3.77

    assert by_key[
        (
            "BE",
            10,
        )
    ].yield_percent == 3.90

    assert by_key[
        (
            "AT",
            10,
        )
    ].yield_percent == 3.59

    assert by_key[
        (
            "NL",
            10,
        )
    ].yield_percent == 3.44

    assert by_key[
        (
            "EU",
            10,
        )
    ].yield_percent == 3.72

    assert by_key[
        (
            "FR",
            10,
        )
    ].curve_name == "OAT"

    assert by_key[
        (
            "FR",
            1,
        )
    ].yield_percent == 2.96

    assert by_key[
        (
            "FR",
            30,
        )
    ].yield_percent == 4.95


def test_observation_metadata() -> None:
    observation = parse_euronext_mts_yields(
        HTML
    )[0]

    assert observation.observation_date == date(
        2026,
        9,
        3,
    )

    assert observation.data_status == "PUBLIC_REFERENCE"
    assert observation.source_name == "Euronext MTS Indices"
