from __future__ import annotations

from datetime import date

import pytest

from src.download_sovereign_spreads import (
    parse_all_spreads,
)


HTML = """
<html>
<body>
<h2>SPREAD BTP VS BUND</h2>
<div>Attuale 82 Punti</div>
<div>Variazione -0,15%</div>
<div>Rendimento BTP 10 Anni +4,17%</div>
<div>Rendimento BUND 10 Anni +3,36%</div>
<div>Aggiornamento 07/09/2026 12.52</div>

<h2>SPREAD BONO VS BUND</h2>
<div>Attuale 44 Punti</div>
<div>Variazione +1,11%</div>
<div>Rendimento BONO 10 Anni +3,79%</div>
<div>Rendimento BUND 10 Anni +3,36%</div>
<div>Aggiornamento 07/09/2026 12.52</div>

<h2>SPREAD OAT VS BUND</h2>
<div>Attuale 86 Punti</div>
<div>Variazione -0,14%</div>
<div>Rendimento OAT 10 Anni +4,22%</div>
<div>Rendimento BUND 10 Anni +3,36%</div>
<div>Aggiornamento 07/09/2026 12.52</div>
</body>
</html>
"""


def test_parse_all_spreads() -> None:
    snapshots = parse_all_spreads(
        HTML
    )

    assert len(
        snapshots
    ) == 3

    btp = snapshots[
        0
    ]

    assert btp.spread_name == "BTP-BUND"
    assert btp.observation_date == date(
        2026,
        9,
        7,
    )
    assert btp.observation_time == "12:52"
    assert btp.spread_bp == pytest.approx(
        82.0
    )
    assert btp.change_percent == pytest.approx(
        -0.15
    )
    assert btp.left_yield_percent == pytest.approx(
        4.17
    )
    assert btp.right_yield_percent == pytest.approx(
        3.36
    )


def test_implied_change_is_derived_from_percentage_move() -> None:
    btp = parse_all_spreads(
        HTML
    )[
        0
    ]

    assert btp.implied_previous_spread_bp is not None
    assert btp.implied_change_bp is not None
    assert btp.implied_change_bp < 0.0


def test_status_is_clean_public_reference() -> None:
    snapshot = parse_all_spreads(
        HTML
    )[
        0
    ]

    assert snapshot.data_status == "PUBLIC_REFERENCE"