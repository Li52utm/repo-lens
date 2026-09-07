from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SovereignMorningCommentary:
    headline: str
    summary: str
    bullets: tuple[str, ...]


def _format_direction(
    implied_change_bp: float | None,
) -> str:
    if implied_change_bp is None:
        return "was broadly unchanged on the available reference data"

    if implied_change_bp > 0.05:
        return f"widened by approximately {implied_change_bp:.1f}bp"

    if implied_change_bp < -0.05:
        return f"tightened by approximately {abs(implied_change_bp):.1f}bp"

    return "was broadly unchanged"


def build_sovereign_morning_commentary(
    latest_spreads: pd.DataFrame,
) -> SovereignMorningCommentary:
    if latest_spreads.empty:
        return SovereignMorningCommentary(
            headline="Sovereign spread context unavailable",
            summary=(
                "No persisted public sovereign-spread observations are "
                "available for the morning brief."
            ),
            bullets=(),
        )

    records = {
        str(
            row[
                "spread_name"
            ]
        ): row
        for _, row in latest_spreads.iterrows()
    }

    btp = records.get(
        "BTP-BUND"
    )

    if btp is not None:
        btp_change = btp.get(
            "implied_change_bp"
        )

        if pd.isna(
            btp_change
        ):
            btp_change = None

        headline = (
            f"BTP-Bund at {float(btp['spread_bp']):.0f}bp, "
            f"{_format_direction(btp_change)}"
        )
    else:
        headline = "European sovereign spread monitor"

    bullets: list[str] = []

    label_map = {
        "BTP-BUND": "Italy vs Germany",
        "BONO-BUND": "Spain vs Germany",
        "OAT-BUND": "France vs Germany",
    }

    for spread_name in (
        "BTP-BUND",
        "OAT-BUND",
        "BONO-BUND",
    ):
        row = records.get(
            spread_name
        )

        if row is None:
            continue

        implied_change = row.get(
            "implied_change_bp"
        )

        if pd.isna(
            implied_change
        ):
            implied_change = None

        bullets.append(
            f"{label_map[spread_name]}: "
            f"{float(row['spread_bp']):.0f}bp; "
            f"{_format_direction(implied_change)}. "
            f"10Y yields: {float(row['left_yield_percent']):.2f}% "
            f"vs Bund {float(row['right_yield_percent']):.2f}%."
        )

    widest = latest_spreads.loc[
        latest_spreads[
            "spread_bp"
        ].idxmax()
    ]

    summary = (
        f"The widest monitored 10Y sovereign spread is "
        f"{widest['spread_name']} at "
        f"{float(widest['spread_bp']):.0f}bp. "
        "Commentary is deterministic and derived from the persisted "
        "reference observations; it is not a trade recommendation."
    )

    return SovereignMorningCommentary(
        headline=headline,
        summary=summary,
        bullets=tuple(
            bullets
        ),
    )
