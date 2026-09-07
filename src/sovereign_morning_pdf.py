from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.sovereign_morning_commentary import (
    SovereignMorningCommentary,
)


def _value(
    value: object,
    *,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    if value is None or pd.isna(
        value
    ):
        return "N/A"

    if isinstance(
        value,
        float,
    ):
        return (
            f"{value:.{decimals}f}"
            f"{suffix}"
        )

    return str(
        value
    )


def build_sovereign_morning_pdf(
    *,
    monitor: pd.DataFrame,
    spreads: pd.DataFrame,
    commentary: SovereignMorningCommentary,
    generated_at: datetime,
) -> bytes:
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(
            A4
        ),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="RepoLens Morning Sovereign Brief",
        author="RepoLens",
    )

    styles = getSampleStyleSheet()
    story: list[object] = []

    story.append(
        Paragraph(
            "RepoLens Morning Sovereign Brief",
            styles[
                "Title"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Generated "
                f"{generated_at.strftime('%d %b %Y %H:%M:%S')}"
            ),
            styles[
                "Normal"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    story.append(
        Paragraph(
            commentary.headline,
            styles[
                "Heading2"
            ],
        )
    )

    story.append(
        Paragraph(
            commentary.summary,
            styles[
                "Normal"
            ],
        )
    )

    for bullet in commentary.bullets:
        story.append(
            Paragraph(
                f"• {bullet}",
                styles[
                    "Normal"
                ],
            )
        )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    if not spreads.empty:
        story.append(
            Paragraph(
                "10Y sovereign spreads",
                styles[
                    "Heading2"
                ],
            )
        )

        spread_rows = [
            [
                "Spread",
                "Current",
                "Implied Δ",
                "Left 10Y",
                "Bund 10Y",
                "As of",
            ]
        ]

        for _, row in spreads.iterrows():
            spread_rows.append(
                [
                    str(
                        row[
                            "spread_name"
                        ]
                    ),
                    _value(
                        float(
                            row[
                                "spread_bp"
                            ]
                        ),
                        decimals=0,
                        suffix="bp",
                    ),
                    _value(
                        row[
                            "implied_change_bp"
                        ],
                        decimals=1,
                        suffix="bp",
                    ),
                    _value(
                        float(
                            row[
                                "left_yield_percent"
                            ]
                        ),
                        decimals=2,
                        suffix="%",
                    ),
                    _value(
                        float(
                            row[
                                "right_yield_percent"
                            ]
                        ),
                        decimals=2,
                        suffix="%",
                    ),
                    str(
                        row[
                            "as_of"
                        ]
                    ),
                ]
            )

        spread_table = Table(
            spread_rows,
            repeatRows=1,
        )

        spread_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        colors.HexColor(
                            "#E9EEF5"
                        ),
                    ),
                    (
                        "GRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.35,
                        colors.HexColor(
                            "#A8B2C1"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                ]
            )
        )

        story.append(
            spread_table
        )

        story.append(
            Spacer(
                1,
                5 * mm,
            )
        )

    story.append(
        Paragraph(
            "Italian bond observations",
            styles[
                "Heading2"
            ],
        )
    )

    bond_rows = [
        [
            "Sector",
            "Bond",
            "ISIN",
            "Maturity",
            "Price",
            "Yield",
            "Obs",
            "Status",
        ]
    ]

    for _, row in monitor.iterrows():
        bond_rows.append(
            [
                str(
                    row[
                        "Sector"
                    ]
                ),
                str(
                    row[
                        "Bond"
                    ]
                ),
                str(
                    row[
                        "ISIN"
                    ]
                ),
                str(
                    row[
                        "Maturity"
                    ]
                ),
                _value(
                    row[
                        "Price"
                    ],
                    decimals=4,
                ),
                _value(
                    row[
                        "Yield"
                    ],
                    decimals=3,
                    suffix="%",
                ),
                _value(
                    row[
                        "Obs"
                    ],
                ),
                str(
                    row[
                        "Status"
                    ]
                ),
            ]
        )

    bond_table = Table(
        bond_rows,
        repeatRows=1,
        colWidths=[
            18 * mm,
            58 * mm,
            31 * mm,
            26 * mm,
            20 * mm,
            20 * mm,
            25 * mm,
            31 * mm,
        ],
    )

    bond_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.HexColor(
                        "#E9EEF5"
                    ),
                ),
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.35,
                    colors.HexColor(
                        "#A8B2C1"
                    ),
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7.5,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "TOP",
                ),
            ]
        )
    )

    story.append(
        bond_table
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "Data status: public/official reference observations and "
                "RepoLens-derived calculations. Reference data are not "
                "executable bid/offer quotes. Implied spread changes are "
                "derived from the source-reported percentage move."
            ),
            styles[
                "Normal"
            ],
        )
    )

    document.build(
        story
    )

    return buffer.getvalue()
