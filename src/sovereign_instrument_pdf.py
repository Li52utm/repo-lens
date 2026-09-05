

from io import BytesIO
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.sovereign_instrument_report import (
    InstrumentReportHistoricalMetric,
    InstrumentReportHistoricalWindow,
    SovereignInstrumentReportModel,
)


class SovereignInstrumentPdfError(Exception):
    """Base exception for RepoLens instrument-report PDF generation."""


PAGE_WIDTH, PAGE_HEIGHT = A4
PAGE_MARGIN = 15 * mm


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "RepoLensTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            textColor=colors.HexColor("#111827"),
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "RepoLensSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=4 * mm,
        ),
        "section": ParagraphStyle(
            "RepoLensSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#111827"),
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "RepoLensBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "RepoLensSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.5,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=1.5 * mm,
        ),
        "metric_label": ParagraphStyle(
            "RepoLensMetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#6B7280"),
            alignment=TA_LEFT,
        ),
        "metric_value": ParagraphStyle(
            "RepoLensMetricValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#111827"),
            alignment=TA_RIGHT,
        ),
    }


def _fmt_number(
    value: float | None,
    *,
    decimals: int = 2,
    suffix: str = "",
    signed: bool = False,
) -> str:
    if value is None:
        return "N/A"

    sign = "+" if signed else ""

    return f"{float(value):{sign},.{decimals}f}{suffix}"


def _fmt_euro(
    value: float | None,
    *,
    decimals: int = 0,
    signed: bool = False,
) -> str:
    if value is None:
        return "N/A"

    sign = "+" if signed else ""

    return f"EUR {float(value):{sign},.{decimals}f}"


def _metric_cell(
    label: str,
    value: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    table = Table(
        [
            [
                Paragraph(
                    label,
                    styles["metric_label"],
                ),
            ],
            [
                Paragraph(
                    value,
                    styles["metric_value"],
                ),
            ],
        ],
        colWidths=[
            42 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3 * mm,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3 * mm,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
            ]
        )
    )

    return table


def _historical_metric_value(
    metric: InstrumentReportHistoricalMetric | None,
    field: str,
    *,
    decimals: int = 2,
    suffix: str = "",
    signed: bool = False,
) -> str:
    if metric is None:
        return "N/A"

    return _fmt_number(
        getattr(
            metric,
            field,
        ),
        decimals=decimals,
        suffix=suffix,
        signed=signed,
    )


def _historical_table(
    windows: Iterable[InstrumentReportHistoricalWindow],
    styles: dict[str, ParagraphStyle],
) -> Table:
    header = [
        "Window",
        "Yield low",
        "Yield high",
        "Median",
        "Current",
        "Percentile",
        "Move",
        "Spread",
    ]

    rows = [
        [
            Paragraph(
                heading,
                styles["small"],
            )
            for heading in header
        ]
    ]

    for window in windows:
        rows.append(
            [
                window.label,
                _historical_metric_value(
                    window.yield_percent,
                    "low",
                    decimals=3,
                    suffix="%",
                ),
                _historical_metric_value(
                    window.yield_percent,
                    "high",
                    decimals=3,
                    suffix="%",
                ),
                _historical_metric_value(
                    window.yield_percent,
                    "median",
                    decimals=3,
                    suffix="%",
                ),
                _historical_metric_value(
                    window.yield_percent,
                    "current",
                    decimals=3,
                    suffix="%",
                ),
                _historical_metric_value(
                    window.yield_percent,
                    "percentile",
                    decimals=1,
                    suffix="th",
                ),
                _historical_metric_value(
                    window.yield_percent,
                    "change_from_window_start",
                    decimals=1,
                    suffix=" bp",
                    signed=True,
                )
                if window.yield_percent is not None
                else "N/A",
                _historical_metric_value(
                    window.benchmark_spread_bp,
                    "current",
                    decimals=1,
                    suffix=" bp",
                    signed=True,
                ),
            ]
        )

    table = Table(
        rows,
        repeatRows=1,
        colWidths=[
            14 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#E5E7EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#111827"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
            ]
        )
    )

    return table


def _page_footer(
    canvas,
    doc,
) -> None:
    canvas.saveState()

    canvas.setStrokeColor(
        colors.HexColor("#D1D5DB")
    )
    canvas.setLineWidth(
        0.4
    )
    canvas.line(
        PAGE_MARGIN,
        11 * mm,
        PAGE_WIDTH - PAGE_MARGIN,
        11 * mm,
    )

    canvas.setFont(
        "Helvetica",
        6.5,
    )
    canvas.setFillColor(
        colors.HexColor("#6B7280")
    )

    canvas.drawString(
        PAGE_MARGIN,
        7 * mm,
        "RepoLens - research analytics, not an executable quote or trade recommendation.",
    )

    canvas.drawRightString(
        PAGE_WIDTH - PAGE_MARGIN,
        7 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def render_sovereign_instrument_report_pdf(
    report: SovereignInstrumentReportModel,
) -> bytes:
    """
    Render a professional, auditable PDF from the shared instrument report model.

    The function does not fetch data or create market values. It only formats
    already-sourced data and RepoLens-derived analytics from the report model.
    """
    styles = _styles()
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"RepoLens - {report.identity.display_name}",
        author="RepoLens",
        subject="Sovereign instrument market and repo intelligence report",
    )

    story = []

    story.append(
        Paragraph(
            "RepoLens Instrument Report",
            styles["title"],
        )
    )

    identity_line = (
        f"{report.identity.display_name} | "
        f"{report.identity.isin} | "
        f"{report.identity.country} | "
        f"{report.identity.currency} | "
        f"Maturity {report.identity.maturity_date.isoformat()}"
    )

    if report.identity.coupon_percent is not None:
        identity_line += (
            f" | Coupon {report.identity.coupon_percent:.3f}%"
        )

    story.append(
        Paragraph(
            identity_line,
            styles["subtitle"],
        )
    )

    generated_line = (
        f"Generated {report.generated_at.isoformat(sep=' ', timespec='minutes')} | "
        f"Market as of {report.market.as_of_date.isoformat()} | "
        f"{report.market.data_status} | "
        f"{report.market.source_name}"
    )

    story.append(
        Paragraph(
            generated_line,
            styles["small"],
        )
    )

    market_metrics = Table(
        [
            [
                _metric_cell(
                    "Price / 100",
                    _fmt_number(
                        report.market.price_per_100,
                        decimals=4,
                    ),
                    styles,
                ),
                _metric_cell(
                    "Yield",
                    _fmt_number(
                        report.market.yield_percent,
                        decimals=3,
                        suffix="%",
                    ),
                    styles,
                ),
                _metric_cell(
                    (
                        report.market.benchmark_name
                        or "Benchmark spread"
                    ),
                    _fmt_number(
                        report.market.benchmark_spread_bp,
                        decimals=1,
                        suffix=" bp",
                        signed=True,
                    ),
                    styles,
                ),
                _metric_cell(
                    "Data status",
                    report.market.data_status,
                    styles,
                ),
            ]
        ],
        colWidths=[
            44 * mm,
            44 * mm,
            44 * mm,
            44 * mm,
        ],
    )

    market_metrics.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1 * mm,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1 * mm,
                ),
            ]
        )
    )

    story.extend(
        [
            Paragraph(
                "Market snapshot",
                styles["section"],
            ),
            market_metrics,
            Spacer(
                1,
                3 * mm,
            ),
            Paragraph(
                "Historical trading context",
                styles["section"],
            ),
            _historical_table(
                report.historical_windows,
                styles,
            ),
        ]
    )

    if report.repo is not None:
        repo = report.repo

        repo_rows = [
            [
                "Repo term",
                f"{repo.repo_days} days",
                "Source",
                f"{repo.data_status} - {repo.source_name}",
            ],
            [
                "Specific repo",
                _fmt_number(
                    repo.specific_repo_rate_percent,
                    decimals=4,
                    suffix="%",
                ),
                "Matched GC",
                _fmt_number(
                    repo.gc_repo_rate_percent,
                    decimals=4,
                    suffix="%",
                ),
            ],
            [
                "Specialness",
                _fmt_number(
                    repo.specialness_bp,
                    decimals=2,
                    suffix=" bp",
                    signed=True,
                ),
                "History percentile",
                _fmt_number(
                    repo.historical_percentile,
                    decimals=1,
                    suffix="th",
                ),
            ],
            [
                "1D specialness move",
                _fmt_number(
                    repo.change_1d_bp,
                    decimals=2,
                    suffix=" bp",
                    signed=True,
                ),
                "1W specialness move",
                _fmt_number(
                    repo.change_1w_bp,
                    decimals=2,
                    suffix=" bp",
                    signed=True,
                ),
            ],
            [
                "Funding advantage vs GC",
                _fmt_euro(
                    repo.funding_advantage_vs_gc_eur,
                    decimals=0,
                    signed=True,
                ),
                "Repo-adjusted carry",
                _fmt_euro(
                    repo.repo_adjusted_carry_eur,
                    decimals=0,
                    signed=True,
                ),
            ],
        ]

        repo_table = Table(
            repo_rows,
            colWidths=[
                38 * mm,
                47 * mm,
                38 * mm,
                53 * mm,
            ],
        )

        repo_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#D1D5DB"),
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor("#F3F4F6"),
                    ),
                    (
                        "BACKGROUND",
                        (2, 0),
                        (2, -1),
                        colors.HexColor("#F3F4F6"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (2, 0),
                        (2, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (1, 0),
                        (1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTNAME",
                        (3, 0),
                        (3, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        2 * mm,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        2 * mm,
                    ),
                ]
            )
        )

        story.extend(
            [
                Paragraph(
                    "Repo intelligence",
                    styles["section"],
                ),
                repo_table,
            ]
        )

    narrative_items = [
        (
            "What happened",
            report.narrative.what_happened,
        ),
        (
            "Historical context",
            report.narrative.historical_context,
        ),
    ]

    if report.narrative.repo_context:
        narrative_items.append(
            (
                "Repo context",
                report.narrative.repo_context,
            )
        )

    if report.narrative.event_risk:
        narrative_items.append(
            (
                "Event risk",
                report.narrative.event_risk,
            )
        )

    if report.narrative.interpretation:
        narrative_items.append(
            (
                "Interpretation",
                report.narrative.interpretation,
            )
        )

    if report.narrative.uncertainty:
        narrative_items.append(
            (
                "Uncertainty",
                report.narrative.uncertainty,
            )
        )

    narrative_rows = [
        [
            Paragraph(
                heading,
                styles["metric_label"],
            ),
            Paragraph(
                text,
                styles["body"],
            ),
        ]
        for heading, text in narrative_items
    ]

    narrative_table = Table(
        narrative_rows,
        colWidths=[
            34 * mm,
            142 * mm,
        ],
    )

    narrative_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#F3F4F6"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2.5 * mm,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2.5 * mm,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2 * mm,
                ),
            ]
        )
    )

    story.extend(
        [
            Paragraph(
                "RepoLens narrative",
                styles["section"],
            ),
            narrative_table,
        ]
    )

    if report.events:
        event_rows = [
            [
                "Date",
                "Category",
                "Event",
                "Relevance",
                "Source",
            ]
        ]

        for event in report.events:
            event_rows.append(
                [
                    event.event_date.isoformat(),
                    event.category,
                    event.title,
                    event.relevance,
                    event.source_name,
                ]
            )

        event_table = Table(
            event_rows,
            repeatRows=1,
            colWidths=[
                23 * mm,
                28 * mm,
                50 * mm,
                51 * mm,
                24 * mm,
            ],
        )

        event_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#E5E7EB"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (-1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#D1D5DB"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        2 * mm,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        2 * mm,
                    ),
                ]
            )
        )

        story.extend(
            [
                Paragraph(
                    "Upcoming events",
                    styles["section"],
                ),
                event_table,
            ]
        )

    if report.provenance:
        provenance_rows = [
            [
                "Component",
                "Source",
                "Status",
                "As of",
                "Notes",
            ]
        ]

        for item in report.provenance:
            provenance_rows.append(
                [
                    item.label,
                    item.source_name,
                    item.data_status,
                    item.as_of,
                    item.notes or "",
                ]
            )

        provenance_table = Table(
            provenance_rows,
            repeatRows=1,
            colWidths=[
                31 * mm,
                42 * mm,
                28 * mm,
                28 * mm,
                47 * mm,
            ],
        )

        provenance_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#E5E7EB"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (-1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        6.8,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#D1D5DB"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        1.8 * mm,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        1.8 * mm,
                    ),
                ]
            )
        )

        story.extend(
            [
                Paragraph(
                    "Data provenance",
                    styles["section"],
                ),
                provenance_table,
            ]
        )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "Methodology note: historical statistics are descriptive. "
                "Repo inputs may be explicit desk/broker assumptions. "
                "RepoLens does not manufacture executable market or repo quotes. "
                "Historical analogues, when added, must be interpreted as context "
                "rather than forecasts."
            ),
            styles["small"],
        )
    )

    try:
        doc.build(
            story,
            onFirstPage=_page_footer,
            onLaterPages=_page_footer,
        )
    except Exception as error:
        raise SovereignInstrumentPdfError(
            f"Could not render sovereign instrument PDF: {error}"
        ) from error

    return buffer.getvalue()