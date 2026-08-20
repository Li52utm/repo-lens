from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.repo_specialness_store import (
    RepoSpecialnessStoreError,
    load_repo_specialness_records,
)
from src.repo_specials_screener import (
    RepoSpecialsScreenRow,
    RepoSpecialsScreenerError,
    build_repo_specials_screen,
)


def format_age(
    seconds: float,
) -> str:
    """
    Format quote age without implying a market-specific stale threshold.
    """
    if seconds < 60.0:
        return f"{seconds:.0f}s"

    minutes = seconds / 60.0

    if minutes < 60.0:
        return f"{minutes:.0f}m"

    hours = minutes / 60.0

    if hours < 24.0:
        return f"{hours:.1f}h"

    days = hours / 24.0
    return f"{days:.1f}d"


def context_status(
    row: RepoSpecialsScreenRow,
) -> str:
    """
    Summarise comparison context without inventing a quality score.
    """
    if row.fully_context_matched:
        return "Matched"

    if row.context_warning_count <= 0:
        return "Unknown"

    return f"{row.context_warning_count} warning(s)"


def optional_number(
    value: float | None,
    *,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value:,.{decimals}f}"
        f"{suffix}"
    )


def screen_to_frame(
    rows: tuple[
        RepoSpecialsScreenRow,
        ...,
    ],
) -> pd.DataFrame:
    """
    Convert screen rows to a trader-readable table.
    """
    return pd.DataFrame(
        [
            {
                "isin": row.isin,
                "currency": row.currency,
                "repo_days": row.repo_days,
                "specific_repo_rate_percent": (
                    row.specific_repo_rate_percent
                ),
                "gc_repo_rate_percent": (
                    row.gc_repo_rate_percent
                ),
                "specialness_bp": row.specialness_bp,
                "historical_percentile": (
                    row.historical_percentile
                ),
                "historical_median_bp": (
                    row.historical_median_bp
                ),
                "change_vs_previous_bp": (
                    row.change_vs_previous_bp
                ),
                "z_score": row.z_score,
                "historical_observation_count": (
                    row.historical_observation_count
                ),
                "quote_age": format_age(
                    row.quote_age_seconds
                ),
                "quote_age_seconds": row.quote_age_seconds,
                "quote_time_gap_seconds": (
                    row.quote_time_gap_seconds
                ),
                "context": context_status(
                    row
                ),
                "context_warning_count": (
                    row.context_warning_count
                ),
                "financing_benefit_vs_gc_eur": (
                    row.financing_benefit_vs_gc_eur
                ),
                "purchase_price_eur": (
                    row.purchase_price_eur
                ),
                "gc_basket": (
                    row.gc_basket_name
                    or "Unspecified"
                ),
                "specific_venue": (
                    row.specific_venue
                    or "Unspecified"
                ),
                "gc_venue": (
                    row.gc_venue
                    or "Unspecified"
                ),
                "specific_source": (
                    row.specific_source_name
                ),
                "gc_source": row.gc_source_name,
                "observation_timestamp": (
                    row.observation_timestamp
                ),
            }
            for row in rows
        ]
    )


def build_specialness_chart(
    frame: pd.DataFrame,
) -> go.Figure:
    """
    Plot current specialness against historical percentile where available.
    """
    chart_data = frame.dropna(
        subset=[
            "historical_percentile",
        ]
    ).copy()

    figure = go.Figure()

    if not chart_data.empty:
        figure.add_trace(
            go.Scatter(
                x=chart_data[
                    "historical_percentile"
                ],
                y=chart_data[
                    "specialness_bp"
                ],
                mode="markers",
                customdata=chart_data[
                    [
                        "isin",
                        "repo_days",
                        "specific_repo_rate_percent",
                        "gc_repo_rate_percent",
                        "quote_age",
                        "context",
                    ]
                ],
                hovertemplate=(
                    "%{customdata[0]} · "
                    "%{customdata[1]}d<br>"
                    "Specialness: %{y:+.2f} bp<br>"
                    "Historical percentile: %{x:.1f}%<br>"
                    "Specific: %{customdata[2]:.3f}%<br>"
                    "GC: %{customdata[3]:.3f}%<br>"
                    "Age: %{customdata[4]}<br>"
                    "Context: %{customdata[5]}"
                    "<extra></extra>"
                ),
                name="Current observations",
            )
        )

    figure.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
    )

    figure.update_layout(
        title="Current specialness vs matched historical percentile",
        xaxis_title="Historical percentile (%)",
        yaxis_title="GC − specific repo (bp)",
        hovermode="closest",
        height=430,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        showlegend=False,
    )

    return figure


def apply_filters(
    frame: pd.DataFrame,
    *,
    currencies: list[str],
    repo_terms: list[int],
    context_filter: str,
    minimum_specialness_bp: float | None,
) -> pd.DataFrame:
    """
    Apply user-selected desk filters without changing the analytical ranking.
    """
    filtered = frame.copy()

    if currencies:
        filtered = filtered.loc[
            filtered[
                "currency"
            ].isin(
                currencies
            )
        ]

    if repo_terms:
        filtered = filtered.loc[
            filtered[
                "repo_days"
            ].isin(
                repo_terms
            )
        ]

    if context_filter == "Fully matched only":
        filtered = filtered.loc[
            filtered[
                "context_warning_count"
            ].eq(
                0
            )
        ]

    elif context_filter == "Warnings only":
        filtered = filtered.loc[
            filtered[
                "context_warning_count"
            ].gt(
                0
            )
        ]

    if minimum_specialness_bp is not None:
        filtered = filtered.loc[
            filtered[
                "specialness_bp"
            ].ge(
                minimum_specialness_bp
            )
        ]

    return filtered.reset_index(
        drop=True
    )


def render_screen_table(
    frame: pd.DataFrame,
) -> None:
    """
    Render the ranked specials screen.
    """
    display_columns = [
        "isin",
        "currency",
        "repo_days",
        "specific_repo_rate_percent",
        "gc_repo_rate_percent",
        "specialness_bp",
        "historical_percentile",
        "historical_median_bp",
        "change_vs_previous_bp",
        "z_score",
        "historical_observation_count",
        "quote_age",
        "quote_time_gap_seconds",
        "context",
        "financing_benefit_vs_gc_eur",
        "gc_basket",
    ]

    st.dataframe(
        frame[
            display_columns
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "isin": st.column_config.TextColumn(
                "ISIN",
                width="medium",
            ),
            "currency": st.column_config.TextColumn(
                "CCY"
            ),
            "repo_days": st.column_config.NumberColumn(
                "Term",
                format="%dd",
            ),
            "specific_repo_rate_percent": (
                st.column_config.NumberColumn(
                    "Specific",
                    format="%.3f%%",
                )
            ),
            "gc_repo_rate_percent": (
                st.column_config.NumberColumn(
                    "GC",
                    format="%.3f%%",
                )
            ),
            "specialness_bp": (
                st.column_config.NumberColumn(
                    "Specialness",
                    format="%+.2f bp",
                )
            ),
            "historical_percentile": (
                st.column_config.NumberColumn(
                    "Hist percentile",
                    format="%.1f%%",
                )
            ),
            "historical_median_bp": (
                st.column_config.NumberColumn(
                    "Hist median",
                    format="%+.2f bp",
                )
            ),
            "change_vs_previous_bp": (
                st.column_config.NumberColumn(
                    "Δ previous",
                    format="%+.2f bp",
                )
            ),
            "z_score": st.column_config.NumberColumn(
                "Z-score",
                format="%+.2f",
            ),
            "historical_observation_count": (
                st.column_config.NumberColumn(
                    "Hist obs",
                    format="%d",
                )
            ),
            "quote_age": st.column_config.TextColumn(
                "Age"
            ),
            "quote_time_gap_seconds": (
                st.column_config.NumberColumn(
                    "Quote gap",
                    format="%.0f sec",
                )
            ),
            "context": st.column_config.TextColumn(
                "Context",
                width="medium",
            ),
            "financing_benefit_vs_gc_eur": (
                st.column_config.NumberColumn(
                    "Funding benefit",
                    format="€%,.2f",
                )
            ),
            "gc_basket": st.column_config.TextColumn(
                "GC basket",
                width="medium",
            ),
        },
    )


def main() -> None:
    """
    Render the RepoLens Specials Screener.
    """
    as_of = datetime.now(
        timezone.utc
    )

    try:
        records = (
            load_repo_specialness_records()
        )

        screen_rows = (
            build_repo_specials_screen(
                records=records,
                as_of=as_of,
            )
        )

    except (
        RepoSpecialnessStoreError,
        RepoSpecialsScreenerError,
        OSError,
    ) as error:
        st.error(
            "RepoLens could not build the Specials Screener."
        )
        st.code(
            str(
                error
            )
        )
        st.stop()

    st.markdown(
        """
        <div class="repolens-kicker">
            Repo and collateral intelligence
        </div>
        <div class="repolens-title">
            Specials Screener
        </div>
        <div class="repolens-subtitle">
            Rank saved specific-repo observations against GC and their own
            matched history. Current specialness, historical context, quote
            freshness and market-comparison quality are kept separate so raw
            market inputs are not mistaken for RepoLens interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not screen_rows:
        st.info(
            "No saved structured repo observations are available yet. "
            "Use Repo Calculator → GC comparison → Save market observation "
            "to build the specials universe."
        )

        st.markdown(
            """
            **The screener will rank observations using:**

            - GC − specific repo specialness
            - matched historical percentile and median
            - change versus the previous saved observation
            - quote age and GC/specific timestamp gap
            - venue / clearing / market-segment comparison context
            - financing benefit versus GC when cash economics are available
            """
        )

        st.stop()

    frame = screen_to_frame(
        screen_rows
    )

    available_currencies = sorted(
        frame[
            "currency"
        ].dropna().unique().tolist()
    )

    available_terms = sorted(
        int(
            value
        )
        for value
        in frame[
            "repo_days"
        ].dropna().unique().tolist()
    )

    with st.container(
        border=True
    ):
        st.markdown(
            '<div class="section-label">Screen controls</div>',
            unsafe_allow_html=True,
        )

        filter_columns = st.columns(
            [
                1.2,
                1.2,
                1.2,
                1.0,
            ]
        )

        with filter_columns[0]:
            currencies = st.multiselect(
                "Currency",
                options=available_currencies,
                default=available_currencies,
                key="specials_screen_currency",
            )

        with filter_columns[1]:
            repo_terms = st.multiselect(
                "Repo term",
                options=available_terms,
                default=available_terms,
                format_func=lambda value: (
                    f"{value}d"
                ),
                key="specials_screen_term",
            )

        with filter_columns[2]:
            context_filter = st.selectbox(
                "Comparison context",
                options=[
                    "All",
                    "Fully matched only",
                    "Warnings only",
                ],
                index=0,
                key="specials_screen_context",
            )

        with filter_columns[3]:
            minimum_specialness_bp = (
                st.number_input(
                    "Min specialness (bp)",
                    value=None,
                    step=1.0,
                    placeholder="No minimum",
                    key=(
                        "specials_screen_min_specialness"
                    ),
                )
            )

    filtered = apply_filters(
        frame,
        currencies=currencies,
        repo_terms=repo_terms,
        context_filter=context_filter,
        minimum_specialness_bp=(
            float(
                minimum_specialness_bp
            )
            if minimum_specialness_bp
            is not None
            else None
        ),
    )

    if filtered.empty:
        st.warning(
            "No current repo observations match the selected screen filters."
        )
        st.stop()

    st.markdown(
        '<div class="section-label">Desk snapshot</div>',
        unsafe_allow_html=True,
    )

    top_row = filtered.iloc[
        0
    ]

    with_history = filtered[
        "historical_percentile"
    ].notna()

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Markets on screen",
        f"{len(filtered):,}",
        border=True,
    )

    metric_columns[1].metric(
        "Top specialness",
        (
            f"{top_row['specialness_bp']:+.2f} bp"
        ),
        delta=str(
            top_row[
                "isin"
            ]
        ),
        delta_color="off",
        border=True,
    )

    metric_columns[2].metric(
        "Top historical percentile",
        (
            f"{filtered.loc[with_history, 'historical_percentile'].max():.1f}%"
            if with_history.any()
            else "N/A"
        ),
        delta="Matched history only",
        delta_color="off",
        border=True,
    )

    metric_columns[3].metric(
        "Context matched",
        (
            f"{int(filtered['context_warning_count'].eq(0).sum())}"
            f" / {len(filtered)}"
        ),
        border=True,
    )

    metric_columns[4].metric(
        "As of",
        as_of.strftime(
            "%H:%M:%S UTC"
        ),
        delta=as_of.strftime(
            "%d %b %Y"
        ),
        delta_color="off",
        border=True,
    )

    st.caption(
        "Ranking remains descending by current GC-minus-specific specialness, "
        "then historical percentile, then observation recency. RepoLens does "
        "not impose a universal bp threshold for calling collateral special."
    )

    st.divider()

    chart_column, focus_column = st.columns(
        [
            1.6,
            1.0,
        ]
    )

    with chart_column:
        st.plotly_chart(
            build_specialness_chart(
                filtered
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with focus_column:
        st.subheader(
            "Top-ranked observation"
        )

        st.metric(
            "Specialness",
            (
                f"{top_row['specialness_bp']:+.2f} bp"
            ),
            delta=(
                "GC − specific"
            ),
            delta_color="off",
            border=True,
        )

        focus_left, focus_right = st.columns(
            2
        )

        focus_left.metric(
            "Specific repo",
            (
                f"{top_row['specific_repo_rate_percent']:.3f}%"
            ),
            border=True,
        )

        focus_right.metric(
            "GC",
            (
                f"{top_row['gc_repo_rate_percent']:.3f}%"
            ),
            border=True,
        )

        st.markdown(
            f"""
            **ISIN:** `{top_row["isin"]}`  
            **Term:** {int(top_row["repo_days"])}d  
            **Historical percentile:** {
                optional_number(
                    top_row["historical_percentile"],
                    decimals=1,
                    suffix="%",
                )
            }  
            **Historical median:** {
                optional_number(
                    top_row["historical_median_bp"],
                    decimals=2,
                    suffix=" bp",
                )
            }  
            **Change vs previous:** {
                optional_number(
                    top_row["change_vs_previous_bp"],
                    decimals=2,
                    suffix=" bp",
                )
            }  
            **Quote age:** {top_row["quote_age"]}  
            **Context:** {top_row["context"]}  
            **GC basket:** {top_row["gc_basket"]}
            """
        )

    st.divider()

    st.subheader(
        "Ranked collateral"
    )

    render_screen_table(
        filtered
    )

    st.caption(
        "Rows represent the latest saved observation for each exact "
        "ISIN / currency / repo-term market as of the current UTC time. "
        "Earlier matched observations are used only for historical context."
    )

    with st.expander(
        "Methodology and data classification",
        expanded=False,
    ):
        st.markdown(
            """
            **Market inputs**

            Specific repo rates, GC references, timestamps, venues, clearing
            types, market segments and source descriptions originate from the
            observations explicitly saved through Repo Calculator. RepoLens
            does not represent these inputs as executable live quotes.

            **Derived analytics**

            Specialness is `GC repo rate − specific repo rate`, expressed in
            basis points. Historical percentile, median, z-score and change
            versus previous are calculated only against earlier observations
            with the same ISIN, currency and repo term.

            **Comparison quality**

            Currency and repo term are hard matching requirements. Venue,
            clearing type, counterparty segment and GC basket identity are
            surfaced separately. A context warning does not invalidate the
            arithmetic; it warns that the observations may not be economically
            like-for-like.

            **Ranking**

            The default screen ranks by current specialness, then matched
            historical percentile, then recency. No universal threshold is
            used to label an instrument "special".
            """
        )


main()