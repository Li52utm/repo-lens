from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


MORNING_SHEET_PATH = Path(
    "outputs/morning_sheet/repolens_morning_sheet.csv"
)

REQUIRED_COLUMNS = {
    "observation_date",
    "estr_rate",
    "daily_change_bp",
    "funding_regime",
    "is_abnormal_move",
    "deposit_facility_rate",
    "policy_rate_changed_today",
    "estr_policy_spread_bp",
    "spread_z_score",
    "transmission_regime",
    "is_unusual_spread",
    "total_volume_eur_mn",
    "volume_change_pct",
    "active_banks",
    "transaction_count",
    "rate_dispersion_bp",
    "market_quality_score",
    "market_quality_change",
    "quality_regime",
    "is_market_quality_alert",
    "alert_count",
    "overall_status",
    "data_classification",
}


@st.cache_data(
    show_spinner=False
)
def load_morning_sheet(
    input_path: str,
) -> pd.DataFrame:
    """
    Load and validate the consolidated Morning Sheet dataset.
    """
    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Morning Sheet file does not exist: {path}"
        )

    data = pd.read_csv(
        path
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Morning Sheet is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data["observation_date"] = pd.to_datetime(
        data["observation_date"],
        errors="coerce",
    )

    numeric_columns = [
        "estr_rate",
        "daily_change_bp",
        "deposit_facility_rate",
        "estr_policy_spread_bp",
        "spread_z_score",
        "total_volume_eur_mn",
        "volume_change_pct",
        "active_banks",
        "transaction_count",
        "rate_dispersion_bp",
        "market_quality_score",
        "market_quality_change",
        "alert_count",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    boolean_columns = [
        "is_abnormal_move",
        "policy_rate_changed_today",
        "is_unusual_spread",
        "is_market_quality_alert",
    ]

    for column in boolean_columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                    "y",
                }
            )
        )

    data = (
        data
        .dropna(
            subset=[
                "observation_date",
            ]
        )
        .sort_values(
            "observation_date"
        )
        .drop_duplicates(
            subset="observation_date",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if data.empty:
        raise ValueError(
            "Morning Sheet contains no valid observations."
        )

    return data


def format_number(
    value: object,
    decimals: int = 2,
    suffix: str = "",
    prefix: str = "",
) -> str:
    """
    Format optional numeric values.
    """
    if pd.isna(
        value
    ):
        return "N/A"

    return (
        f"{prefix}"
        f"{float(value):,.{decimals}f}"
        f"{suffix}"
    )


def status_css_class(
    status: str,
) -> str:
    """
    Map status text to a dashboard display class.
    """
    if status == "High attention":
        return "status-high"

    if status == "Monitor":
        return "status-monitor"

    if status in {
        "Month-end watch",
        "Quarter-end watch",
    }:
        return "status-event"

    return "status-normal"


def select_history_window(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Render page-specific history controls.
    """
    available_dates = data[
        "observation_date"
    ]

    minimum_date = (
        available_dates
        .min()
        .date()
    )

    maximum_date = (
        available_dates
        .max()
        .date()
    )

    default_start = max(
        minimum_date,
        (
            pd.Timestamp(
                maximum_date
            )
            - pd.DateOffset(
                months=12
            )
        ).date(),
    )

    with st.sidebar.expander(
        "Morning Sheet controls",
        expanded=True,
    ):
        selected_dates = st.date_input(
            "History window",
            value=(
                default_start,
                maximum_date,
            ),
            min_value=minimum_date,
            max_value=maximum_date,
            key="morning_sheet_history_window",
        )

    if (
        isinstance(
            selected_dates,
            tuple,
        )
        and len(
            selected_dates
        ) == 2
    ):
        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = pd.Timestamp(
            selected_dates[1]
        )
    else:
        start_date = pd.Timestamp(
            default_start
        )

        end_date = pd.Timestamp(
            maximum_date
        )

    return data.loc[
        data["observation_date"].between(
            start_date,
            end_date,
        )
    ].copy()


def build_rate_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """
    Plot €STR against the ECB deposit facility rate.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=history["observation_date"],
            y=history["estr_rate"],
            mode="lines",
            name="€STR",
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "€STR: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=history["observation_date"],
            y=history[
                "deposit_facility_rate"
            ],
            mode="lines",
            name="ECB deposit facility",
            line={
                "dash": "dash",
            },
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Deposit facility: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="€STR versus ECB deposit facility rate",
        xaxis_title=None,
        yaxis_title="Rate (%)",
        hovermode="x unified",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure


def build_policy_spread_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """
    Plot the policy spread and rolling z-score.
    """
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    figure.add_trace(
        go.Scatter(
            x=history["observation_date"],
            y=history[
                "estr_policy_spread_bp"
            ],
            mode="lines",
            name="Policy spread",
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Spread: %{y:.3f} bp"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=history["observation_date"],
            y=history["spread_z_score"],
            mode="lines",
            name="Spread z-score",
            line={
                "dash": "dot",
            },
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Z-score: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    figure.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
        secondary_y=False,
    )

    figure.add_hline(
        y=2.0,
        line_width=1,
        line_dash="dot",
        secondary_y=True,
    )

    figure.add_hline(
        y=-2.0,
        line_width=1,
        line_dash="dot",
        secondary_y=True,
    )

    figure.update_yaxes(
        title_text="Spread (bp)",
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Z-score",
        secondary_y=True,
    )

    figure.update_layout(
        title="Policy transmission spread",
        xaxis_title=None,
        hovermode="x unified",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure


def build_market_quality_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """
    Plot market quality and eligible volume.
    """
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    figure.add_trace(
        go.Scatter(
            x=history["observation_date"],
            y=history[
                "market_quality_score"
            ],
            mode="lines",
            name="Quality score",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Bar(
            x=history["observation_date"],
            y=history[
                "total_volume_eur_mn"
            ],
            name="Eligible volume",
            opacity=0.28,
        ),
        secondary_y=True,
    )

    figure.add_hline(
        y=65.0,
        line_width=1,
        line_dash="dot",
        secondary_y=False,
    )

    figure.add_hline(
        y=35.0,
        line_width=1,
        line_dash="dot",
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="RepoLens score",
        range=[
            0,
            100,
        ],
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Volume (€mn)",
        secondary_y=True,
    )

    figure.update_layout(
        title="€STR market quality and eligible volume",
        xaxis_title=None,
        hovermode="x unified",
        height=440,
        barmode="overlay",
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure


def render_latest_summary(
    latest: pd.Series,
) -> None:
    """
    Render the latest Morning Sheet metrics.
    """
    status = str(
        latest["overall_status"]
    )

    st.markdown(
        f"""
        <div class="status-box {status_css_class(status)}">
            <strong>Desk status: {status}</strong><br>
            {int(latest["alert_count"])} active analytical alert(s)
            for {latest["observation_date"].strftime("%d %B %Y")}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Funding conditions</div>',
        unsafe_allow_html=True,
    )

    funding_columns = st.columns(
        4
    )

    funding_columns[0].metric(
        "€STR",
        format_number(
            latest["estr_rate"],
            decimals=3,
            suffix="%",
        ),
        delta=format_number(
            latest["daily_change_bp"],
            decimals=3,
            suffix=" bp",
        ),
        border=True,
    )

    funding_columns[1].metric(
        "ECB deposit facility",
        format_number(
            latest["deposit_facility_rate"],
            decimals=3,
            suffix="%",
        ),
        delta=(
            "Policy change"
            if bool(
                latest[
                    "policy_rate_changed_today"
                ]
            )
            else "Unchanged"
        ),
        delta_color="off",
        border=True,
    )

    funding_columns[2].metric(
        "Policy spread",
        format_number(
            latest[
                "estr_policy_spread_bp"
            ],
            decimals=3,
            suffix=" bp",
        ),
        delta=format_number(
            latest["spread_z_score"],
            decimals=2,
            suffix=" z",
        ),
        border=True,
    )

    funding_columns[3].metric(
        "Funding regime",
        str(
            latest["funding_regime"]
        ),
        delta=str(
            latest["transmission_regime"]
        ),
        delta_color="off",
        border=True,
    )

    st.markdown(
        '<div class="section-label">Market quality</div>',
        unsafe_allow_html=True,
    )

    quality_columns = st.columns(
        5
    )

    quality_columns[0].metric(
        "Quality score",
        format_number(
            latest[
                "market_quality_score"
            ],
            decimals=2,
            suffix=" / 100",
        ),
        delta=format_number(
            latest[
                "market_quality_change"
            ],
            decimals=2,
        ),
        border=True,
    )

    quality_columns[1].metric(
        "Eligible volume",
        format_number(
            latest[
                "total_volume_eur_mn"
            ],
            decimals=0,
            prefix="€",
            suffix="mn",
        ),
        delta=format_number(
            latest[
                "volume_change_pct"
            ],
            decimals=2,
            suffix="%",
        ),
        border=True,
    )

    quality_columns[2].metric(
        "Active banks",
        format_number(
            latest["active_banks"],
            decimals=0,
        ),
        border=True,
    )

    quality_columns[3].metric(
        "Transactions",
        format_number(
            latest["transaction_count"],
            decimals=0,
        ),
        border=True,
    )

    quality_columns[4].metric(
        "Rate dispersion",
        format_number(
            latest[
                "rate_dispersion_bp"
            ],
            decimals=3,
            suffix=" bp",
        ),
        delta=str(
            latest["quality_regime"]
        ),
        delta_color="off",
        border=True,
    )


def render_alerts(
    latest: pd.Series,
) -> None:
    """
    Render transparent alert explanations.
    """
    alert_table = pd.DataFrame(
        [
            {
                "Signal": "€STR daily move",
                "Triggered": bool(
                    latest[
                        "is_abnormal_move"
                    ]
                ),
                "Current regime": str(
                    latest["funding_regime"]
                ),
            },
            {
                "Signal": "Policy spread",
                "Triggered": bool(
                    latest[
                        "is_unusual_spread"
                    ]
                ),
                "Current regime": str(
                    latest[
                        "transmission_regime"
                    ]
                ),
            },
            {
                "Signal": "Market quality",
                "Triggered": bool(
                    latest[
                        "is_market_quality_alert"
                    ]
                ),
                "Current regime": str(
                    latest["quality_regime"]
                ),
            },
        ]
    )

    st.dataframe(
        alert_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Triggered": st.column_config.CheckboxColumn(
                "Triggered"
            ),
        },
    )


def render_history_table(
    history: pd.DataFrame,
) -> None:
    """
    Render a concise historical observation table.
    """
    display_data = history[
        [
            "observation_date",
            "estr_rate",
            "daily_change_bp",
            "estr_policy_spread_bp",
            "spread_z_score",
            "market_quality_score",
            "total_volume_eur_mn",
            "active_banks",
            "transaction_count",
            "alert_count",
            "overall_status",
        ]
    ].copy()

    display_data = display_data.sort_values(
        "observation_date",
        ascending=False,
    )

    st.dataframe(
        display_data,
        hide_index=True,
        width="stretch",
        column_config={
            "observation_date": st.column_config.DateColumn(
                "Date",
                format="DD MMM YYYY",
            ),
            "estr_rate": st.column_config.NumberColumn(
                "€STR",
                format="%.3f%%",
            ),
            "daily_change_bp": st.column_config.NumberColumn(
                "Daily move",
                format="%.3f bp",
            ),
            "estr_policy_spread_bp": (
                st.column_config.NumberColumn(
                    "Policy spread",
                    format="%.3f bp",
                )
            ),
            "spread_z_score": st.column_config.NumberColumn(
                "Spread z-score",
                format="%.2f",
            ),
            "market_quality_score": (
                st.column_config.NumberColumn(
                    "Quality score",
                    format="%.2f",
                )
            ),
            "total_volume_eur_mn": (
                st.column_config.NumberColumn(
                    "Volume (€mn)",
                    format="€%,.0f",
                )
            ),
            "active_banks": st.column_config.NumberColumn(
                "Banks",
                format="%.0f",
            ),
            "transaction_count": (
                st.column_config.NumberColumn(
                    "Transactions",
                    format="%.0f",
                )
            ),
            "alert_count": st.column_config.NumberColumn(
                "Alerts",
                format="%d",
            ),
        },
    )


def main() -> None:
    """
    Render the RepoLens Morning Sheet page.
    """
    try:
        morning_sheet = load_morning_sheet(
            str(
                MORNING_SHEET_PATH
            )
        )
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        st.error(
            "RepoLens could not load the Morning Sheet."
        )

        st.code(
            str(
                error
            )
        )

        st.info(
            "Run `python -m src.run_morning_sheet` "
            "from the repository root, then reload."
        )

        st.stop()

    selected_history = select_history_window(
        morning_sheet
    )

    if selected_history.empty:
        st.warning(
            "No observations fall inside the selected date window."
        )

        st.stop()

    latest = morning_sheet.iloc[-1]

    st.markdown(
        """
        <div class="repolens-kicker">
            European money markets
        </div>
        <div class="repolens-title">
            Morning Sheet
        </div>
        <div class="repolens-subtitle">
            Consolidated euro funding conditions,
            policy transmission and market quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_latest_summary(
        latest
    )

    st.divider()

    chart_left, chart_right = st.columns(
        2
    )

    with chart_left:
        st.plotly_chart(
            build_rate_chart(
                selected_history
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with chart_right:
        st.plotly_chart(
            build_policy_spread_chart(
                selected_history
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.plotly_chart(
        build_market_quality_chart(
            selected_history
        ),
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )

    st.divider()

    alert_column, context_column = st.columns(
        2
    )

    with alert_column:
        st.subheader(
            "Alert monitor"
        )

        render_alerts(
            latest
        )

    with context_column:
        st.subheader(
            "Latest desk context"
        )

        st.markdown(
            f"""
            **Funding regime:** {latest["funding_regime"]}  
            **Policy transmission:** {latest["transmission_regime"]}  
            **Market quality:** {latest["quality_regime"]}  
            **Observation date:** {
                latest["observation_date"].strftime("%d %B %Y")
            }
            """
        )

        st.info(
            "Underlying observations are official ECB data. "
            "Scores, regimes and alerts are RepoLens-derived "
            "research outputs."
        )

    st.divider()

    st.subheader(
        "Historical observations"
    )

    render_history_table(
        selected_history
    )


main()