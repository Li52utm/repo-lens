from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


MORNING_SHEET_PATH = Path(
    "outputs/morning_sheet/repolens_morning_sheet.csv"
)

REQUIRED_COLUMNS = {
    "observation_date",
    "daily_change_bp",
    "change_z_score",
    "is_abnormal_move",
    "estr_policy_spread_bp",
    "spread_z_score",
    "is_unusual_spread",
    "total_volume_eur_mn",
    "volume_change_pct",
    "active_banks",
    "transaction_count",
    "rate_dispersion_bp",
    "market_quality_score",
    "market_quality_change",
    "is_market_quality_alert",
    "alert_count",
    "overall_status",
}


@st.cache_data(
    show_spinner=False
)
def load_risk_data(
    input_path: str,
) -> pd.DataFrame:
    """
    Load the existing Morning Sheet for risk analysis.
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
            "Risk Monitor is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data["observation_date"] = pd.to_datetime(
        data["observation_date"],
        errors="coerce",
    )

    numeric_columns = [
        "daily_change_bp",
        "change_z_score",
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
            "Risk Monitor contains no valid observations."
        )

    return data


def safe_float(
    value: object,
    default: float = 0.0,
) -> float:
    """
    Convert an optional value to float.
    """
    if pd.isna(
        value
    ):
        return default

    return float(
        value
    )


def calculate_risk_score(
    change_z_score: float,
    spread_z_score: float,
    market_quality_score: float,
    volume_change_pct: float,
    rate_dispersion_bp: float,
    funding_z_limit: float,
    spread_z_limit: float,
    quality_warning_level: float,
    volume_decline_limit: float,
    dispersion_limit_bp: float,
) -> tuple[float, dict[str, float]]:
    """
    Calculate a transparent 0-to-100 market-risk score.

    This is signal risk, not portfolio VaR or expected loss.
    """
    funding_component = min(
        abs(
            change_z_score
        )
        / funding_z_limit,
        2.0,
    ) * 20.0

    policy_component = min(
        abs(
            spread_z_score
        )
        / spread_z_limit,
        2.0,
    ) * 25.0

    quality_shortfall = max(
        quality_warning_level
        - market_quality_score,
        0.0,
    )

    quality_component = min(
        quality_shortfall
        / max(
            quality_warning_level,
            1.0,
        ),
        1.0,
    ) * 25.0

    volume_stress = max(
        -volume_change_pct
        - volume_decline_limit,
        0.0,
    )

    volume_component = min(
        volume_stress
        / max(
            volume_decline_limit,
            1.0,
        ),
        1.0,
    ) * 15.0

    dispersion_excess = max(
        rate_dispersion_bp
        - dispersion_limit_bp,
        0.0,
    )

    dispersion_component = min(
        dispersion_excess
        / max(
            dispersion_limit_bp,
            0.01,
        ),
        1.0,
    ) * 15.0

    components = {
        "Funding move": funding_component,
        "Policy spread": policy_component,
        "Market quality": quality_component,
        "Volume deterioration": volume_component,
        "Rate dispersion": dispersion_component,
    }

    total_score = min(
        sum(
            components.values()
        ),
        100.0,
    )

    return total_score, components


def classify_risk_score(
    risk_score: float,
) -> str:
    """
    Classify the composite signal-risk score.
    """
    if risk_score >= 70.0:
        return "High risk"

    if risk_score >= 40.0:
        return "Elevated risk"

    if risk_score >= 20.0:
        return "Moderate risk"

    return "Low risk"


def calculate_limit_breaches(
    latest: pd.Series,
    funding_z_limit: float,
    spread_z_limit: float,
    quality_warning_level: float,
    volume_decline_limit: float,
    dispersion_limit_bp: float,
) -> pd.DataFrame:
    """
    Evaluate current conditions against user-selected limits.
    """
    rows = [
        {
            "Risk factor": "Funding move z-score",
            "Current value": abs(
                safe_float(
                    latest["change_z_score"]
                )
            ),
            "Limit": funding_z_limit,
            "Breached": abs(
                safe_float(
                    latest["change_z_score"]
                )
            ) >= funding_z_limit,
            "Direction": "Higher is riskier",
        },
        {
            "Risk factor": "Policy-spread z-score",
            "Current value": abs(
                safe_float(
                    latest["spread_z_score"]
                )
            ),
            "Limit": spread_z_limit,
            "Breached": abs(
                safe_float(
                    latest["spread_z_score"]
                )
            ) >= spread_z_limit,
            "Direction": "Higher absolute value is riskier",
        },
        {
            "Risk factor": "Market-quality score",
            "Current value": safe_float(
                latest[
                    "market_quality_score"
                ]
            ),
            "Limit": quality_warning_level,
            "Breached": safe_float(
                latest[
                    "market_quality_score"
                ]
            ) <= quality_warning_level,
            "Direction": "Lower is riskier",
        },
        {
            "Risk factor": "Daily volume decline",
            "Current value": max(
                -safe_float(
                    latest[
                        "volume_change_pct"
                    ]
                ),
                0.0,
            ),
            "Limit": volume_decline_limit,
            "Breached": safe_float(
                latest[
                    "volume_change_pct"
                ]
            ) <= -volume_decline_limit,
            "Direction": "Larger decline is riskier",
        },
        {
            "Risk factor": "Rate dispersion",
            "Current value": safe_float(
                latest[
                    "rate_dispersion_bp"
                ]
            ),
            "Limit": dispersion_limit_bp,
            "Breached": safe_float(
                latest[
                    "rate_dispersion_bp"
                ]
            ) >= dispersion_limit_bp,
            "Direction": "Higher is riskier",
        },
    ]

    return pd.DataFrame(
        rows
    )


def calculate_historical_risk(
    data: pd.DataFrame,
    funding_z_limit: float,
    spread_z_limit: float,
    quality_warning_level: float,
    volume_decline_limit: float,
    dispersion_limit_bp: float,
) -> pd.DataFrame:
    """
    Recalculate the transparent risk score for each observation.
    """
    historical = data.copy()

    calculated = [
        calculate_risk_score(
            change_z_score=safe_float(
                row.change_z_score
            ),
            spread_z_score=safe_float(
                row.spread_z_score
            ),
            market_quality_score=safe_float(
                row.market_quality_score,
                default=50.0,
            ),
            volume_change_pct=safe_float(
                row.volume_change_pct
            ),
            rate_dispersion_bp=safe_float(
                row.rate_dispersion_bp
            ),
            funding_z_limit=funding_z_limit,
            spread_z_limit=spread_z_limit,
            quality_warning_level=quality_warning_level,
            volume_decline_limit=volume_decline_limit,
            dispersion_limit_bp=dispersion_limit_bp,
        )[0]
        for row in historical.itertuples(
            index=False
        )
    ]

    historical["risk_score"] = calculated

    historical["risk_regime"] = (
        historical["risk_score"]
        .apply(
            classify_risk_score
        )
    )

    historical["five_day_risk_change"] = (
        historical["risk_score"]
        .diff(
            periods=5
        )
    )

    return historical


def build_risk_history_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """
    Plot the composite risk score and alert count.
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
            y=history["risk_score"],
            mode="lines",
            name="Risk score",
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Risk score: %{y:.1f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Bar(
            x=history["observation_date"],
            y=history["alert_count"],
            name="Source alerts",
            opacity=0.30,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Alerts: %{y:.0f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    figure.add_hrect(
        y0=70.0,
        y1=100.0,
        opacity=0.08,
        line_width=0,
        secondary_y=False,
    )

    figure.add_hrect(
        y0=40.0,
        y1=70.0,
        opacity=0.05,
        line_width=0,
        secondary_y=False,
    )

    figure.add_hline(
        y=70.0,
        line_dash="dot",
        line_width=1,
        secondary_y=False,
    )

    figure.add_hline(
        y=40.0,
        line_dash="dot",
        line_width=1,
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Risk score",
        range=[
            0,
            100,
        ],
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Alerts",
        rangemode="tozero",
        secondary_y=True,
    )

    figure.update_layout(
        title="Composite signal-risk history",
        xaxis_title=None,
        hovermode="x unified",
        height=450,
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


def build_component_chart(
    components: dict[str, float],
) -> go.Figure:
    """
    Plot current contributions to the risk score.
    """
    component_data = pd.DataFrame(
        {
            "Component": list(
                components.keys()
            ),
            "Contribution": list(
                components.values()
            ),
        }
    )

    figure = go.Figure(
        go.Bar(
            x=component_data[
                "Contribution"
            ],
            y=component_data[
                "Component"
            ],
            orientation="h",
            hovertemplate=(
                "%{y}<br>"
                "Contribution: %{x:.1f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Current risk contribution",
        xaxis_title="Risk points",
        yaxis_title=None,
        height=380,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
    )

    return figure


def render_scenario_controls() -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Render interactive analytical risk limits.
    """
    with st.sidebar.expander(
        "Risk limits",
        expanded=True,
    ):
        funding_z_limit = st.slider(
            "Funding-move z-score limit",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
        )

        spread_z_limit = st.slider(
            "Policy-spread z-score limit",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
        )

        quality_warning_level = st.slider(
            "Market-quality warning level",
            min_value=20.0,
            max_value=60.0,
            value=35.0,
            step=1.0,
        )

        volume_decline_limit = st.slider(
            "Daily volume decline limit (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
        )

        dispersion_limit_bp = st.slider(
            "Rate-dispersion limit (bp)",
            min_value=1.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
        )

    return (
        funding_z_limit,
        spread_z_limit,
        quality_warning_level,
        volume_decline_limit,
        dispersion_limit_bp,
    )


def main() -> None:
    """
    Render the RepoLens Risk Monitor.
    """
    try:
        risk_data = load_risk_data(
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
            "RepoLens could not load the Risk Monitor dataset."
        )

        st.code(
            str(
                error
            )
        )

        st.stop()

    (
        funding_z_limit,
        spread_z_limit,
        quality_warning_level,
        volume_decline_limit,
        dispersion_limit_bp,
    ) = render_scenario_controls()

    risk_history = calculate_historical_risk(
        data=risk_data,
        funding_z_limit=funding_z_limit,
        spread_z_limit=spread_z_limit,
        quality_warning_level=quality_warning_level,
        volume_decline_limit=volume_decline_limit,
        dispersion_limit_bp=dispersion_limit_bp,
    )

    latest = risk_history.iloc[-1]

    current_score, components = calculate_risk_score(
        change_z_score=safe_float(
            latest["change_z_score"]
        ),
        spread_z_score=safe_float(
            latest["spread_z_score"]
        ),
        market_quality_score=safe_float(
            latest[
                "market_quality_score"
            ],
            default=50.0,
        ),
        volume_change_pct=safe_float(
            latest["volume_change_pct"]
        ),
        rate_dispersion_bp=safe_float(
            latest["rate_dispersion_bp"]
        ),
        funding_z_limit=funding_z_limit,
        spread_z_limit=spread_z_limit,
        quality_warning_level=quality_warning_level,
        volume_decline_limit=volume_decline_limit,
        dispersion_limit_bp=dispersion_limit_bp,
    )

    current_regime = classify_risk_score(
        current_score
    )

    one_day_change = (
        current_score
        - float(
            risk_history.iloc[-2][
                "risk_score"
            ]
        )
        if len(
            risk_history
        ) >= 2
        else float("nan")
    )

    five_day_change = (
        current_score
        - float(
            risk_history.iloc[-6][
                "risk_score"
            ]
        )
        if len(
            risk_history
        ) >= 6
        else float("nan")
    )

    limit_breaches = calculate_limit_breaches(
        latest=latest,
        funding_z_limit=funding_z_limit,
        spread_z_limit=spread_z_limit,
        quality_warning_level=quality_warning_level,
        volume_decline_limit=volume_decline_limit,
        dispersion_limit_bp=dispersion_limit_bp,
    )

    breach_count = int(
        limit_breaches[
            "Breached"
        ].sum()
    )

    st.markdown(
        """
        <div class="repolens-kicker">
            Market and signal risk
        </div>
        <div class="repolens-title">
            Risk Monitor
        </div>
        <div class="repolens-subtitle">
            Transparent funding-risk limits, breach monitoring
            and deterioration analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This page measures market-signal risk. It is not portfolio "
        "VaR, expected shortfall, P&L-at-risk or regulatory capital."
    )

    summary_columns = st.columns(
        5
    )

    summary_columns[0].metric(
        "Composite risk score",
        f"{current_score:.1f} / 100",
        delta=(
            f"{one_day_change:+.1f} today"
            if pd.notna(
                one_day_change
            )
            else "N/A"
        ),
        border=True,
    )

    summary_columns[1].metric(
        "Risk regime",
        current_regime,
        border=True,
    )

    summary_columns[2].metric(
        "Current limit breaches",
        f"{breach_count} / 5",
        border=True,
    )

    summary_columns[3].metric(
        "Five-day risk change",
        (
            f"{five_day_change:+.1f}"
            if pd.notna(
                five_day_change
            )
            else "N/A"
        ),
        border=True,
    )

    summary_columns[4].metric(
        "Source alerts",
        f"{int(latest['alert_count'])}",
        delta=str(
            latest["overall_status"]
        ),
        delta_color="off",
        border=True,
    )

    st.divider()

    chart_left, chart_right = st.columns(
        [
            1.7,
            1,
        ]
    )

    with chart_left:
        st.plotly_chart(
            build_risk_history_chart(
                risk_history
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with chart_right:
        st.plotly_chart(
            build_component_chart(
                components
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.divider()

    st.subheader(
        "Risk-limit monitor"
    )

    st.dataframe(
        limit_breaches,
        hide_index=True,
        width="stretch",
        column_config={
            "Current value": st.column_config.NumberColumn(
                "Current value",
                format="%.2f",
            ),
            "Limit": st.column_config.NumberColumn(
                "Limit",
                format="%.2f",
            ),
            "Breached": st.column_config.CheckboxColumn(
                "Breached"
            ),
        },
    )

    st.subheader(
        "Desk interpretation"
    )

    if current_score >= 70.0:
        st.error(
            "Multiple risk channels are materially elevated. "
            "Review funding moves, policy transmission and market "
            "depth before relying on normal-condition assumptions."
        )
    elif current_score >= 40.0:
        st.warning(
            "Conditions are elevated. Review breached limits and "
            "whether stress is broadening across multiple channels."
        )
    elif current_score >= 20.0:
        st.info(
            "Moderate signal risk is present, but conditions are not "
            "currently classified as severe."
        )
    else:
        st.success(
            "Current market signals remain inside the selected risk "
            "limits with limited evidence of broad funding stress."
        )

    st.markdown(
        """
        **Risk score construction**

        The score combines funding-move extremity, policy-spread
        extremity, market-quality deterioration, volume decline and
        rate dispersion. Each component is capped so one extreme input
        cannot dominate the entire score.

        **Limitations**

        The score contains no positions, notionals, duration,
        convexity, counterparty exposure, collateral haircuts or P&L.
        It therefore supports market monitoring rather than replacing
        formal portfolio risk management.
        """
    )


main()