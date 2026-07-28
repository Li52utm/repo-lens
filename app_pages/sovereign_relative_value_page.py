from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sovereign_instruments import (
    SOVEREIGN_INSTRUMENTS,
    SovereignCountry,
    SovereignInstrument,
)
from src.sovereign_relative_value import (
    DEFAULT_PARALLEL_SHOCKS_BP,
    DEFAULT_SPREAD_SHOCKS_BP,
    PositionDirection,
    RelativeValueLeg,
    RelativeValueValidationError,
    build_dv01_neutral_position,
    build_parallel_scenarios,
    build_spread_scenarios,
    position_to_frame,
)
from src.sovereign_snapshot import SovereignYieldInput


GERMAN_BENCHMARK_PATH = Path(
    "data/raw/sovereign/germany_benchmark_yields.csv"
)

REQUIRED_BENCHMARK_COLUMNS = {
    "observation_date",
    "country_code",
    "tenor_years",
    "yield_percent",
    "source_name",
    "data_status",
}


@st.cache_data(
    show_spinner=False
)
def load_german_benchmark_data(
    input_path: str,
) -> pd.DataFrame:
    """
    Load the official German benchmark-yield history.
    """
    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"German benchmark file does not exist: {path}"
        )

    data = pd.read_csv(
        path
    )

    missing_columns = (
        REQUIRED_BENCHMARK_COLUMNS
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "German benchmark data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data["observation_date"] = pd.to_datetime(
        data["observation_date"],
        errors="coerce",
    )

    data["tenor_years"] = pd.to_numeric(
        data["tenor_years"],
        errors="coerce",
    )

    data["yield_percent"] = pd.to_numeric(
        data["yield_percent"],
        errors="coerce",
    )

    data["country_code"] = (
        data["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data = data.dropna(
        subset=[
            "observation_date",
            "tenor_years",
            "yield_percent",
            "source_name",
            "data_status",
        ]
    )

    data = data.loc[
        data["country_code"].eq(
            "DE"
        )
    ]

    if data.empty:
        raise ValueError(
            "German benchmark data contains no valid observations."
        )

    return data


def format_number(
    value: object,
    decimals: int = 2,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Format optional dashboard values.
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


def format_euro(
    value: object,
    decimals: int = 0,
) -> str:
    """
    Format euro amounts.
    """
    return format_number(
        value=value,
        decimals=decimals,
        prefix="€",
    )


def instrument_label(
    instrument: SovereignInstrument,
) -> str:
    """
    Create a compact instrument selector label.
    """
    return (
        f"{instrument.country.value} "
        f"{instrument.benchmark_tenor_years}Y | "
        f"{instrument.display_name} | "
        f"{instrument.isin}"
    )


def latest_benchmark_date(
    benchmark_data: pd.DataFrame,
) -> date:
    """
    Return the latest official benchmark observation date.
    """
    latest_date = pd.to_datetime(
        benchmark_data[
            "observation_date"
        ],
        errors="coerce",
    ).max()

    if pd.isna(
        latest_date
    ):
        raise ValueError(
            "German benchmark data has no valid observation date."
        )

    return pd.Timestamp(
        latest_date
    ).date()


def latest_german_yield(
    benchmark_data: pd.DataFrame,
    tenor_years: int,
) -> float:
    """
    Return the latest official German yield for one tenor.
    """
    matches = benchmark_data.loc[
        benchmark_data[
            "tenor_years"
        ].eq(
            tenor_years
        )
    ].copy()

    if matches.empty:
        raise ValueError(
            f"No German {tenor_years}Y benchmark yield is available."
        )

    matches = matches.sort_values(
        "observation_date"
    )

    return float(
        matches.iloc[-1][
            "yield_percent"
        ]
    )


def create_yield_input(
    instrument: SovereignInstrument,
    yield_percent: float,
    observation_date: date,
    source_name: str,
) -> SovereignYieldInput | None:
    """
    Create a desk-input yield for Italian instruments.

    German instruments use official Bundesbank benchmark data.
    """
    if instrument.country == SovereignCountry.GERMANY:
        return None

    return SovereignYieldInput(
        isin=instrument.isin,
        yield_percent=float(
            yield_percent
        ),
        observation_date=observation_date,
        source_name=source_name,
    )


def build_spread_scenario_chart(
    scenarios: pd.DataFrame,
) -> go.Figure:
    """
    Plot full-repricing P&L under spread shocks.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scenarios[
                "spread_shock_bp"
            ],
            y=scenarios[
                "total_pnl_eur"
            ],
            name="Trade P&L",
            customdata=scenarios[
                [
                    "shocked_spread_bp",
                    "anchor_pnl_eur",
                    "hedge_pnl_eur",
                ]
            ],
            hovertemplate=(
                "Spread shock: %{x:+.0f} bp<br>"
                "Shocked spread: %{customdata[0]:.2f} bp<br>"
                "Anchor P&L: €%{customdata[1]:,.0f}<br>"
                "Hedge P&L: €%{customdata[2]:,.0f}<br>"
                "Total P&L: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
    )

    figure.update_layout(
        title="Spread-shock P&L",
        xaxis_title="Anchor-minus-hedge spread shock (bp)",
        yaxis_title="Trade P&L (€)",
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


def build_parallel_scenario_chart(
    scenarios: pd.DataFrame,
) -> go.Figure:
    """
    Plot trade P&L under parallel yield shocks.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scenarios[
                "parallel_shock_bp"
            ],
            y=scenarios[
                "total_pnl_eur"
            ],
            name="Trade P&L",
            customdata=scenarios[
                [
                    "anchor_pnl_eur",
                    "hedge_pnl_eur",
                ]
            ],
            hovertemplate=(
                "Parallel shock: %{x:+.0f} bp<br>"
                "Anchor P&L: €%{customdata[0]:,.0f}<br>"
                "Hedge P&L: €%{customdata[1]:,.0f}<br>"
                "Total P&L: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
    )

    figure.update_layout(
        title="Parallel-rate-shock P&L",
        xaxis_title="Parallel yield shock (bp)",
        yaxis_title="Trade P&L (€)",
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


def render_position_table(
    position_frame: pd.DataFrame,
) -> None:
    """
    Display both position legs.
    """
    st.dataframe(
        position_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "leg": st.column_config.TextColumn(
                "Leg"
            ),
            "isin": st.column_config.TextColumn(
                "ISIN"
            ),
            "direction": st.column_config.TextColumn(
                "Direction"
            ),
            "notional_eur": st.column_config.NumberColumn(
                "Face value",
                format="€%,.0f",
            ),
            "yield_percent": st.column_config.NumberColumn(
                "Yield",
                format="%.3f%%",
            ),
            "position_dv01_eur": (
                st.column_config.NumberColumn(
                    "Absolute DV01",
                    format="€%,.0f",
                )
            ),
            "signed_dv01_eur": (
                st.column_config.NumberColumn(
                    "Signed DV01",
                    format="€%,.0f",
                )
            ),
        },
    )


def render_spread_scenario_table(
    scenarios: pd.DataFrame,
) -> None:
    """
    Display spread-shock scenario results.
    """
    st.dataframe(
        scenarios,
        hide_index=True,
        width="stretch",
        column_config={
            "spread_shock_bp": (
                st.column_config.NumberColumn(
                    "Spread shock",
                    format="%+.0f bp",
                )
            ),
            "anchor_yield_shock_bp": (
                st.column_config.NumberColumn(
                    "Anchor yield shock",
                    format="%+.1f bp",
                )
            ),
            "hedge_yield_shock_bp": (
                st.column_config.NumberColumn(
                    "Hedge yield shock",
                    format="%+.1f bp",
                )
            ),
            "shocked_spread_bp": (
                st.column_config.NumberColumn(
                    "Shocked spread",
                    format="%.2f bp",
                )
            ),
            "anchor_pnl_eur": (
                st.column_config.NumberColumn(
                    "Anchor P&L",
                    format="€%,.0f",
                )
            ),
            "hedge_pnl_eur": (
                st.column_config.NumberColumn(
                    "Hedge P&L",
                    format="€%,.0f",
                )
            ),
            "total_pnl_eur": (
                st.column_config.NumberColumn(
                    "Total P&L",
                    format="€%,.0f",
                )
            ),
        },
    )


def render_parallel_scenario_table(
    scenarios: pd.DataFrame,
) -> None:
    """
    Display parallel-shock scenario results.
    """
    st.dataframe(
        scenarios,
        hide_index=True,
        width="stretch",
        column_config={
            "parallel_shock_bp": (
                st.column_config.NumberColumn(
                    "Parallel shock",
                    format="%+.0f bp",
                )
            ),
            "anchor_pnl_eur": (
                st.column_config.NumberColumn(
                    "Anchor P&L",
                    format="€%,.0f",
                )
            ),
            "hedge_pnl_eur": (
                st.column_config.NumberColumn(
                    "Hedge P&L",
                    format="€%,.0f",
                )
            ),
            "total_pnl_eur": (
                st.column_config.NumberColumn(
                    "Total P&L",
                    format="€%,.0f",
                )
            ),
        },
    )


def main() -> None:
    """
    Render the RepoLens Sovereign Relative Value Monitor.
    """
    try:
        benchmark_data = load_german_benchmark_data(
            str(
                GERMAN_BENCHMARK_PATH
            )
        )

        official_observation_date = latest_benchmark_date(
            benchmark_data
        )
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        st.error(
            "RepoLens could not load the German sovereign curve."
        )

        st.code(
            str(
                error
            )
        )

        st.info(
            "Run `python -m src.download_german_benchmark_yields` "
            "from the repository root, then reload."
        )

        st.stop()

    instruments_by_label = {
        instrument_label(
            instrument
        ): instrument
        for instrument in SOVEREIGN_INSTRUMENTS
    }

    labels = list(
        instruments_by_label
    )

    default_anchor_label = next(
        label
        for label, instrument in instruments_by_label.items()
        if (
            instrument.country == SovereignCountry.ITALY
            and instrument.benchmark_tenor_years == 10
        )
    )

    default_hedge_label = next(
        label
        for label, instrument in instruments_by_label.items()
        if (
            instrument.country == SovereignCountry.GERMANY
            and instrument.benchmark_tenor_years == 10
        )
    )

    with st.sidebar.expander(
        "Relative Value controls",
        expanded=True,
    ):
        anchor_label = st.selectbox(
            "Anchor instrument",
            options=labels,
            index=labels.index(
                default_anchor_label
            ),
            key="rv_anchor_instrument",
        )

        hedge_label = st.selectbox(
            "Hedge instrument",
            options=labels,
            index=labels.index(
                default_hedge_label
            ),
            key="rv_hedge_instrument",
        )

        anchor_instrument = instruments_by_label[
            anchor_label
        ]

        hedge_instrument = instruments_by_label[
            hedge_label
        ]

        anchor_direction_text = st.radio(
            "Anchor direction",
            options=[
                "Long",
                "Short",
            ],
            horizontal=True,
            key="rv_anchor_direction",
        )

        if anchor_direction_text == "Long":
            anchor_direction = PositionDirection.LONG
            hedge_direction = PositionDirection.SHORT
        else:
            anchor_direction = PositionDirection.SHORT
            hedge_direction = PositionDirection.LONG

        st.caption(
            f"Hedge direction: {hedge_direction.value}"
        )

        anchor_notional_eur = st.number_input(
            "Anchor face value (€)",
            min_value=1_000_000.0,
            value=10_000_000.0,
            step=1_000_000.0,
            format="%.0f",
            key="rv_anchor_notional",
        )

        earliest_settlement = max(
            anchor_instrument.issue_date,
            hedge_instrument.issue_date,
        )

        latest_settlement = min(
            anchor_instrument.maturity_date,
            hedge_instrument.maturity_date,
        ) - timedelta(
            days=1
        )

        default_settlement = max(
            earliest_settlement,
            official_observation_date,
        )

        default_settlement = min(
            default_settlement,
            latest_settlement,
        )

        settlement_date = st.date_input(
            "Settlement date",
            value=default_settlement,
            min_value=earliest_settlement,
            max_value=latest_settlement,
            key="rv_settlement_date",
        )

        st.markdown(
            "**Market yields**"
        )

        anchor_german_yield = latest_german_yield(
            benchmark_data=benchmark_data,
            tenor_years=anchor_instrument.benchmark_tenor_years,
        )

        hedge_german_yield = latest_german_yield(
            benchmark_data=benchmark_data,
            tenor_years=hedge_instrument.benchmark_tenor_years,
        )

        if anchor_instrument.country == SovereignCountry.ITALY:
            anchor_yield_percent = st.number_input(
                "Anchor yield (%)",
                min_value=-10.0,
                max_value=25.0,
                value=float(
                    anchor_german_yield
                    + 1.00
                ),
                step=0.01,
                format="%.3f",
                key=(
                    "rv_anchor_yield_"
                    f"{anchor_instrument.isin}"
                ),
            )

            anchor_source_name = st.text_input(
                "Anchor yield source",
                value="Desk input",
                key=(
                    "rv_anchor_source_"
                    f"{anchor_instrument.isin}"
                ),
            )
        else:
            anchor_yield_percent = anchor_german_yield
            anchor_source_name = "Deutsche Bundesbank"

            st.caption(
                "Anchor yield: "
                f"{anchor_yield_percent:.3f}% "
                "official German daily benchmark"
            )

        if hedge_instrument.country == SovereignCountry.ITALY:
            hedge_yield_percent = st.number_input(
                "Hedge yield (%)",
                min_value=-10.0,
                max_value=25.0,
                value=float(
                    hedge_german_yield
                    + 1.00
                ),
                step=0.01,
                format="%.3f",
                key=(
                    "rv_hedge_yield_"
                    f"{hedge_instrument.isin}"
                ),
            )

            hedge_source_name = st.text_input(
                "Hedge yield source",
                value="Desk input",
                key=(
                    "rv_hedge_source_"
                    f"{hedge_instrument.isin}"
                ),
            )
        else:
            hedge_yield_percent = hedge_german_yield
            hedge_source_name = "Deutsche Bundesbank"

            st.caption(
                "Hedge yield: "
                f"{hedge_yield_percent:.3f}% "
                "official German daily benchmark"
            )

    if anchor_instrument.isin == hedge_instrument.isin:
        st.warning(
            "Select two different instruments to construct a trade."
        )

        st.stop()

    try:
        anchor_yield_input = create_yield_input(
            instrument=anchor_instrument,
            yield_percent=float(
                anchor_yield_percent
            ),
            observation_date=settlement_date,
            source_name=anchor_source_name,
        )

        hedge_yield_input = create_yield_input(
            instrument=hedge_instrument,
            yield_percent=float(
                hedge_yield_percent
            ),
            observation_date=settlement_date,
            source_name=hedge_source_name,
        )

        anchor_leg = RelativeValueLeg(
            instrument=anchor_instrument,
            direction=anchor_direction,
            yield_input=anchor_yield_input,
        )

        hedge_leg = RelativeValueLeg(
            instrument=hedge_instrument,
            direction=hedge_direction,
            yield_input=hedge_yield_input,
        )

        position = build_dv01_neutral_position(
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            german_curve=benchmark_data,
            settlement_date=settlement_date,
            anchor_notional_eur=float(
                anchor_notional_eur
            ),
        )

        spread_scenarios = build_spread_scenarios(
            position=position,
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            spread_shocks_bp=DEFAULT_SPREAD_SHOCKS_BP,
        )

        parallel_scenarios = build_parallel_scenarios(
            position=position,
            anchor_leg=anchor_leg,
            hedge_leg=hedge_leg,
            parallel_shocks_bp=DEFAULT_PARALLEL_SHOCKS_BP,
        )

        position_frame = position_to_frame(
            position
        )
    except RelativeValueValidationError as error:
        st.error(
            "RepoLens could not construct the relative-value position."
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
            European sovereign relative value
        </div>
        <div class="repolens-title">
            Relative Value Monitor
        </div>
        <div class="repolens-subtitle">
            Construct DV01-neutral sovereign trades and analyse
            spread risk separately from parallel interest-rate risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-box status-event">
            <strong>
                {position.anchor_direction.value}
                {anchor_instrument.display_name}
                versus
                {position.hedge_direction.value}
                {hedge_instrument.display_name}
            </strong><br>
            Hedge face value is calculated by matching absolute DV01,
            not by using equal cash notionals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Trade construction</div>',
        unsafe_allow_html=True,
    )

    construction_columns = st.columns(
        5
    )

    construction_columns[0].metric(
        "Current spread",
        format_number(
            position.spread_bp,
            decimals=2,
            suffix=" bp",
        ),
        delta="Anchor minus hedge",
        delta_color="off",
        border=True,
    )

    construction_columns[1].metric(
        "Anchor face value",
        format_euro(
            position.anchor_notional_eur
        ),
        delta=position.anchor_direction.value,
        delta_color="off",
        border=True,
    )

    construction_columns[2].metric(
        "Hedge face value",
        format_euro(
            position.hedge_notional_eur
        ),
        delta=position.hedge_direction.value,
        delta_color="off",
        border=True,
    )

    construction_columns[3].metric(
        "Hedge ratio",
        format_number(
            position.hedge_notional_per_anchor_euro,
            decimals=4,
            suffix="x",
        ),
        delta="Hedge € / anchor €",
        delta_color="off",
        border=True,
    )

    construction_columns[4].metric(
        "DV01 hedge error",
        format_euro(
            position.dv01_hedge_error_eur,
            decimals=2,
        ),
        delta="Absolute residual",
        delta_color="off",
        border=True,
    )

    st.markdown(
        '<div class="section-label">Rate-risk profile</div>',
        unsafe_allow_html=True,
    )

    risk_columns = st.columns(
        4
    )

    risk_columns[0].metric(
        "Anchor DV01",
        format_euro(
            position.anchor_position_dv01_eur
        ),
        border=True,
    )

    risk_columns[1].metric(
        "Hedge DV01",
        format_euro(
            position.hedge_position_dv01_eur
        ),
        border=True,
    )

    risk_columns[2].metric(
        "Gross DV01",
        format_euro(
            position.gross_dv01_eur
        ),
        border=True,
    )

    risk_columns[3].metric(
        "Net signed DV01",
        format_euro(
            position.net_dv01_eur,
            decimals=2,
        ),
        delta="Near zero by construction",
        delta_color="off",
        border=True,
    )

    st.divider()

    spread_chart_column, parallel_chart_column = st.columns(
        2
    )

    with spread_chart_column:
        st.plotly_chart(
            build_spread_scenario_chart(
                spread_scenarios
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with parallel_chart_column:
        st.plotly_chart(
            build_parallel_scenario_chart(
                parallel_scenarios
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.divider()

    st.subheader(
        "DV01-neutral position"
    )

    render_position_table(
        position_frame
    )

    st.caption(
        "Signed DV01 incorporates long or short direction. "
        "The anchor and hedge legs offset at first order, while "
        "convexity and cross-market spread risk remain."
    )

    st.divider()

    spread_table_column, parallel_table_column = st.columns(
        2
    )

    with spread_table_column:
        st.subheader(
            "Spread scenario matrix"
        )

        render_spread_scenario_table(
            spread_scenarios
        )

    with parallel_table_column:
        st.subheader(
            "Parallel scenario matrix"
        )

        render_parallel_scenario_table(
            parallel_scenarios
        )

    st.divider()

    methodology_left, methodology_right = st.columns(
        2
    )

    with methodology_left:
        st.subheader(
            "Trade interpretation"
        )

        st.markdown(
            f"""
            The displayed spread is:

            **{anchor_instrument.display_name} yield −
            {hedge_instrument.display_name} yield**

            A positive spread shock means the anchor yield rises relative
            to the hedge yield. A negative shock means the spread narrows.

            For a long-anchor, short-hedge position, spread narrowing
            normally produces a gain and spread widening normally produces
            a loss.
            """
        )

    with methodology_right:
        st.subheader(
            "Data classification"
        )

        st.markdown(
            """
            German yields are official daily Deutsche Bundesbank
            benchmark observations and are not executable quotes.

            Italian yields are explicit user-supplied desk inputs.
            RepoLens does not manufacture or interpolate BTP market prices.

            Hedge ratios, DV01, spread measures and scenario P&L are
            RepoLens-derived research analytics.
            """
        )


main()