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
from src.sovereign_portfolio import (
    DEFAULT_ITALY_GERMANY_SPREAD_SHOCKS_BP,
    DEFAULT_PARALLEL_SHOCKS_BP,
    SovereignPortfolioPosition,
    SovereignPortfolioValidationError,
    aggregate_dv01_by_country,
    aggregate_dv01_by_tenor,
    build_italy_germany_spread_scenarios,
    build_parallel_portfolio_scenarios,
    build_portfolio,
    positions_to_frame,
    risk_contribution_frame,
)
from src.sovereign_relative_value import PositionDirection
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

DEFAULT_BOOK = (
    {
        "country": SovereignCountry.ITALY,
        "tenor": 10,
        "direction": PositionDirection.LONG,
        "notional": 15_000_000.0,
    },
    {
        "country": SovereignCountry.GERMANY,
        "tenor": 10,
        "direction": PositionDirection.SHORT,
        "notional": 12_000_000.0,
    },
    {
        "country": SovereignCountry.GERMANY,
        "tenor": 5,
        "direction": PositionDirection.LONG,
        "notional": 8_000_000.0,
    },
    {
        "country": SovereignCountry.ITALY,
        "tenor": 30,
        "direction": PositionDirection.SHORT,
        "notional": 5_000_000.0,
    },
)


@st.cache_data(
    show_spinner=False
)
def load_german_benchmark_data(
    input_path: str,
) -> pd.DataFrame:
    """
    Load and validate the German benchmark-yield history.
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


def format_euro(
    value: object,
    decimals: int = 0,
) -> str:
    """
    Format euro-denominated values.
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
    Create a compact desk-style selector label.
    """
    coupon_percent = (
        instrument.annual_coupon_rate
        * 100.0
    )

    return (
        f"{instrument.country_code} · "
        f"{instrument.benchmark_tenor_years}Y · "
        f"{coupon_percent:.2f}% "
        f"{instrument.security_type.value} · "
        f"{instrument.maturity_date.strftime('%b-%Y')}"
    )


def find_default_label(
    instruments_by_label: dict[
        str,
        SovereignInstrument,
    ],
    country: SovereignCountry,
    tenor_years: int,
) -> str:
    """
    Resolve a default instrument selector label.
    """
    return next(
        label
        for label, instrument in instruments_by_label.items()
        if (
            instrument.country == country
            and instrument.benchmark_tenor_years
            == tenor_years
        )
    )


def latest_benchmark_date(
    benchmark_data: pd.DataFrame,
) -> date:
    """
    Return the latest official German observation date.
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
    Return the latest German benchmark yield for one tenor.
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


def build_country_risk_chart(
    country_risk: pd.DataFrame,
) -> go.Figure:
    """
    Plot gross and net DV01 by country.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=country_risk["country"],
            y=country_risk[
                "gross_dv01_eur"
            ],
            name="Gross DV01",
            hovertemplate=(
                "%{x}<br>"
                "Gross DV01: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            x=country_risk["country"],
            y=country_risk[
                "net_dv01_eur"
            ],
            name="Net signed DV01",
            hovertemplate=(
                "%{x}<br>"
                "Net DV01: €%{y:,.0f}"
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
        title="DV01 by country",
        xaxis_title=None,
        yaxis_title="DV01 (€ per bp)",
        barmode="group",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
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


def build_tenor_risk_chart(
    tenor_risk: pd.DataFrame,
) -> go.Figure:
    """
    Plot gross and net DV01 by tenor bucket.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=tenor_risk[
                "benchmark_tenor_years"
            ],
            y=tenor_risk[
                "gross_dv01_eur"
            ],
            name="Gross DV01",
            hovertemplate=(
                "%{x:.0f}Y<br>"
                "Gross DV01: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            x=tenor_risk[
                "benchmark_tenor_years"
            ],
            y=tenor_risk[
                "net_dv01_eur"
            ],
            name="Net signed DV01",
            hovertemplate=(
                "%{x:.0f}Y<br>"
                "Net DV01: €%{y:,.0f}"
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
        title="DV01 by tenor bucket",
        xaxis_title="Benchmark tenor",
        yaxis_title="DV01 (€ per bp)",
        barmode="group",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        xaxis={
            "ticksuffix": "Y",
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


def build_parallel_scenario_chart(
    scenarios: pd.DataFrame,
) -> go.Figure:
    """
    Plot portfolio P&L under parallel yield shocks.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scenarios[
                "yield_shock_bp"
            ],
            y=scenarios[
                "portfolio_pnl_eur"
            ],
            customdata=scenarios[
                [
                    "largest_gain_eur",
                    "largest_loss_eur",
                ]
            ],
            hovertemplate=(
                "Parallel shock: %{x:+.0f} bp<br>"
                "Portfolio P&L: €%{y:,.0f}<br>"
                "Largest position gain: €%{customdata[0]:,.0f}<br>"
                "Largest position loss: €%{customdata[1]:,.0f}"
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
        title="Parallel-rate scenario P&L",
        xaxis_title="Yield shock (bp)",
        yaxis_title="Portfolio P&L (€)",
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


def build_spread_scenario_chart(
    scenarios: pd.DataFrame,
) -> go.Figure:
    """
    Plot portfolio P&L under Italy-Germany spread shocks.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scenarios[
                "spread_shock_bp"
            ],
            y=scenarios[
                "portfolio_pnl_eur"
            ],
            customdata=scenarios[
                [
                    "italy_yield_shock_bp",
                    "germany_yield_shock_bp",
                    "italy_pnl_eur",
                    "germany_pnl_eur",
                ]
            ],
            hovertemplate=(
                "Italy-Germany spread shock: %{x:+.0f} bp<br>"
                "Italy yield shock: %{customdata[0]:+.1f} bp<br>"
                "Germany yield shock: %{customdata[1]:+.1f} bp<br>"
                "Italy P&L: €%{customdata[2]:,.0f}<br>"
                "Germany P&L: €%{customdata[3]:,.0f}<br>"
                "Portfolio P&L: €%{y:,.0f}"
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
        title="Italy-Germany spread scenario P&L",
        xaxis_title="Spread shock (bp)",
        yaxis_title="Portfolio P&L (€)",
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


def render_positions_table(
    positions: pd.DataFrame,
) -> None:
    """
    Display the valued portfolio positions.
    """
    st.dataframe(
        positions,
        hide_index=True,
        width="stretch",
        column_config={
            "position_id": st.column_config.TextColumn(
                "Position"
            ),
            "isin": st.column_config.TextColumn(
                "ISIN"
            ),
            "display_name": st.column_config.TextColumn(
                "Instrument",
                width="large",
            ),
            "country": st.column_config.TextColumn(
                "Country"
            ),
            "security_type": st.column_config.TextColumn(
                "Type"
            ),
            "benchmark_tenor_years": (
                st.column_config.NumberColumn(
                    "Tenor",
                    format="%.0fY",
                )
            ),
            "direction": st.column_config.TextColumn(
                "Direction"
            ),
            "notional_eur": st.column_config.NumberColumn(
                "Face value",
                format="€%,.0f",
            ),
            "signed_notional_eur": (
                st.column_config.NumberColumn(
                    "Signed face value",
                    format="€%,.0f",
                )
            ),
            "yield_percent": st.column_config.NumberColumn(
                "Yield",
                format="%.3f%%",
            ),
            "dirty_price": st.column_config.NumberColumn(
                "Dirty price",
                format="%.4f",
            ),
            "market_value_eur": (
                st.column_config.NumberColumn(
                    "Market value",
                    format="€%,.0f",
                )
            ),
            "signed_market_value_eur": (
                st.column_config.NumberColumn(
                    "Signed market value",
                    format="€%,.0f",
                )
            ),
            "absolute_dv01_eur": (
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
            "data_status": st.column_config.TextColumn(
                "Data status"
            ),
            "source_name": st.column_config.TextColumn(
                "Yield source"
            ),
        },
    )


def render_risk_contribution_table(
    contributions: pd.DataFrame,
) -> None:
    """
    Display position-level gross DV01 contribution.
    """
    st.dataframe(
        contributions,
        hide_index=True,
        width="stretch",
        column_config={
            "position_id": st.column_config.TextColumn(
                "Position"
            ),
            "isin": st.column_config.TextColumn(
                "ISIN"
            ),
            "display_name": st.column_config.TextColumn(
                "Instrument",
                width="large",
            ),
            "country": st.column_config.TextColumn(
                "Country"
            ),
            "benchmark_tenor_years": (
                st.column_config.NumberColumn(
                    "Tenor",
                    format="%.0fY",
                )
            ),
            "direction": st.column_config.TextColumn(
                "Direction"
            ),
            "absolute_dv01_eur": (
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
            "gross_dv01_share": (
                st.column_config.ProgressColumn(
                    "Gross DV01 share",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1%%",
                )
            ),
        },
    )


def main() -> None:
    """
    Render the RepoLens Sovereign Portfolio Risk Book.
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

    st.markdown(
        """
        <div class="repolens-kicker">
            European sovereign risk management
        </div>
        <div class="repolens-title">
            Portfolio Risk Book
        </div>
        <div class="repolens-subtitle">
            Aggregate long and short sovereign positions into country,
            tenor, DV01 and full-repricing scenario risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    instruments_by_label = {
        instrument_label(
            instrument
        ): instrument
        for instrument in SOVEREIGN_INSTRUMENTS
    }

    labels = list(
        instruments_by_label
    )

    with st.container(border=True):
        st.markdown(
            '<div class="section-label">Book setup</div>',
            unsafe_allow_html=True,
        )

        setup_left, setup_middle, setup_right = st.columns(3)

        with setup_left:
            position_count = st.number_input(
                "Number of positions",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                key="portfolio_position_count",
            )

        with setup_middle:
            concentration_threshold = st.slider(
                "Concentration warning threshold",
                min_value=0.10,
                max_value=1.00,
                value=0.50,
                step=0.05,
                format="%.0f%%",
                key="portfolio_concentration_threshold",
            )

        with setup_right:
            st.caption(
                "German yields use official daily benchmark data. "
                "Italian yields remain explicit desk/broker inputs."
            )

    selected_positions: list[dict[str, object]] = []

    st.markdown(
        '<div class="section-label">Positions</div>',
        unsafe_allow_html=True,
    )

    for index in range(int(position_count)):
        default = DEFAULT_BOOK[index % len(DEFAULT_BOOK)]

        default_label = find_default_label(
            instruments_by_label=instruments_by_label,
            country=default["country"],
            tenor_years=int(default["tenor"]),
        )

        with st.container(border=True):
            st.markdown(f"**POS-{index + 1:02d}**")

            top_left, top_middle, top_right = st.columns(
                [2.4, 1.0, 1.2]
            )

            with top_left:
                selected_label = st.selectbox(
                    "Instrument",
                    options=labels,
                    index=labels.index(default_label),
                    key=f"portfolio_instrument_{index}",
                )

            selected_instrument = instruments_by_label[selected_label]

            default_direction = default["direction"]

            with top_middle:
                direction_text = st.radio(
                    "Direction",
                    options=["Long", "Short"],
                    index=(
                        0
                        if default_direction == PositionDirection.LONG
                        else 1
                    ),
                    horizontal=True,
                    key=f"portfolio_direction_{index}",
                )

            direction = (
                PositionDirection.LONG
                if direction_text == "Long"
                else PositionDirection.SHORT
            )

            with top_right:
                notional_eur = st.number_input(
                    "Face value (€)",
                    min_value=1_000_000.0,
                    value=float(default["notional"]),
                    step=1_000_000.0,
                    format="%.0f",
                    key=f"portfolio_notional_{index}",
                )

            st.caption(
                f"{selected_instrument.display_name} · "
                f"{selected_instrument.isin}"
            )

            matched_german_yield = latest_german_yield(
                benchmark_data=benchmark_data,
                tenor_years=selected_instrument.benchmark_tenor_years,
            )

            yield_percent: float | None = None
            source_name: str | None = None

            if selected_instrument.country == SovereignCountry.ITALY:
                market_left, market_right = st.columns(2)

                with market_left:
                    yield_percent = float(
                        st.number_input(
                            "Desk-input yield (%)",
                            min_value=-10.0,
                            max_value=25.0,
                            value=float(matched_german_yield + 1.00),
                            step=0.01,
                            format="%.3f",
                            key=f"portfolio_yield_{index}",
                        )
                    )

                with market_right:
                    source_name = st.text_input(
                        "Yield source",
                        value="Desk input",
                        key=f"portfolio_source_{index}",
                    )
            else:
                st.caption(
                    "Official German benchmark yield: "
                    f"{matched_german_yield:.3f}%"
                )

            selected_positions.append(
                {
                    "position_id": f"POS-{index + 1:02d}",
                    "instrument": selected_instrument,
                    "direction": direction,
                    "notional_eur": float(notional_eur),
                    "yield_percent": yield_percent,
                    "source_name": source_name,
                }
            )

    earliest_settlement = max(
        position["instrument"].issue_date
        for position in selected_positions
    )

    latest_settlement = min(
        position["instrument"].maturity_date
        for position in selected_positions
    ) - timedelta(days=1)

    default_settlement = max(
        earliest_settlement,
        official_observation_date,
    )

    default_settlement = min(
        default_settlement,
        latest_settlement,
    )

    with st.container(border=True):
        valuation_left, valuation_right = st.columns([1, 2])

        with valuation_left:
            settlement_date = st.date_input(
                "Settlement date",
                value=default_settlement,
                min_value=earliest_settlement,
                max_value=latest_settlement,
                key="portfolio_settlement_date",
            )

        with valuation_right:
            st.caption(
                "All positions are valued on the same settlement date. "
                "RepoLens uses full bond repricing for scenario analysis."
            )

    portfolio_positions: list[
        SovereignPortfolioPosition
    ] = []

    for position in selected_positions:
        instrument = position[
            "instrument"
        ]

        yield_input: SovereignYieldInput | None = None

        if instrument.country == SovereignCountry.ITALY:
            yield_input = SovereignYieldInput(
                isin=instrument.isin,
                yield_percent=float(
                    position[
                        "yield_percent"
                    ]
                ),
                observation_date=settlement_date,
                source_name=str(
                    position[
                        "source_name"
                    ]
                ),
            )

        portfolio_positions.append(
            SovereignPortfolioPosition(
                position_id=str(
                    position[
                        "position_id"
                    ]
                ),
                instrument=instrument,
                direction=position[
                    "direction"
                ],
                notional_eur=float(
                    position[
                        "notional_eur"
                    ]
                ),
                yield_input=yield_input,
            )
        )

    try:
        results, summary = build_portfolio(
            positions=tuple(
                portfolio_positions
            ),
            german_curve=benchmark_data,
            settlement_date=settlement_date,
            concentration_warning_threshold=float(
                concentration_threshold
            ),
        )

        positions_frame = positions_to_frame(
            results
        )

        country_risk = aggregate_dv01_by_country(
            results
        )

        tenor_risk = aggregate_dv01_by_tenor(
            results
        )

        contributions = risk_contribution_frame(
            results
        )

        parallel_scenarios = (
            build_parallel_portfolio_scenarios(
                results=results,
                yield_shocks_bp=DEFAULT_PARALLEL_SHOCKS_BP,
            )
        )

        spread_scenarios = (
            build_italy_germany_spread_scenarios(
                results=results,
                spread_shocks_bp=(
                    DEFAULT_ITALY_GERMANY_SPREAD_SHOCKS_BP
                ),
            )
        )
    except SovereignPortfolioValidationError as error:
        st.error(
            "RepoLens could not build the sovereign portfolio."
        )

        st.code(
            str(
                error
            )
        )

        st.stop()

    if summary.concentration_warning:
        st.markdown(
            f"""
            <div class="status-box status-monitor">
                <strong>Concentration warning</strong><br>
                Position {summary.largest_risk_position_id} contributes
                {summary.largest_risk_share:.1%} of gross portfolio DV01,
                above the configured
                {summary.concentration_warning_threshold:.0%} threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-box status-normal">
                <strong>Risk distribution within threshold</strong><br>
                The largest position contributes
                {summary.largest_risk_share:.1%} of gross portfolio DV01,
                below the configured
                {summary.concentration_warning_threshold:.0%} threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">Portfolio exposure</div>',
        unsafe_allow_html=True,
    )

    exposure_columns = st.columns(
        5
    )

    exposure_columns[0].metric(
        "Positions",
        f"{summary.position_count}",
        border=True,
    )

    exposure_columns[1].metric(
        "Gross face value",
        format_euro(
            summary.gross_notional_eur
        ),
        border=True,
    )

    exposure_columns[2].metric(
        "Net face value",
        format_euro(
            summary.net_notional_eur
        ),
        border=True,
    )

    exposure_columns[3].metric(
        "Gross market value",
        format_euro(
            summary.gross_market_value_eur
        ),
        border=True,
    )

    exposure_columns[4].metric(
        "Net market value",
        format_euro(
            summary.net_market_value_eur
        ),
        border=True,
    )

    st.markdown(
        '<div class="section-label">Interest-rate risk</div>',
        unsafe_allow_html=True,
    )

    risk_columns = st.columns(
        4
    )

    risk_columns[0].metric(
        "Gross DV01",
        format_euro(
            summary.gross_dv01_eur
        ),
        delta="Absolute risk",
        delta_color="off",
        border=True,
    )

    risk_columns[1].metric(
        "Net signed DV01",
        format_euro(
            summary.net_dv01_eur
        ),
        delta=(
            "Long duration"
            if summary.net_dv01_eur > 0.0
            else (
                "Short duration"
                if summary.net_dv01_eur < 0.0
                else "Neutral"
            )
        ),
        delta_color="off",
        border=True,
    )

    risk_columns[2].metric(
        "Largest risk position",
        summary.largest_risk_position_id,
        delta=format_euro(
            summary.largest_risk_position_dv01_eur
        ),
        delta_color="off",
        border=True,
    )

    risk_columns[3].metric(
        "Largest DV01 share",
        format_number(
            summary.largest_risk_share
            * 100.0,
            decimals=1,
            suffix="%",
        ),
        border=True,
    )

    st.divider()

    country_column, tenor_column = st.columns(
        2
    )

    with country_column:
        st.plotly_chart(
            build_country_risk_chart(
                country_risk
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with tenor_column:
        st.plotly_chart(
            build_tenor_risk_chart(
                tenor_risk
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    parallel_column, spread_column = st.columns(
        2
    )

    with parallel_column:
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

    with spread_column:
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

    st.divider()

    st.subheader(
        "Valued positions"
    )

    render_positions_table(
        positions_frame
    )

    st.caption(
        "German yields are official daily benchmark observations. "
        "Italian yields are explicit desk inputs. Market values, DV01 "
        "and scenario results are RepoLens-derived analytics."
    )

    st.divider()

    st.subheader(
        "Position risk contribution"
    )

    render_risk_contribution_table(
        contributions
    )

    st.divider()

    methodology_left, methodology_right = st.columns(
        2
    )

    with methodology_left:
        st.subheader(
            "Risk interpretation"
        )

        st.markdown(
            """
            **Gross DV01** measures total absolute first-order rate risk
            across all positions.

            **Net signed DV01** offsets long and short exposures and
            indicates the book's residual parallel-duration direction.

            Country and tenor views reveal where that risk is concentrated.
            """
        )

    with methodology_right:
        st.subheader(
            "Scenario methodology"
        )

        st.markdown(
            """
            Parallel scenarios reprice every bond under the same yield move.

            Italy-Germany spread scenarios split the requested move equally:
            Italian yields move by half the spread shock while German yields
            move by the opposite half.

            Scenario P&L uses full bond repricing rather than DV01 alone.
            """
        )


main()