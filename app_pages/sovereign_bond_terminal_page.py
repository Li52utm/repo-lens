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
from src.sovereign_snapshot import (
    DEFAULT_SCENARIO_SHOCKS_BP,
    SnapshotDataStatus,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    prepare_german_benchmark_curve,
    snapshot_scenarios,
)


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
    Load and validate the official German benchmark-yield dataset.
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

    data = data.dropna(
        subset=[
            "observation_date",
            "tenor_years",
            "yield_percent",
        ]
    )

    if data.empty:
        raise ValueError(
            "German benchmark data contains no valid observations."
        )

    return data


def format_number(
    value: object,
    decimals: int = 2,
    suffix: str = "",
    prefix: str = "",
) -> str:
    """
    Format optional numeric dashboard values.
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
    Format a euro-denominated amount.
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
    Create a concise selector label.
    """
    return (
        f"{instrument.country.value} "
        f"{instrument.benchmark_tenor_years}Y | "
        f"{instrument.display_name} | "
        f"{instrument.isin}"
    )


def status_css_class(
    status: SnapshotDataStatus,
) -> str:
    """
    Map snapshot data status to shared dashboard styling.
    """
    if status == SnapshotDataStatus.OFFICIAL_DAILY:
        return "status-normal"

    if status == SnapshotDataStatus.DESK_INPUT:
        return "status-event"

    return "status-monitor"


def status_description(
    status: SnapshotDataStatus,
) -> str:
    """
    Explain the market-data status used in the valuation.
    """
    if status == SnapshotDataStatus.OFFICIAL_DAILY:
        return (
            "Official daily German benchmark yield. "
            "This is reference market data, not an executable quote."
        )

    if status == SnapshotDataStatus.DESK_INPUT:
        return (
            "Yield supplied by the dashboard user. "
            "RepoLens has not independently verified the quote."
        )

    return (
        "No permitted instrument-level market observation is available."
    )


def latest_benchmark_date(
    benchmark_data: pd.DataFrame,
) -> date:
    """
    Return the latest valid official benchmark observation date.
    """
    latest = pd.to_datetime(
        benchmark_data[
            "observation_date"
        ],
        errors="coerce",
    ).max()

    if pd.isna(
        latest
    ):
        raise ValueError(
            "German benchmark data has no valid observation date."
        )

    return pd.Timestamp(
        latest
    ).date()


def benchmark_yield_for_instrument(
    instrument: SovereignInstrument,
    benchmark_data: pd.DataFrame,
) -> float:
    """
    Return the matched latest German benchmark yield.
    """
    prepared_curve = prepare_german_benchmark_curve(
        benchmark_data
    )

    matched = prepared_curve.loc[
        prepared_curve[
            "tenor_years"
        ].eq(
            instrument.benchmark_tenor_years
        )
    ]

    if len(
        matched
    ) != 1:
        raise SovereignSnapshotValidationError(
            "RepoLens could not resolve exactly one German benchmark "
            f"for {instrument.benchmark_tenor_years}Y."
        )

    return float(
        matched.iloc[0][
            "yield_percent"
        ]
    )


def build_curve_chart(
    benchmark_data: pd.DataFrame,
    selected_tenor_years: int,
) -> go.Figure:
    """
    Plot the latest official German sovereign benchmark curve.
    """
    curve = prepare_german_benchmark_curve(
        benchmark_data
    ).copy()

    selected = curve.loc[
        curve[
            "tenor_years"
        ].eq(
            selected_tenor_years
        )
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=curve["tenor_years"],
            y=curve["yield_percent"],
            mode="lines+markers",
            name="German benchmark curve",
            customdata=curve[
                [
                    "observation_date",
                    "source_name",
                ]
            ],
            hovertemplate=(
                "%{x:.0f}Y benchmark<br>"
                "Yield: %{y:.3f}%<br>"
                "Date: %{customdata[0]|%d %b %Y}<br>"
                "Source: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["tenor_years"],
                y=selected["yield_percent"],
                mode="markers",
                name="Matched benchmark",
                marker={
                    "size": 14,
                    "symbol": "diamond",
                },
                hovertemplate=(
                    "Matched %{x:.0f}Y benchmark<br>"
                    "Yield: %{y:.3f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="Official German benchmark curve",
        xaxis_title="Benchmark tenor",
        yaxis_title="Yield (%)",
        xaxis={
            "tickmode": "array",
            "tickvals": curve[
                "tenor_years"
            ].tolist(),
            "ticksuffix": "Y",
        },
        hovermode="closest",
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


def build_scenario_chart(
    scenario_data: pd.DataFrame,
) -> go.Figure:
    """
    Plot position P&L under parallel yield shocks.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scenario_data[
                "yield_shock_bp"
            ],
            y=scenario_data[
                "position_pnl_eur"
            ],
            name="Position P&L",
            customdata=scenario_data[
                [
                    "shocked_yield_percent",
                    "shocked_clean_price",
                    "clean_price_change",
                ]
            ],
            hovertemplate=(
                "Yield shock: %{x:+.0f} bp<br>"
                "Position P&L: €%{y:,.0f}<br>"
                "Shocked yield: %{customdata[0]:.3f}%<br>"
                "Shocked clean price: %{customdata[1]:.4f}<br>"
                "Price change: %{customdata[2]:+.4f}"
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
        title="Parallel yield-shock scenario P&L",
        xaxis_title="Yield shock (bp)",
        yaxis_title="Position P&L (€)",
        hovermode="closest",
        height=440,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        showlegend=False,
    )

    return figure


def render_instrument_reference(
    instrument: SovereignInstrument,
) -> None:
    """
    Display the selected bond's official reference terms.
    """
    reference_data = pd.DataFrame(
        [
            {
                "Field": "ISIN",
                "Value": instrument.isin,
            },
            {
                "Field": "Issuer",
                "Value": instrument.issuer,
            },
            {
                "Field": "Security type",
                "Value": instrument.security_type.value,
            },
            {
                "Field": "Issue date",
                "Value": instrument.issue_date.strftime(
                    "%d %B %Y"
                ),
            },
            {
                "Field": "Maturity date",
                "Value": instrument.maturity_date.strftime(
                    "%d %B %Y"
                ),
            },
            {
                "Field": "Annual coupon",
                "Value": (
                    f"{instrument.annual_coupon_rate * 100.0:.3f}%"
                ),
            },
            {
                "Field": "Coupon frequency",
                "Value": (
                    "Annual"
                    if instrument.coupon_frequency == 1
                    else "Semi-annual"
                ),
            },
            {
                "Field": "Benchmark tenor",
                "Value": (
                    f"{instrument.benchmark_tenor_years}Y"
                ),
            },
            {
                "Field": "Terms source",
                "Value": instrument.source_name,
            },
            {
                "Field": "Terms checked",
                "Value": instrument.source_checked_date.strftime(
                    "%d %B %Y"
                ),
            },
        ]
    )

    st.dataframe(
        reference_data,
        hide_index=True,
        width="stretch",
        column_config={
            "Field": st.column_config.TextColumn(
                "Instrument field",
                width="medium",
            ),
            "Value": st.column_config.TextColumn(
                "Reference value",
                width="large",
            ),
        },
    )


def render_snapshot_metrics(
    snapshot: object,
) -> None:
    """
    Render valuation and risk metrics for the selected bond.
    """
    st.markdown(
        '<div class="section-label">Market valuation</div>',
        unsafe_allow_html=True,
    )

    valuation_columns = st.columns(
        5
    )

    valuation_columns[0].metric(
        "Yield to maturity",
        format_number(
            snapshot.yield_percent,
            decimals=3,
            suffix="%",
        ),
        border=True,
    )

    valuation_columns[1].metric(
        "Clean price",
        format_number(
            snapshot.clean_price,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[2].metric(
        "Dirty price",
        format_number(
            snapshot.dirty_price,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[3].metric(
        "Accrued interest",
        format_number(
            snapshot.accrued_interest,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[4].metric(
        "Germany spread",
        format_number(
            snapshot.spread_to_germany_bp,
            decimals=2,
            suffix=" bp",
        ),
        delta=(
            "Matched benchmark"
            if snapshot.country == "Germany"
            else (
                f"Germany "
                f"{snapshot.benchmark_tenor_years}Y: "
                f"{snapshot.german_benchmark_yield_percent:.3f}%"
            )
        ),
        delta_color="off",
        border=True,
    )

    st.markdown(
        '<div class="section-label">Interest-rate risk</div>',
        unsafe_allow_html=True,
    )

    risk_columns = st.columns(
        5
    )

    risk_columns[0].metric(
        "Modified duration",
        format_number(
            snapshot.modified_duration,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[1].metric(
        "Macaulay duration",
        format_number(
            snapshot.macaulay_duration,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[2].metric(
        "Convexity",
        format_number(
            snapshot.convexity,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[3].metric(
        "DV01 per €1mn",
        format_euro(
            snapshot.dv01_per_eur_1m,
            decimals=0,
        ),
        border=True,
    )

    risk_columns[4].metric(
        "Position DV01",
        format_euro(
            snapshot.position_dv01_eur,
            decimals=0,
        ),
        delta=(
            f"Face value "
            f"{format_euro(snapshot.position_notional_eur, 0)}"
        ),
        delta_color="off",
        border=True,
    )


def render_scenario_table(
    scenario_data: pd.DataFrame,
) -> None:
    """
    Render the detailed position scenario table.
    """
    display_data = scenario_data.copy()

    st.dataframe(
        display_data,
        hide_index=True,
        width="stretch",
        column_config={
            "isin": st.column_config.TextColumn(
                "ISIN"
            ),
            "yield_shock_bp": st.column_config.NumberColumn(
                "Yield shock",
                format="%+.0f bp",
            ),
            "shocked_yield_percent": (
                st.column_config.NumberColumn(
                    "Shocked yield",
                    format="%.3f%%",
                )
            ),
            "shocked_clean_price": (
                st.column_config.NumberColumn(
                    "Shocked clean price",
                    format="%.4f",
                )
            ),
            "clean_price_change": (
                st.column_config.NumberColumn(
                    "Price change",
                    format="%+.4f",
                )
            ),
            "position_pnl_eur": (
                st.column_config.NumberColumn(
                    "Position P&L",
                    format="€%,.0f",
                )
            ),
        },
    )


def main() -> None:
    """
    Render the RepoLens Sovereign Bond Terminal page.
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
        SovereignSnapshotValidationError,
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

    default_label = next(
        label
        for label, instrument in instruments_by_label.items()
        if (
            instrument.country == SovereignCountry.GERMANY
            and instrument.benchmark_tenor_years == 10
        )
    )

    with st.sidebar.expander(
        "Bond Terminal controls",
        expanded=True,
    ):
        selected_label = st.selectbox(
            "Instrument",
            options=list(
                instruments_by_label
            ),
            index=list(
                instruments_by_label
            ).index(
                default_label
            ),
            key="sovereign_terminal_instrument",
        )

        instrument = instruments_by_label[
            selected_label
        ]

        if instrument.country == SovereignCountry.GERMANY:
            market_mode = st.radio(
                "Market input",
                options=[
                    "Official benchmark",
                    "Desk-input yield",
                ],
                index=0,
                key="sovereign_terminal_market_mode",
            )
        else:
            market_mode = "Desk-input yield"

            st.info(
                "BTP valuation requires a desk-input yield. "
                "RepoLens does not manufacture an Italian quote."
            )

        earliest_settlement = instrument.issue_date

        latest_settlement = (
            instrument.maturity_date
            - timedelta(
                days=1
            )
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
            key=(
                "sovereign_terminal_settlement_"
                f"{instrument.isin}"
            ),
        )

        position_notional_eur = st.number_input(
            "Position face value (€)",
            min_value=0.0,
            value=10_000_000.0,
            step=1_000_000.0,
            format="%.0f",
            key="sovereign_terminal_notional",
        )

        matched_german_yield = (
            benchmark_yield_for_instrument(
                instrument=instrument,
                benchmark_data=benchmark_data,
            )
        )

        explicit_yield_input: SovereignYieldInput | None = None

        if market_mode == "Desk-input yield":
            default_yield = (
                matched_german_yield
                if instrument.country
                == SovereignCountry.GERMANY
                else matched_german_yield
                + 1.00
            )

            desk_yield_percent = st.number_input(
                "Yield to maturity (%)",
                min_value=-10.0,
                max_value=25.0,
                value=float(
                    default_yield
                ),
                step=0.01,
                format="%.3f",
                key=(
                    "sovereign_terminal_desk_yield_"
                    f"{instrument.isin}"
                ),
            )

            desk_observation_date = st.date_input(
                "Yield observation date",
                value=settlement_date,
                max_value=settlement_date,
                key=(
                    "sovereign_terminal_observation_"
                    f"{instrument.isin}"
                ),
            )

            desk_source_name = st.text_input(
                "Yield source description",
                value="Desk input",
                key=(
                    "sovereign_terminal_source_"
                    f"{instrument.isin}"
                ),
            )

            explicit_yield_input = SovereignYieldInput(
                isin=instrument.isin,
                yield_percent=float(
                    desk_yield_percent
                ),
                observation_date=desk_observation_date,
                source_name=desk_source_name,
            )

    try:
        snapshot = build_instrument_snapshot(
            instrument=instrument,
            german_curve=benchmark_data,
            settlement_date=settlement_date,
            position_notional_eur=float(
                position_notional_eur
            ),
            explicit_yield_input=explicit_yield_input,
        )

        scenario_data = snapshot_scenarios(
            instrument=instrument,
            settlement_date=settlement_date,
            yield_percent=snapshot.yield_percent,
            position_notional_eur=float(
                position_notional_eur
            ),
            yield_shocks_bp=DEFAULT_SCENARIO_SHOCKS_BP,
        )
    except SovereignSnapshotValidationError as error:
        st.error(
            "RepoLens could not value the selected instrument."
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
            Sovereign Bond Terminal
        </div>
        <div class="repolens-subtitle">
            Instrument-level Bund and BTP valuation,
            duration risk, DV01 and parallel-shock scenario analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-box {status_css_class(snapshot.data_status)}">
            <strong>{snapshot.data_status.value}</strong><br>
            {status_description(snapshot.data_status)}<br>
            Observation date: {
                snapshot.observation_date.strftime("%d %B %Y")
                if snapshot.observation_date is not None
                else "Unavailable"
            } · Settlement date: {
                snapshot.settlement_date.strftime("%d %B %Y")
            } · Source: {snapshot.source_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        snapshot.display_name
    )

    st.caption(
        f"{snapshot.isin} · "
        f"{snapshot.security_type} · "
        f"{snapshot.benchmark_tenor_years}Y benchmark segment"
    )

    render_snapshot_metrics(
        snapshot
    )

    st.divider()

    scenario_column, curve_column = st.columns(
        2
    )

    with scenario_column:
        st.plotly_chart(
            build_scenario_chart(
                scenario_data
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with curve_column:
        st.plotly_chart(
            build_curve_chart(
                benchmark_data=benchmark_data,
                selected_tenor_years=(
                    instrument.benchmark_tenor_years
                ),
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.divider()

    scenario_table_column, reference_column = st.columns(
        2
    )

    with scenario_table_column:
        st.subheader(
            "Position scenario matrix"
        )

        render_scenario_table(
            scenario_data
        )

        st.caption(
            "Scenario P&L is calculated from full bond repricing "
            "under parallel yield shocks. Positive figures represent "
            "gains for a long position."
        )

    with reference_column:
        st.subheader(
            "Instrument reference"
        )

        render_instrument_reference(
            instrument
        )

    st.divider()

    st.subheader(
        "Methodology and data classification"
    )

    methodology_left, methodology_right = st.columns(
        2
    )

    with methodology_left:
        st.markdown(
            """
            **Valuation**

            RepoLens discounts the remaining contractual coupon and
            principal cash flows using the selected yield to maturity.
            Clean price excludes accrued interest; dirty price includes it.

            **Risk**

            Modified duration, Macaulay duration, convexity and DV01 are
            derived from the instrument-level cash-flow schedule.
            """
        )

    with methodology_right:
        st.markdown(
            """
            **German observations**

            German yields are official daily benchmark observations from
            the Deutsche Bundesbank. They are reference data and should not
            be interpreted as executable bid or offer prices.

            **Italian observations**

            BTP analytics use an explicit user-supplied yield. RepoLens does
            not infer an Italian bond quote from the German curve.
            The displayed BTP–Bund spread is a derived research measure.
            """
        )


main()