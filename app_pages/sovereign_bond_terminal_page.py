from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sovereign_instrument_catalog import (
    all_instruments,
    master_record_by_isin,
)
from src.sovereign_instrument_master import (
    BenchmarkStatus,
)
from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
    SovereignSecurityType,
)
from src.sovereign_snapshot import (
    DEFAULT_SCENARIO_SHOCKS_BP,
    SnapshotDataStatus,
    SovereignSnapshotValidationError,
    SovereignYieldInput,
    build_instrument_snapshot,
    optional_german_benchmark_for_tenor,
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

    data[
        "observation_date"
    ] = pd.to_datetime(
        data[
            "observation_date"
        ],
        errors="coerce",
    )

    data[
        "tenor_years"
    ] = pd.to_numeric(
        data[
            "tenor_years"
        ],
        errors="coerce",
    )

    data[
        "yield_percent"
    ] = pd.to_numeric(
        data[
            "yield_percent"
        ],
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
    Format an optional numeric value.
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
    Create the bond selector label.
    """
    return (
        f"{instrument.country.value} | "
        f"{instrument.benchmark_tenor_years}Y | "
        f"{instrument.display_name} | "
        f"{instrument.isin}"
    )


def latest_benchmark_date(
    benchmark_data: pd.DataFrame,
) -> date:
    """
    Return the latest valid German benchmark observation date.
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


def exact_german_benchmark(
    instrument: SovereignInstrument,
    benchmark_data: pd.DataFrame,
) -> pd.Series | None:
    """
    Return the exact German tenor observation when available.

    RepoLens does not interpolate missing sovereign benchmark tenors.
    """
    prepared_curve = prepare_german_benchmark_curve(
        benchmark_data
    )

    return optional_german_benchmark_for_tenor(
        prepared_curve=prepared_curve,
        tenor_years=(
            instrument.benchmark_tenor_years
        ),
    )


def status_css_class(
    status: SnapshotDataStatus,
) -> str:
    """
    Map market-data status to shared page styling.
    """
    if (
        status
        == SnapshotDataStatus.OFFICIAL_DAILY
    ):
        return "status-normal"

    if (
        status
        == SnapshotDataStatus.DESK_INPUT
    ):
        return "status-event"

    return "status-monitor"


def status_description(
    status: SnapshotDataStatus,
) -> str:
    """
    Explain the origin of the valuation yield.
    """
    if (
        status
        == SnapshotDataStatus.OFFICIAL_DAILY
    ):
        return (
            "Official daily German benchmark yield. "
            "Reference market data, not an executable quote."
        )

    if (
        status
        == SnapshotDataStatus.DESK_INPUT
    ):
        return (
            "Instrument-level yield supplied by the user. "
            "RepoLens has not independently verified the quote."
        )

    return (
        "No permitted instrument-level market observation is available."
    )


def benchmark_status_label(
    status: BenchmarkStatus,
) -> str:
    """
    Convert internal benchmark status into readable text.
    """
    mapping = {
        BenchmarkStatus.PRIMARY_BENCHMARK: (
            "Primary benchmark"
        ),
        BenchmarkStatus.REFERENCE_BOND: (
            "Reference bond"
        ),
        BenchmarkStatus.OFF_THE_RUN: (
            "Off-the-run"
        ),
    }

    return mapping[
        status
    ]


def coupon_frequency_label(
    frequency: int,
) -> str:
    """
    Convert coupon frequency into a readable label.
    """
    mapping = {
        1: "Annual",
        2: "Semi-annual",
        4: "Quarterly",
    }

    return mapping.get(
        frequency,
        f"{frequency} payments/year",
    )


def build_curve_chart(
    benchmark_data: pd.DataFrame,
    selected_tenor_years: int,
) -> go.Figure:
    """
    Plot the available official German benchmark curve.
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
            x=curve[
                "tenor_years"
            ],
            y=curve[
                "yield_percent"
            ],
            mode="lines+markers",
            name="German official benchmarks",
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
                x=selected[
                    "tenor_years"
                ],
                y=selected[
                    "yield_percent"
                ],
                mode="markers",
                name="Exact matched tenor",
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
            "tickvals": (
                curve[
                    "tenor_years"
                ]
                .tolist()
            ),
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
    Plot long-position P&L under parallel yield shocks.
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


def render_snapshot_metrics(
    snapshot: object,
) -> None:
    """
    Render valuation and interest-rate-risk metrics.
    """
    st.markdown(
        '<div class="section-label">Market valuation</div>',
        unsafe_allow_html=True,
    )

    valuation_columns = st.columns(
        5
    )

    valuation_columns[
        0
    ].metric(
        "Yield to maturity",
        format_number(
            snapshot.yield_percent,
            decimals=3,
            suffix="%",
        ),
        border=True,
    )

    valuation_columns[
        1
    ].metric(
        "Clean price",
        format_number(
            snapshot.clean_price,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[
        2
    ].metric(
        "Dirty price",
        format_number(
            snapshot.dirty_price,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[
        3
    ].metric(
        "Accrued interest",
        format_number(
            snapshot.accrued_interest,
            decimals=4,
        ),
        border=True,
    )

    valuation_columns[
        4
    ].metric(
        "Germany spread",
        format_number(
            snapshot.spread_to_germany_bp,
            decimals=2,
            suffix=" bp",
        ),
        delta=(
            "Exact official benchmark"
            if pd.notna(
                snapshot
                .german_benchmark_yield_percent
            )
            else "Exact benchmark unavailable"
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

    risk_columns[
        0
    ].metric(
        "Modified duration",
        format_number(
            snapshot.modified_duration,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[
        1
    ].metric(
        "Macaulay duration",
        format_number(
            snapshot.macaulay_duration,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[
        2
    ].metric(
        "Convexity",
        format_number(
            snapshot.convexity,
            decimals=4,
        ),
        border=True,
    )

    risk_columns[
        3
    ].metric(
        "DV01 per €1mn",
        format_euro(
            snapshot.dv01_per_eur_1m,
            decimals=0,
        ),
        border=True,
    )

    risk_columns[
        4
    ].metric(
        "Position DV01",
        format_euro(
            snapshot.position_dv01_eur,
            decimals=0,
        ),
        delta=(
            "Face value "
            f"{format_euro(snapshot.position_notional_eur)}"
        ),
        delta_color="off",
        border=True,
    )


def render_scenario_table(
    scenario_data: pd.DataFrame,
) -> None:
    """
    Display detailed parallel-yield scenarios.
    """
    st.dataframe(
        scenario_data,
        hide_index=True,
        width="stretch",
        column_config={
            "isin": st.column_config.TextColumn(
                "ISIN"
            ),
            "yield_shock_bp": (
                st.column_config.NumberColumn(
                    "Yield shock",
                    format="%+.0f bp",
                )
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


def render_instrument_reference(
    instrument: SovereignInstrument,
) -> None:
    """
    Display contractual and catalogue reference data.
    """
    record = master_record_by_isin(
        instrument.isin
    )

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
                "Field": "Country",
                "Value": instrument.country.value,
            },
            {
                "Field": "Security type",
                "Value": instrument.security_type.value,
            },
            {
                "Field": "Issue date",
                "Value": (
                    instrument.issue_date.strftime(
                        "%d %B %Y"
                    )
                ),
            },
            {
                "Field": "Maturity date",
                "Value": (
                    instrument.maturity_date.strftime(
                        "%d %B %Y"
                    )
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
                    coupon_frequency_label(
                        instrument.coupon_frequency
                    )
                ),
            },
            {
                "Field": "Benchmark sector",
                "Value": (
                    f"{instrument.benchmark_tenor_years}Y"
                ),
            },
            {
                "Field": "Original maturity",
                "Value": (
                    f"{record.original_maturity_years}Y"
                ),
            },
            {
                "Field": "Catalogue classification",
                "Value": (
                    benchmark_status_label(
                        record.benchmark_status
                    )
                ),
            },
            {
                "Field": "Primary benchmark",
                "Value": (
                    "Yes"
                    if record.is_primary_benchmark
                    else "No"
                ),
            },
            {
                "Field": "Terms source",
                "Value": instrument.source_name,
            },
            {
                "Field": "Terms checked",
                "Value": (
                    instrument
                    .source_checked_date
                    .strftime(
                        "%d %B %Y"
                    )
                ),
            },
            {
                "Field": "Reference status",
                "Value": (
                    instrument.data_status.value
                ),
            },
        ]
    )

    st.dataframe(
        reference_data,
        hide_index=True,
        width="stretch",
        column_config={
            "Field": (
                st.column_config.TextColumn(
                    "Instrument field",
                    width="medium",
                )
            ),
            "Value": (
                st.column_config.TextColumn(
                    "Reference value",
                    width="large",
                )
            ),
        },
    )


def filtered_instrument_universe(
    country_filter: str,
    security_filter: str,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Filter the complete sovereign catalogue.
    """
    instruments = all_instruments()

    selected = tuple(
        instrument
        for instrument in instruments
        if (
            (
                country_filter == "All"
                or instrument.country.value
                == country_filter
            )
            and (
                security_filter == "All"
                or instrument.security_type.value
                == security_filter
            )
        )
    )

    return selected


def main() -> None:
    """
    Render the RepoLens Sovereign Bond Terminal.
    """
    try:
        benchmark_data = (
            load_german_benchmark_data(
                str(
                    GERMAN_BENCHMARK_PATH
                )
            )
        )

        official_observation_date = (
            latest_benchmark_date(
                benchmark_data
            )
        )

        complete_catalogue = (
            all_instruments()
        )

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
        SovereignSnapshotValidationError,
        RuntimeError,
    ) as error:
        st.error(
            "RepoLens could not initialise the Sovereign Bond Terminal."
        )

        st.code(
            str(
                error
            )
        )

        st.stop()

    country_options = [
        "All",
        *[
            country.value
            for country
            in SovereignCountry
        ],
    ]

    security_options = [
        "All",
        *sorted(
            {
                instrument
                .security_type
                .value
                for instrument
                in complete_catalogue
            }
        ),
    ]

    with st.sidebar.expander(
        "Bond Terminal controls",
        expanded=True,
    ):
        country_filter = st.selectbox(
            "Country",
            options=country_options,
            index=0,
            key="sovereign_terminal_country_filter",
        )

        security_filter = st.selectbox(
            "Security type",
            options=security_options,
            index=0,
            key="sovereign_terminal_security_filter",
        )

        filtered_instruments = (
            filtered_instrument_universe(
                country_filter=country_filter,
                security_filter=security_filter,
            )
        )

        if not filtered_instruments:
            st.warning(
                "No bonds match the selected filters."
            )

            st.stop()

        instruments_by_label = {
            instrument_label(
                instrument
            ): instrument
            for instrument
            in filtered_instruments
        }

        labels = list(
            instruments_by_label
        )

        default_index = 0

        for (
            index,
            label,
        ) in enumerate(
            labels
        ):
            candidate = (
                instruments_by_label[
                    label
                ]
            )

            if (
                candidate.country
                == SovereignCountry.GERMANY
                and candidate
                .benchmark_tenor_years
                == 10
            ):
                default_index = index
                break

        selected_label = st.selectbox(
            "Instrument",
            options=labels,
            index=default_index,
            key="sovereign_terminal_instrument",
        )

        instrument = (
            instruments_by_label[
                selected_label
            ]
        )

        record = master_record_by_isin(
            instrument.isin
        )

        benchmark = exact_german_benchmark(
            instrument=instrument,
            benchmark_data=benchmark_data,
        )

        has_exact_german_benchmark = (
            benchmark is not None
        )

        st.caption(
            f"{benchmark_status_label(record.benchmark_status)} · "
            f"{instrument.benchmark_tenor_years}Y sector"
        )

        if (
            instrument.country
            == SovereignCountry.GERMANY
            and has_exact_german_benchmark
        ):
            market_mode = st.radio(
                "Market input",
                options=[
                    "Official benchmark",
                    "Desk-input yield",
                ],
                index=0,
                key=(
                    "sovereign_terminal_market_mode_"
                    f"{instrument.isin}"
                ),
            )

        elif (
            instrument.country
            == SovereignCountry.GERMANY
        ):
            market_mode = (
                "Desk-input yield"
            )

            st.warning(
                "No exact official German "
                f"{instrument.benchmark_tenor_years}Y "
                "benchmark observation is available. "
                "An instrument-level desk yield is required. "
                "RepoLens does not interpolate the German curve."
            )

        else:
            market_mode = (
                "Desk-input yield"
            )

            if has_exact_german_benchmark:
                st.info(
                    "BTP valuation requires an instrument-level "
                    "desk yield. The exact German benchmark is used "
                    "only for the derived sovereign spread."
                )

            else:
                st.warning(
                    "BTP valuation requires an instrument-level "
                    "desk yield. No exact permitted German "
                    f"{instrument.benchmark_tenor_years}Y benchmark "
                    "is available, so RepoLens will not display a "
                    "BTP–Bund spread for this instrument."
                )

        earliest_settlement = (
            instrument.issue_date
        )

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

        position_notional_eur = (
            st.number_input(
                "Position face value (€)",
                min_value=0.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="sovereign_terminal_notional",
            )
        )

        explicit_yield_input: (
            SovereignYieldInput
            | None
        ) = None

        if (
            market_mode
            == "Desk-input yield"
        ):
            desk_yield_percent = (
                st.number_input(
                    "Yield to maturity (%)",
                    min_value=-10.0,
                    max_value=25.0,
                    value=None,
                    step=0.01,
                    format="%.3f",
                    placeholder=(
                        "Enter instrument yield"
                    ),
                    key=(
                        "sovereign_terminal_desk_yield_"
                        f"{instrument.isin}"
                    ),
                )
            )

            desk_observation_date = (
                st.date_input(
                    "Yield observation date",
                    value=(
                        settlement_date
                    ),
                    max_value=(
                        settlement_date
                    ),
                    key=(
                        "sovereign_terminal_observation_"
                        f"{instrument.isin}"
                    ),
                )
            )

            desk_source_name = (
                st.text_input(
                    "Yield source description",
                    value="Desk input",
                    key=(
                        "sovereign_terminal_source_"
                        f"{instrument.isin}"
                    ),
                )
            )

            if (
                desk_yield_percent
                is None
            ):
                st.info(
                    "Enter the instrument's current yield to maturity "
                    "to calculate price, accrued interest, DV01, "
                    "duration and scenarios."
                )

                st.stop()

            explicit_yield_input = (
                SovereignYieldInput(
                    isin=instrument.isin,
                    yield_percent=float(
                        desk_yield_percent
                    ),
                    observation_date=(
                        desk_observation_date
                    ),
                    source_name=(
                        desk_source_name
                    ),
                )
            )

    try:
        snapshot = build_instrument_snapshot(
            instrument=instrument,
            german_curve=benchmark_data,
            settlement_date=settlement_date,
            position_notional_eur=float(
                position_notional_eur
            ),
            explicit_yield_input=(
                explicit_yield_input
            ),
        )

        scenario_data = snapshot_scenarios(
            instrument=instrument,
            settlement_date=settlement_date,
            yield_percent=(
                snapshot.yield_percent
            ),
            position_notional_eur=float(
                position_notional_eur
            ),
            yield_shocks_bp=(
                DEFAULT_SCENARIO_SHOCKS_BP
            ),
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
            Multi-instrument Bund and BTP valuation,
            reference-data lineage, duration risk, DV01
            and full-repricing scenario analysis.
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
            } ·
            Settlement date: {
                snapshot.settlement_date.strftime("%d %B %Y")
            } ·
            Yield source: {snapshot.source_name}
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
        f"{snapshot.benchmark_tenor_years}Y sector · "
        f"{benchmark_status_label(record.benchmark_status)}"
    )

    if pd.isna(
        snapshot
        .german_benchmark_yield_percent
    ):
        st.warning(
            "No exact permitted German benchmark observation exists "
            f"for the {snapshot.benchmark_tenor_years}Y sector. "
            "The bond valuation remains available from the selected "
            "instrument yield, but Germany spread is intentionally N/A."
        )

    render_snapshot_metrics(
        snapshot
    )

    st.divider()

    scenario_column, curve_column = (
        st.columns(
            2
        )
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
                    instrument
                    .benchmark_tenor_years
                ),
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.divider()

    scenario_table_column, reference_column = (
        st.columns(
            2
        )
    )

    with scenario_table_column:
        st.subheader(
            "Position scenario matrix"
        )

        render_scenario_table(
            scenario_data
        )

        st.caption(
            "Scenario P&L uses full bond repricing under parallel "
            "yield shocks. Positive figures represent gains for a "
            "long position."
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

    methodology_left, methodology_right = (
        st.columns(
            2
        )
    )

    with methodology_left:
        st.markdown(
            """
            **Valuation**

            RepoLens discounts the remaining contractual coupon and
            principal cash flows using the selected yield to maturity.

            Clean price excludes accrued interest. Dirty price includes
            accrued interest.

            **Risk**

            Modified duration, Macaulay duration, convexity and DV01 are
            calculated from the instrument-level cash-flow schedule.

            Parallel-shock P&L uses full repricing rather than a
            duration-only approximation.
            """
        )

    with methodology_right:
        st.markdown(
            """
            **German market observations**

            Available German yields are official daily benchmark
            observations. They are reference data, not executable
            bid/offer quotes.

            RepoLens requires an exact tenor match. It does not silently
            interpolate missing German benchmark sectors.

            **Italian market observations**

            BTP valuation requires an explicit instrument-level yield.
            RepoLens does not infer an Italian quote from Germany.

            When an exact German tenor exists, the displayed BTP–Bund
            spread is a derived research measure. When it does not,
            the spread is reported as unavailable.
            """
        )


main()