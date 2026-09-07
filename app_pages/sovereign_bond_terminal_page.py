from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_components.sovereign_instrument_picker import (
    sovereign_instrument_picker,
)
from src.sovereign_instrument_catalog import (
    master_record_by_isin,
)
from src.sovereign_historical_context import (
    DEFAULT_HISTORICAL_WINDOWS,
    SovereignHistoricalContext,
    SovereignHistoricalObservation,
    build_sovereign_historical_context,
)
from src.sovereign_history_store import (
    SovereignHistoryStore,
    SovereignHistoryStoreValidationError,
)
from src.sovereign_instrument_pdf import (
    SovereignInstrumentPdfError,
    render_sovereign_instrument_report_pdf,
)
from src.sovereign_instrument_report import (
    InstrumentReportIdentity,
    InstrumentReportMarketSnapshot,
    InstrumentReportNarrative,
    InstrumentReportProvenance,
    build_sovereign_instrument_report_model,
)
from src.sovereign_instrument_master import (
    BenchmarkStatus,
)
from src.sovereign_instruments import (
    SovereignCountry,
    SovereignInstrument,
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

SOVEREIGN_HISTORY_PATH = Path(
    "data/market/sovereign_history.csv"
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




def load_persisted_instrument_history(
    instrument: SovereignInstrument,
) -> tuple[SovereignHistoricalObservation, ...]:
    """
    Load persisted sourced market history for one exact sovereign ISIN.

    Missing history is a valid state. RepoLens must not manufacture a historical
    series from today's snapshot merely to populate the dashboard.
    """
    store = SovereignHistoryStore(
        SOVEREIGN_HISTORY_PATH
    )

    return store.for_isin(
        instrument.isin
    )


def build_available_historical_context(
    instrument: SovereignInstrument,
) -> SovereignHistoricalContext | None:
    """
    Build 1W/1M/3M/6M/9M/1Y context when persisted observations exist.
    """
    observations = load_persisted_instrument_history(
        instrument
    )

    if not observations:
        return None

    try:
        return build_sovereign_historical_context(
            observations=observations,
            windows=DEFAULT_HISTORICAL_WINDOWS,
        )
    except Exception:
        # Partial datasets can legitimately lack observations in one of the
        # requested windows. Do not manufacture missing history.
        return None


def history_context_frame(
    context: SovereignHistoricalContext,
) -> pd.DataFrame:
    """
    Convert historical context into the broker-facing range table.
    """
    rows: list[dict[str, object]] = []

    for window in context.windows:
        yield_metric = window.yield_percent
        price_metric = window.price
        spread_metric = window.benchmark_spread_bp

        rows.append(
            {
                "Window": window.window.label,
                "Observations": window.observation_count,
                "Yield low (%)": (
                    yield_metric.low
                    if yield_metric is not None
                    else None
                ),
                "Yield high (%)": (
                    yield_metric.high
                    if yield_metric is not None
                    else None
                ),
                "Yield median (%)": (
                    yield_metric.median
                    if yield_metric is not None
                    else None
                ),
                "Current yield (%)": (
                    yield_metric.current
                    if yield_metric is not None
                    else None
                ),
                "Yield percentile": (
                    yield_metric.percentile
                    if yield_metric is not None
                    else None
                ),
                "Yield move": (
                    (
                        yield_metric.change_from_window_start
                        * 100.0
                    )
                    if yield_metric is not None
                    else None
                ),
                "Price low": (
                    price_metric.low
                    if price_metric is not None
                    else None
                ),
                "Price high": (
                    price_metric.high
                    if price_metric is not None
                    else None
                ),
                "Current price": (
                    price_metric.current
                    if price_metric is not None
                    else None
                ),
                "Spread current (bp)": (
                    spread_metric.current
                    if spread_metric is not None
                    else None
                ),
                "Spread move (bp)": (
                    spread_metric.change_from_window_start
                    if spread_metric is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_history_chart(
    observations: tuple[SovereignHistoricalObservation, ...],
    current_yield_percent: float,
) -> go.Figure:
    """
    Plot sourced instrument-level yield history when available.
    """
    rows = [
        {
            "observation_date": observation.observation_date,
            "yield_percent": observation.yield_percent,
            "source_name": observation.source_name,
            "data_status": observation.data_status,
        }
        for observation in observations
        if observation.yield_percent is not None
    ]

    frame = pd.DataFrame(
        rows
    )

    figure = go.Figure()

    if not frame.empty:
        frame = frame.sort_values(
            "observation_date"
        )

        figure.add_trace(
            go.Scatter(
                x=frame["observation_date"],
                y=frame["yield_percent"],
                mode="lines+markers",
                name="Sourced yield history",
                customdata=frame[
                    [
                        "source_name",
                        "data_status",
                    ]
                ],
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Yield: %{y:.3f}%<br>"
                    "Source: %{customdata[0]}<br>"
                    "Status: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=float(
            current_yield_percent
        ),
        line_width=1,
        line_dash="dash",
        annotation_text="Current valuation yield",
        annotation_position="top left",
    )

    figure.update_layout(
        title="Instrument yield history",
        xaxis_title="Observation date",
        yaxis_title="Yield (%)",
        height=410,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        showlegend=False,
    )

    return figure


def deterministic_instrument_narrative(
    *,
    snapshot: object,
    context: SovereignHistoricalContext | None,
) -> InstrumentReportNarrative:
    """
    Build concise evidence-led prose without an opaque prediction layer.
    """
    observation = (
        f"{snapshot.display_name} is valued at "
        f"{snapshot.yield_percent:.3f}% yield with a clean price of "
        f"{snapshot.clean_price:.4f}."
    )

    if pd.notna(
        snapshot.spread_to_germany_bp
    ):
        observation += (
            f" The exact-tenor Germany spread is "
            f"{snapshot.spread_to_germany_bp:.1f} bp."
        )

    if context is None:
        return InstrumentReportNarrative(
            what_happened=observation,
            historical_context=(
                "No persisted sourced instrument-level history is currently "
                "available for the selected ISIN, so RepoLens does not display "
                "or infer historical ranges."
            ),
            repo_context=(
                "Repo intelligence is sourced separately from saved "
                "specific-repo and GC observations."
            ),
            event_risk=(
                "No verified instrument-event dataset is loaded on this page."
            ),
            interpretation=(
                "Current valuation and risk analytics remain available, but "
                "historical conclusions require sourced observations."
            ),
            uncertainty=(
                "Desk inputs, official references and RepoLens-derived "
                "analytics are classified separately."
            ),
        )

    longest = context.windows[-1]
    yield_metric = longest.yield_percent

    if yield_metric is None:
        historical_text = (
            "Persisted observations exist, but the selected historical series "
            "does not contain a usable yield history."
        )
    else:
        distance_bp = (
            yield_metric.distance_from_median
            * 100.0
        )
        side = (
            "above"
            if distance_bp >= 0.0
            else "below"
        )

        historical_text = (
            f"Over the {longest.window.label} window, the current yield is "
            f"{abs(distance_bp):.1f} bp {side} the median and sits at the "
            f"{yield_metric.percentile:.1f} empirical percentile of "
            f"{yield_metric.observation_count} sourced observations."
        )

    return InstrumentReportNarrative(
        what_happened=observation,
        historical_context=historical_text,
        repo_context=(
            "RepoLens keeps cash-market history and collateral funding "
            "economics separate so specific-versus-GC inputs are not confused "
            "with cash-bond yields."
        ),
        event_risk=(
            "No verified instrument-event dataset is loaded on this page yet; "
            "RepoLens therefore does not manufacture an event warning."
        ),
        interpretation=(
            "Historical statistics are descriptive context rather than a "
            "directional trade signal."
        ),
        uncertainty=(
            "Percentiles and ranges depend on the available sourced sample and "
            "should be interpreted alongside data status and observation count."
        ),
    )


def render_historical_context(
    *,
    snapshot: object,
    instrument: SovereignInstrument,
    context: SovereignHistoricalContext | None,
    observations: tuple[SovereignHistoricalObservation, ...],
) -> None:
    """
    Render broker-facing historical trading context.
    """
    st.markdown(
        '<div class="section-label">Historical trading context</div>',
        unsafe_allow_html=True,
    )

    if context is None:
        st.info(
            "No persisted sourced instrument-level history is available for "
            f"{instrument.isin}. RepoLens will not manufacture 1W / 1M / 3M / "
            "6M / 9M / 1Y ranges from benchmark or desk data."
        )
        return

    frame = history_context_frame(
        context
    )

    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Window": st.column_config.TextColumn(
                "Window"
            ),
            "Observations": st.column_config.NumberColumn(
                "Obs",
                format="%d",
            ),
            "Yield low (%)": st.column_config.NumberColumn(
                "Yield low",
                format="%.3f%%",
            ),
            "Yield high (%)": st.column_config.NumberColumn(
                "Yield high",
                format="%.3f%%",
            ),
            "Yield median (%)": st.column_config.NumberColumn(
                "Median",
                format="%.3f%%",
            ),
            "Current yield (%)": st.column_config.NumberColumn(
                "Current",
                format="%.3f%%",
            ),
            "Yield percentile": st.column_config.NumberColumn(
                "Percentile",
                format="%.1f",
            ),
            "Yield move": st.column_config.NumberColumn(
                "Move",
                format="%+.1f bp",
            ),
            "Price low": st.column_config.NumberColumn(
                "Price low",
                format="%.4f",
            ),
            "Price high": st.column_config.NumberColumn(
                "Price high",
                format="%.4f",
            ),
            "Current price": st.column_config.NumberColumn(
                "Current price",
                format="%.4f",
            ),
            "Spread current (bp)": st.column_config.NumberColumn(
                "Spread",
                format="%+.1f bp",
            ),
            "Spread move (bp)": st.column_config.NumberColumn(
                "Spread move",
                format="%+.1f bp",
            ),
        },
    )

    st.plotly_chart(
        build_history_chart(
            observations=observations,
            current_yield_percent=snapshot.yield_percent,
        ),
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )

    st.caption(
        f"History source status: {context.latest_data_status} · "
        f"{context.latest_source_name} · Latest observation "
        f"{context.latest_observation_date.strftime('%d %B %Y')}. "
        "Percentiles are empirical and sample counts are shown explicitly."
    )


def render_repo_intelligence_placeholder(
    instrument: SovereignInstrument,
) -> None:
    """
    Reserve the instrument-level repo workflow without inventing repo quotes.
    """
    st.markdown(
        '<div class="section-label">Repo intelligence</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Instrument-level repo intelligence will appear here when sourced "
        f"specific-repo and matched GC observations exist for {instrument.isin}. "
        "RepoLens does not manufacture executable repo rates."
    )

    st.caption(
        "Planned outputs: current specialness, own-history percentile, 1D/1W "
        "change, financing advantage versus GC, repo-adjusted carry and "
        "term-structure context."
    )


def render_event_intelligence_placeholder(
    instrument: SovereignInstrument,
) -> None:
    """
    Surface an explicit empty event state until verified event feeds are wired.
    """
    st.markdown(
        '<div class="section-label">Event intelligence</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "No verified instrument-event feed is loaded for this page yet. "
        "RepoLens therefore does not display auction, coupon, redemption or "
        "central-bank countdowns until their source and date are verified."
    )

    st.caption(
        f"Selected instrument: {instrument.isin}. Event relevance will be "
        "instrument-, country- and currency-aware."
    )


def render_narrative_and_report(
    *,
    instrument: SovereignInstrument,
    snapshot: object,
    context: SovereignHistoricalContext | None,
) -> None:
    """
    Render deterministic narrative and an auditable PDF export when history exists.
    """
    narrative = deterministic_instrument_narrative(
        snapshot=snapshot,
        context=context,
    )

    st.markdown(
        '<div class="section-label">RepoLens narrative</div>',
        unsafe_allow_html=True,
    )

    narrative_left, narrative_right = st.columns(
        [
            3,
            1,
        ]
    )

    with narrative_left:
        st.markdown(
            f"""
            **What happened**

            {narrative.what_happened}

            **Historical context**

            {narrative.historical_context}

            **Repo context**

            {narrative.repo_context}

            **Event risk**

            {narrative.event_risk}

            **Interpretation**

            {narrative.interpretation}

            **Uncertainty**

            {narrative.uncertainty}
            """
        )

    with narrative_right:
        st.markdown(
            "**Instrument report**"
        )

        if context is None:
            st.button(
                "Generate Report",
                disabled=True,
                width="stretch",
                help=(
                    "A sourced historical series is required before RepoLens "
                    "can produce the full instrument report."
                ),
            )

            st.caption(
                "Report export unlocks when sourced instrument history exists."
            )
            return

        market_source = (
            snapshot.source_name
            or "RepoLens market input"
        )

        market_status = (
            snapshot.data_status.value
            if hasattr(
                snapshot.data_status,
                "value",
            )
            else str(
                snapshot.data_status
            )
        )

        report_model = build_sovereign_instrument_report_model(
            identity=InstrumentReportIdentity(
                isin=instrument.isin,
                display_name=snapshot.display_name,
                country=instrument.country.value,
                currency="EUR",
                maturity_date=instrument.maturity_date,
                coupon_percent=(
                    instrument.annual_coupon_rate
                    * 100.0
                ),
            ),
            market=InstrumentReportMarketSnapshot(
                as_of_date=context.latest_observation_date,
                source_name=context.latest_source_name,
                data_status=context.latest_data_status,
                price_per_100=(
                    context.windows[-1].price.current
                    if context.windows[-1].price is not None
                    else None
                ),
                yield_percent=(
                    context.windows[-1].yield_percent.current
                    if context.windows[-1].yield_percent is not None
                    else None
                ),
                benchmark_spread_bp=(
                    context.windows[-1].benchmark_spread_bp.current
                    if context.windows[-1].benchmark_spread_bp is not None
                    else None
                ),
                benchmark_name=context.benchmark_name,
            ),
            historical_context=context,
            narrative=narrative,
            generated_at=datetime.now(),
            repo=None,
            events=(),
            provenance=(
                InstrumentReportProvenance(
                    label="Historical market",
                    source_name=context.latest_source_name,
                    data_status=context.latest_data_status,
                    as_of=context.latest_observation_date.isoformat(),
                    notes=(
                        "Persisted sourced instrument-level observations."
                    ),
                ),
                InstrumentReportProvenance(
                    label="Current terminal valuation",
                    source_name=market_source,
                    data_status=market_status,
                    as_of=(
                        snapshot.observation_date.isoformat()
                        if snapshot.observation_date is not None
                        else snapshot.settlement_date.isoformat()
                    ),
                    notes=(
                        "Displayed terminal valuation may differ from the "
                        "latest persisted historical observation."
                    ),
                ),
            ),
        )

        try:
            pdf_bytes = render_sovereign_instrument_report_pdf(
                report_model
            )
        except SovereignInstrumentPdfError as error:
            st.error(
                "RepoLens could not generate the instrument report."
            )
            st.code(
                str(
                    error
                )
            )
            return

        safe_isin = instrument.isin.replace(
            "/",
            "_",
        )

        st.download_button(
            "Generate Report",
            data=pdf_bytes,
            file_name=(
                f"RepoLens_{safe_isin}_Instrument_Report.pdf"
            ),
            mime="application/pdf",
            width="stretch",
        )

        st.caption(
            "PDF is generated from the same structured report model used by "
            "the instrument intelligence workflow."
        )


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

    with st.sidebar.expander(
        "Bond Terminal controls",
        expanded=True,
    ):
        instrument = sovereign_instrument_picker()

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

        try:
            historical_observations = load_persisted_instrument_history(
                instrument
            )
            historical_context = build_available_historical_context(
                instrument
            )
        except SovereignHistoryStoreValidationError as error:
            st.warning(
                "RepoLens found persisted sovereign history but could not "
                "validate it. Historical context is disabled for this instrument."
            )
            st.code(
                str(
                    error
                )
            )
            historical_observations = ()
            historical_context = None

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
            Sovereign valuation, historical market context,
            repo and event intelligence, duration risk, DV01
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

    render_historical_context(
        snapshot=snapshot,
        instrument=instrument,
        context=historical_context,
        observations=historical_observations,
    )

    st.divider()

    intelligence_left, intelligence_right = st.columns(
        2
    )

    with intelligence_left:
        render_repo_intelligence_placeholder(
            instrument
        )

    with intelligence_right:
        render_event_intelligence_placeholder(
            instrument
        )

    st.divider()

    render_narrative_and_report(
        instrument=instrument,
        snapshot=snapshot,
        context=historical_context,
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


            **Historical context and reports**

            Historical windows are calculated only from persisted sourced
            instrument-level observations. Missing history is left unavailable
            rather than inferred from benchmarks or today's desk input.

            PDF reports use the same structured RepoLens report model as the
            instrument intelligence workflow and preserve source/status metadata.
            """
        )


main()