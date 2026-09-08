from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sovereign_history_store import (
    SovereignHistoryStore,
    SovereignHistoryStoreValidationError,
)
from src.sovereign_desk_universe import (
    DeskSovereignInstrument,
    load_desk_sovereign_universe,
    universe_counts_by_country,
    universe_country_names,
)
from src.sovereign_morning_commentary import (
    build_sovereign_morning_commentary,
)
from src.sovereign_morning_pdf import (
    build_sovereign_morning_pdf,
)


SOVEREIGN_HISTORY_PATH = Path(
    "data/market/sovereign_history.csv"
)

SOVEREIGN_SPREAD_PATH = Path(
    "data/market/sovereign_spreads.csv"
)

EUROPEAN_BENCHMARK_YIELD_PATH = Path(
    "data/market/european_sovereign_benchmark_yields.csv"
)

WINDOWS = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "6M": 182,
    "9M": 274,
    "1Y": 365,
}

STATUS_DISPLAY = {
    "DELAYED_PUBLIC_REFERENCE": "PUBLIC REFERENCE",
    "PUBLIC_REFERENCE": "PUBLIC REFERENCE",
    "OFFICIAL_REFERENCE": "OFFICIAL",
    "OFFICIAL_DAILY": "OFFICIAL",
    "DESK_INPUT": "DESK INPUT",
    "BROKER_INPUT": "BROKER INPUT",
    "REPOLENS_DERIVED": "REPOLENS DERIVED",
}


def display_status(
    value: object,
) -> str:
    text = str(
        value
    ).strip()

    return STATUS_DISPLAY.get(
        text,
        text.replace(
            "_",
            " ",
        ),
    )


def observations_frame(
    store: SovereignHistoryStore,
) -> pd.DataFrame:
    observations = store.load()

    rows = [
        {
            "observation_date": observation.observation_date,
            "isin": observation.isin,
            "source_name": observation.source_name,
            "data_status": observation.data_status,
            "price_per_100": observation.price_per_100,
            "yield_percent": observation.yield_percent,
            "benchmark_spread_bp": observation.benchmark_spread_bp,
            "benchmark_name": observation.benchmark_name,
        }
        for observation in observations
    ]

    return pd.DataFrame(
        rows
    )


def load_spread_history(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "spread_name",
                "observation_date",
                "observation_time",
                "spread_bp",
                "change_percent",
                "left_yield_percent",
                "right_yield_percent",
                "left_label",
                "right_label",
                "source_name",
                "data_status",
            ]
        )

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        "spread_name",
        "observation_date",
        "observation_time",
        "spread_bp",
        "change_percent",
        "left_yield_percent",
        "right_yield_percent",
        "left_label",
        "right_label",
        "source_name",
        "data_status",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Sovereign spread history is missing required columns: "
            f"{sorted(missing)}"
        )

    frame[
        "observation_date"
    ] = pd.to_datetime(
        frame[
            "observation_date"
        ],
        errors="coerce",
    ).dt.date

    for column in (
        "spread_bp",
        "change_percent",
        "left_yield_percent",
        "right_yield_percent",
    ):
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "spread_name",
            "observation_date",
            "spread_bp",
            "left_yield_percent",
            "right_yield_percent",
        ]
    )

    return frame


def load_european_benchmark_yields(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "country_code",
                "country_name",
                "curve_name",
                "tenor_years",
                "observation_date",
                "yield_percent",
                "index_name",
                "index_isin",
                "symbol",
                "source_name",
                "data_status",
            ]
        )

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        "country_code",
        "country_name",
        "curve_name",
        "tenor_years",
        "observation_date",
        "yield_percent",
        "index_name",
        "index_isin",
        "symbol",
        "source_name",
        "data_status",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "European sovereign benchmark history is missing required columns: "
            f"{sorted(missing)}"
        )

    frame[
        "observation_date"
    ] = pd.to_datetime(
        frame[
            "observation_date"
        ],
        errors="coerce",
    ).dt.date

    frame[
        "tenor_years"
    ] = pd.to_numeric(
        frame[
            "tenor_years"
        ],
        errors="coerce",
    )

    frame[
        "yield_percent"
    ] = pd.to_numeric(
        frame[
            "yield_percent"
        ],
        errors="coerce",
    )

    return frame.dropna(
        subset=[
            "country_code",
            "country_name",
            "tenor_years",
            "observation_date",
            "yield_percent",
        ]
    )


def latest_european_benchmark_yields(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        return history.copy()

    latest = (
        history
        .sort_values(
            [
                "country_code",
                "tenor_years",
                "observation_date",
            ]
        )
        .groupby(
            [
                "country_code",
                "tenor_years",
            ],
            as_index=False,
        )
        .tail(
            1
        )
        .sort_values(
            [
                "country_name",
                "tenor_years",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return latest


def render_european_benchmark_monitor(
    latest_yields: pd.DataFrame,
) -> None:
    st.markdown(
        '<div class="section-label">European benchmark yields</div>',
        unsafe_allow_html=True,
    )

    if latest_yields.empty:
        st.info(
            "No persisted European benchmark-yield observations yet. "
            "Run `python -m src.download_euronext_mts_sovereign_yields`."
        )
        return

    ten_year = latest_yields.loc[
        latest_yields[
            "tenor_years"
        ].eq(
            10
        )
    ].copy()

    german_rows = ten_year.loc[
        ten_year[
            "country_code"
        ].eq(
            "DE"
        )
    ]

    german_yield = (
        float(
            german_rows.iloc[
                -1
            ][
                "yield_percent"
            ]
        )
        if not german_rows.empty
        else None
    )

    if not ten_year.empty:
        ten_year[
            "Spread vs Bund (bp)"
        ] = ten_year[
            "yield_percent"
        ].apply(
            lambda value: (
                (
                    float(
                        value
                    )
                    - german_yield
                )
                * 100.0
                if german_yield is not None
                else None
            )
        )

        display = ten_year[
            [
                "country_name",
                "curve_name",
                "yield_percent",
                "Spread vs Bund (bp)",
                "observation_date",
                "symbol",
                "data_status",
            ]
        ].rename(
            columns={
                "country_name": "Country",
                "curve_name": "Curve",
                "yield_percent": "10Y yield",
                "observation_date": "As of",
                "symbol": "Symbol",
                "data_status": "Status",
            }
        )

        display[
            "Status"
        ] = display[
            "Status"
        ].map(
            display_status
        )

        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Country": st.column_config.TextColumn(
                    "Country"
                ),
                "Curve": st.column_config.TextColumn(
                    "Curve"
                ),
                "10Y yield": st.column_config.NumberColumn(
                    "10Y yield",
                    format="%.3f%%",
                ),
                "Spread vs Bund (bp)": (
                    st.column_config.NumberColumn(
                        "Spread vs Bund",
                        format="%+.1f bp",
                    )
                ),
                "As of": st.column_config.DateColumn(
                    "As of",
                    format="DD MMM YYYY",
                ),
                "Symbol": st.column_config.TextColumn(
                    "MTS symbol"
                ),
                "Status": st.column_config.TextColumn(
                    "Status"
                ),
            },
        )

    curve_countries = (
        latest_yields[
            "country_name"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not curve_countries:
        return

    selected_country = st.selectbox(
        "Benchmark country",
        options=curve_countries,
        index=(
            curve_countries.index(
                "Italy"
            )
            if "Italy" in curve_countries
            else 0
        ),
        key="sovereign_european_benchmark_curve_country",
    )

    curve = latest_yields.loc[
        latest_yields[
            "country_name"
        ].eq(
            selected_country
        )
    ].sort_values(
        "tenor_years"
    )

    figure = go.Figure()

    chart_mode = (
        "lines+markers+text"
        if len(
            curve
        ) > 1
        else "markers+text"
    )

    figure.add_trace(
        go.Scatter(
            x=curve[
                "tenor_years"
            ],
            y=curve[
                "yield_percent"
            ],
            mode=chart_mode,
            text=[
                f"{int(value)}Y"
                for value in curve[
                    "tenor_years"
                ]
            ],
            textposition="top center",
            customdata=curve[
                [
                    "curve_name",
                    "symbol",
                    "observation_date",
                ]
            ],
            hovertemplate=(
                "%{customdata[0]} %{x:.0f}Y<br>"
                "Yield: %{y:.3f}%<br>"
                "Symbol: %{customdata[1]}<br>"
                "As of: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=f"{selected_country} public benchmark curve",
        xaxis_title="Tenor",
        yaxis_title="Yield (%)",
        height=380,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 30,
        },
        showlegend=False,
    )

    figure.update_xaxes(
        ticksuffix="Y"
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )

    if len(
        curve
    ) == 1:
        st.caption(
            f"Only one public benchmark tenor is currently persisted for "
            f"{selected_country}. The selector still lets you inspect every "
            "covered country rather than hiding single-tenor markets."
        )

    st.caption(
        "These are public Euronext MTS benchmark/index yields. They broaden "
        "country-level market coverage and do not replace exact ISIN-level "
        "cash-bond observations."
    )


def latest_spreads_frame(
    spread_history: pd.DataFrame,
) -> pd.DataFrame:
    if spread_history.empty:
        return pd.DataFrame(
            columns=[
                "spread_name",
                "spread_bp",
                "change_percent",
                "implied_change_bp",
                "left_yield_percent",
                "right_yield_percent",
                "observation_date",
                "observation_time",
                "as_of",
                "source_name",
                "data_status",
            ]
        )

    rows: list[
        dict[str, object]
    ] = []

    for spread_name, group in spread_history.groupby(
        "spread_name"
    ):
        latest = group.sort_values(
            [
                "observation_date",
                "observation_time",
            ]
        ).iloc[
            -1
        ]

        change_percent = float(
            latest[
                "change_percent"
            ]
        )

        denominator = (
            1.0
            + change_percent
            / 100.0
        )

        previous = (
            float(
                latest[
                    "spread_bp"
                ]
            )
            / denominator
            if denominator > 0.0
            else None
        )

        implied_change_bp = (
            float(
                latest[
                    "spread_bp"
                ]
            )
            - previous
            if previous is not None
            else None
        )

        observation_date = latest[
            "observation_date"
        ]

        observation_time = str(
            latest.get(
                "observation_time",
                "",
            )
        )

        if observation_time.lower() == "nan":
            observation_time = ""

        as_of = (
            f"{observation_date.strftime('%d %b %Y')} "
            f"{observation_time}"
        ).strip()

        rows.append(
            {
                "spread_name": spread_name,
                "spread_bp": float(
                    latest[
                        "spread_bp"
                    ]
                ),
                "change_percent": change_percent,
                "implied_change_bp": implied_change_bp,
                "left_yield_percent": float(
                    latest[
                        "left_yield_percent"
                    ]
                ),
                "right_yield_percent": float(
                    latest[
                        "right_yield_percent"
                    ]
                ),
                "observation_date": observation_date,
                "observation_time": observation_time,
                "as_of": as_of,
                "source_name": latest[
                    "source_name"
                ],
                "data_status": display_status(
                    latest[
                        "data_status"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "spread_name"
    ).reset_index(
        drop=True
    )


def latest_on_or_before(
    frame: pd.DataFrame,
    *,
    target_date: date,
    column: str,
) -> float | None:
    eligible = frame.loc[
        (
            frame[
                "observation_date"
            ]
            <= target_date
        )
        & frame[
            column
        ].notna()
    ]

    if eligible.empty:
        return None

    latest_row = eligible.sort_values(
        "observation_date"
    ).iloc[
        -1
    ]

    return float(
        latest_row[
            column
        ]
    )


def change_from_lookback(
    frame: pd.DataFrame,
    *,
    latest_date: date,
    latest_value: float | None,
    column: str,
    days: int,
    multiplier: float = 1.0,
) -> float | None:
    if latest_value is None:
        return None

    prior = latest_on_or_before(
        frame,
        target_date=(
            latest_date
            - timedelta(
                days=days
            )
        ),
        column=column,
    )

    if prior is None:
        return None

    return (
        latest_value
        - prior
    ) * multiplier


def latest_instrument_row(
    *,
    instrument: DeskSovereignInstrument,
    frame: pd.DataFrame,
) -> dict[
    str,
    object,
]:
    instrument_history = frame.loc[
        frame[
            "isin"
        ].eq(
            instrument.isin
        )
    ].copy()

    if instrument_history.empty:
        return {
            "Country": instrument.country.value,
            "Sector": f"{instrument.benchmark_tenor_years}Y bucket",
            "Bond": instrument.display_name,
            "ISIN": instrument.isin,
            "Maturity": instrument.maturity_date,
            "Price": None,
            "Yield": None,
            "1D": None,
            "1W": None,
            "1M": None,
            "3M": None,
            "6M": None,
            "9M": None,
            "1Y": None,
            "Obs": None,
            "Source": "No observation",
            "Status": "UNAVAILABLE",
        }

    instrument_history = instrument_history.sort_values(
        "observation_date"
    )

    latest = instrument_history.iloc[
        -1
    ]

    latest_date = latest[
        "observation_date"
    ]

    latest_yield = (
        float(
            latest[
                "yield_percent"
            ]
        )
        if pd.notna(
            latest[
                "yield_percent"
            ]
        )
        else None
    )

    changes = {
        label: change_from_lookback(
            instrument_history,
            latest_date=latest_date,
            latest_value=latest_yield,
            column="yield_percent",
            days=days,
            multiplier=100.0,
        )
        for label, days in WINDOWS.items()
    }

    return {
        "Country": instrument.country.value,
        "Sector": f"{instrument.benchmark_tenor_years}Y bucket",
        "Bond": instrument.display_name,
        "ISIN": instrument.isin,
        "Maturity": instrument.maturity_date,
        "Price": (
            float(
                latest[
                    "price_per_100"
                ]
            )
            if pd.notna(
                latest[
                    "price_per_100"
                ]
            )
            else None
        ),
        "Yield": latest_yield,
        "1D": changes["1D"],
        "1W": changes["1W"],
        "1M": changes["1M"],
        "3M": changes["3M"],
        "6M": changes["6M"],
        "9M": changes["9M"],
        "1Y": changes["1Y"],
        "Obs": latest_date,
        "Source": latest[
            "source_name"
        ],
        "Status": display_status(
            latest[
                "data_status"
            ]
        ),
    }


def market_monitor_frame(
    *,
    instruments: tuple[
        SovereignInstrument,
        ...,
    ],
    history: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            latest_instrument_row(
                instrument=instrument,
                frame=history,
            )
            for instrument in instruments
        ]
    )


def render_spread_strip(
    latest_spreads: pd.DataFrame,
) -> None:
    st.markdown(
        '<div class="section-label">10Y sovereign spread monitor</div>',
        unsafe_allow_html=True,
    )

    if latest_spreads.empty:
        st.info(
            "No persisted sovereign-spread observations yet. "
            "Run `python -m src.download_sovereign_spreads`."
        )
        return

    order = [
        "BTP-BUND",
        "OAT-BUND",
        "BONO-BUND",
    ]

    lookup = {
        row[
            "spread_name"
        ]: row
        for _, row in latest_spreads.iterrows()
    }

    columns = st.columns(
        3
    )

    for column, spread_name in zip(
        columns,
        order,
    ):
        row = lookup.get(
            spread_name
        )

        if row is None:
            column.metric(
                spread_name,
                "N/A",
                border=True,
            )
            continue

        change = row[
            "implied_change_bp"
        ]

        delta = (
            f"{float(change):+.1f} bp implied"
            if pd.notna(
                change
            )
            else None
        )

        column.metric(
            spread_name,
            f"{float(row['spread_bp']):.0f} bp",
            delta=delta,
            border=True,
        )

        column.caption(
            f"{float(row['left_yield_percent']):.2f}% vs "
            f"Bund {float(row['right_yield_percent']):.2f}% · "
            f"{row['as_of']}"
        )

    st.caption(
        "Implied bp change is RepoLens-derived from the source-reported "
        "percentage change in the spread. Source observations remain labelled "
        "PUBLIC REFERENCE."
    )


def render_commentary(
    latest_spreads: pd.DataFrame,
) -> object:
    commentary = build_sovereign_morning_commentary(
        latest_spreads
    )

    st.markdown(
        '<div class="section-label">Morning market commentary</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        commentary.headline
    )

    st.write(
        commentary.summary
    )

    for bullet in commentary.bullets:
        st.markdown(
            f"- {bullet}"
        )

    return commentary


def build_curve_chart(
    monitor: pd.DataFrame,
) -> go.Figure:
    available = monitor.loc[
        monitor[
            "Yield"
        ].notna()
    ].copy()

    if not available.empty:
        available[
            "Tenor"
        ] = pd.to_numeric(
            available[
                "Sector"
            ]
            .astype(
                "string"
            )
            .str.extract(
                r"^\s*(\d+(?:\.\d+)?)",
                expand=False,
            ),
            errors="coerce",
        )

        available = (
            available.loc[
                available[
                    "Tenor"
                ].notna()
            ]
            .sort_values(
                "Tenor"
            )
            .copy()
        )

    figure = go.Figure()

    if not available.empty:
        figure.add_trace(
            go.Scatter(
                x=available[
                    "Tenor"
                ],
                y=available[
                    "Yield"
                ],
                mode="lines+markers+text",
                text=available[
                    "Sector"
                ],
                textposition="top center",
                customdata=available[
                    [
                        "Bond",
                        "ISIN",
                        "Price",
                        "Obs",
                    ]
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "%{customdata[1]}<br>"
                    "Yield: %{y:.3f}%<br>"
                    "Price: %{customdata[2]:.4f}<br>"
                    "Observation: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="Observed sovereign curve",
        xaxis_title="Benchmark sector",
        yaxis_title="Yield (%)",
        height=390,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 30,
        },
        showlegend=False,
    )

    figure.update_xaxes(
        ticksuffix="Y"
    )

    return figure


def render_bond_history(
    *,
    instrument: DeskSovereignInstrument,
    history: pd.DataFrame,
) -> None:
    instrument_history = history.loc[
        history[
            "isin"
        ].eq(
            instrument.isin
        )
    ].copy()

    st.subheader(
        instrument.display_name
    )

    st.caption(
        f"{instrument.isin} · "
        f"{instrument.benchmark_tenor_years}Y maturity bucket · "
        f"maturity {instrument.maturity_date.strftime('%d %b %Y')}"
    )

    if instrument_history.empty:
        st.info(
            "No persisted market observations are available for this ISIN."
        )
        return

    instrument_history = instrument_history.sort_values(
        "observation_date"
    )

    latest = instrument_history.iloc[
        -1
    ]

    columns = st.columns(
        4
    )

    columns[
        0
    ].metric(
        "Reference price",
        (
            f"{float(latest['price_per_100']):.4f}"
            if pd.notna(
                latest[
                    "price_per_100"
                ]
            )
            else "N/A"
        ),
        border=True,
    )

    columns[
        1
    ].metric(
        "Gross yield",
        (
            f"{float(latest['yield_percent']):.3f}%"
            if pd.notna(
                latest[
                    "yield_percent"
                ]
            )
            else "N/A"
        ),
        border=True,
    )

    columns[
        2
    ].metric(
        "Observations",
        f"{len(instrument_history):,}",
        border=True,
    )

    columns[
        3
    ].metric(
        "Latest date",
        latest[
            "observation_date"
        ].strftime(
            "%d %b %Y"
        ),
        border=True,
    )

    chart_data = instrument_history.loc[
        instrument_history[
            "yield_percent"
        ].notna()
    ]

    if not chart_data.empty:
        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=chart_data[
                    "observation_date"
                ],
                y=chart_data[
                    "yield_percent"
                ],
                mode="lines+markers",
                customdata=chart_data[
                    [
                        "price_per_100",
                        "source_name",
                        "data_status",
                    ]
                ],
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Yield: %{y:.3f}%<br>"
                    "Price: %{customdata[0]:.4f}<br>"
                    "Source: %{customdata[1]}<br>"
                    "Status: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

        figure.update_layout(
            title="Yield history",
            xaxis_title="Observation date",
            yaxis_title="Yield (%)",
            height=360,
            margin={
                "l": 20,
                "r": 20,
                "t": 55,
                "b": 30,
            },
            showlegend=False,
        )

        st.plotly_chart(
            figure,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.caption(
        f"Source: {latest['source_name']} · "
        f"Status: {display_status(latest['data_status'])}. "
        "Reference observations are not executable quotes."
    )


def main() -> None:
    st.markdown(
        """
        <div class="repolens-kicker">
            European sovereign morning market
        </div>
        <div class="repolens-title">
            Sovereign Observations
        </div>
        <div class="repolens-subtitle">
            Cash sovereign observations, 10Y cross-market spreads,
            deterministic morning commentary and a broker-ready PDF brief.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        history = observations_frame(
            SovereignHistoryStore(
                SOVEREIGN_HISTORY_PATH
            )
        )

        if not history.empty:
            history[
                "observation_date"
            ] = pd.to_datetime(
                history[
                    "observation_date"
                ],
                errors="coerce",
            ).dt.date

        spread_history = load_spread_history(
            SOVEREIGN_SPREAD_PATH
        )

        latest_spreads = latest_spreads_frame(
            spread_history
        )

        european_benchmark_history = (
            load_european_benchmark_yields(
                EUROPEAN_BENCHMARK_YIELD_PATH
            )
        )

        latest_european_yields = (
            latest_european_benchmark_yields(
                european_benchmark_history
            )
        )

        instruments = load_desk_sovereign_universe()

    except (
        SovereignHistoryStoreValidationError,
        RuntimeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        st.error(
            "RepoLens could not load the sovereign morning market."
        )
        st.code(
            str(
                error
            )
        )
        st.stop()

    render_spread_strip(
        latest_spreads
    )

    commentary = render_commentary(
        latest_spreads
    )

    st.divider()

    render_european_benchmark_monitor(
        latest_european_yields
    )

    st.divider()

    universe_counts = universe_counts_by_country(
        instruments
    )

    st.subheader(
        "Cash sovereign universe"
    )

    st.caption(
        "Reference-universe coverage is separate from exact market-observation "
        "coverage. Instruments without a persisted price/yield stay visible "
        "and are marked unavailable rather than being fabricated."
    )

    universe_metric_columns = st.columns(
        4
    )

    universe_metric_columns[
        0
    ].metric(
        "Countries",
        len(
            universe_counts
        ),
    )

    universe_metric_columns[
        1
    ].metric(
        "Reference instruments",
        len(
            instruments
        ),
    )

    universe_metric_columns[
        2
    ].metric(
        "Italy instruments",
        universe_counts.get(
            "Italy",
            0,
        ),
    )

    universe_metric_columns[
        3
    ].metric(
        "UK gilts",
        universe_counts.get(
            "United Kingdom",
            0,
        ),
    )

    country_options = [
        "All",
        *universe_country_names(
            instruments
        ),
    ]

    control_left, control_middle, control_right = st.columns(
        [
            2,
            2,
            3,
        ]
    )

    with control_left:
        country_filter = st.selectbox(
            "Cash bond country",
            options=country_options,
            index=(
                country_options.index(
                    "Italy"
                )
                if "Italy" in country_options
                else 0
            ),
            key="sovereign_observations_country",
        )

    selected_country_instruments = tuple(
        instrument
        for instrument in instruments
        if (
            country_filter == "All"
            or instrument.country.value == country_filter
        )
    )

    sector_options = [
        "All",
        *sorted(
            {
                f"{instrument.benchmark_tenor_years}Y bucket"
                for instrument in selected_country_instruments
            },
            key=lambda value: int(
                value.split(
                    "Y",
                    1,
                )[
                    0
                ]
            ),
        ),
    ]

    with control_middle:
        sector_filter = st.selectbox(
            "Maturity sector",
            options=sector_options,
            index=0,
            key="sovereign_observations_sector",
        )

    sector_filtered_instruments = tuple(
        instrument
        for instrument in selected_country_instruments
        if (
            sector_filter == "All"
            or f"{instrument.benchmark_tenor_years}Y bucket" == sector_filter
        )
    )

    bond_options = {
        "All bonds": None,
        **{
            (
                f"{instrument.benchmark_tenor_years}Y · "
                f"{instrument.display_name} · "
                f"{instrument.isin}"
            ): instrument.isin
            for instrument in sector_filtered_instruments
        },
    }

    with control_right:
        selected_bond_label = st.selectbox(
            "Bond",
            options=list(
                bond_options
            ),
            index=0,
            key="sovereign_observations_bond_filter",
        )

    selected_bond_isin = bond_options[
        selected_bond_label
    ]

    filtered_instruments = tuple(
        instrument
        for instrument in sector_filtered_instruments
        if (
            selected_bond_isin is None
            or instrument.isin == selected_bond_isin
        )
    )

    monitor = market_monitor_frame(
        instruments=filtered_instruments,
        history=history,
    )

    available_count = (
        int(
            monitor[
                "Yield"
            ].notna().sum()
        )
        if not monitor.empty
        else 0
    )

    latest_dates = [
        value
        for value in monitor.get(
            "Obs",
            pd.Series(
                dtype=object
            ),
        ).tolist()
        if value is not None
        and not pd.isna(
            value
        )
    ]

    latest_market_date = (
        max(
            latest_dates
        )
        if latest_dates
        else None
    )

    top_metrics = st.columns(
        4
    )

    top_metrics[
        0
    ].metric(
        "Bonds",
        f"{len(monitor):,}",
        border=True,
    )

    top_metrics[
        1
    ].metric(
        "With observations",
        f"{available_count:,}",
        border=True,
    )

    top_metrics[
        2
    ].metric(
        "Coverage",
        (
            f"{available_count / len(monitor) * 100.0:.0f}%"
            if len(
                monitor
            )
            else "N/A"
        ),
        border=True,
    )

    top_metrics[
        3
    ].metric(
        "Latest bond observation",
        (
            latest_market_date.strftime(
                "%d %b %Y"
            )
            if latest_market_date is not None
            else "N/A"
        ),
        border=True,
    )

    st.markdown(
        '<div class="section-label">Cash bond monitor</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        monitor,
        hide_index=True,
        width="stretch",
        height=(
            min(
                680,
                92
                + max(
                    len(
                        monitor
                    ),
                    1,
                )
                * 35,
            )
        ),
        column_config={
            "Country": st.column_config.TextColumn(
                "Country"
            ),
            "Sector": st.column_config.TextColumn(
                "Sector"
            ),
            "Bond": st.column_config.TextColumn(
                "Bond",
                width="large",
            ),
            "ISIN": st.column_config.TextColumn(
                "ISIN",
                width="medium",
            ),
            "Maturity": st.column_config.DateColumn(
                "Maturity",
                format="DD MMM YYYY",
            ),
            "Price": st.column_config.NumberColumn(
                "Price",
                format="%.4f",
            ),
            "Yield": st.column_config.NumberColumn(
                "Yield",
                format="%.3f%%",
            ),
            "1D": st.column_config.NumberColumn(
                "1D",
                format="%+.1f bp",
            ),
            "1W": st.column_config.NumberColumn(
                "1W",
                format="%+.1f bp",
            ),
            "1M": st.column_config.NumberColumn(
                "1M",
                format="%+.1f bp",
            ),
            "3M": st.column_config.NumberColumn(
                "3M",
                format="%+.1f bp",
            ),
            "6M": st.column_config.NumberColumn(
                "6M",
                format="%+.1f bp",
            ),
            "9M": st.column_config.NumberColumn(
                "9M",
                format="%+.1f bp",
            ),
            "1Y": st.column_config.NumberColumn(
                "1Y",
                format="%+.1f bp",
            ),
            "Obs": st.column_config.DateColumn(
                "Obs",
                format="DD MMM YYYY",
            ),
            "Source": st.column_config.TextColumn(
                "Source",
                width="large",
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                width="medium",
            ),
        },
    )

    st.caption(
        "1D–1Y columns are changes in yield, expressed in basis points, "
        "versus the latest available observation on or before each lookback "
        "date. They remain N/A until sufficient history exists."
    )

    if not monitor.empty:
        st.plotly_chart(
            build_curve_chart(
                monitor
            ),
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Broker export</div>',
        unsafe_allow_html=True,
    )

    pdf_bytes = build_sovereign_morning_pdf(
        monitor=monitor,
        spreads=latest_spreads,
        commentary=commentary,
        generated_at=datetime.now(),
    )

    st.download_button(
        "Generate Morning Sovereign Brief PDF",
        data=pdf_bytes,
        file_name=(
            "RepoLens_Morning_Sovereign_Brief_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        ),
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

    st.caption(
        "The PDF uses the same persisted bond and sovereign-spread observations "
        "shown on this page, with source/status provenance and deterministic "
        "market commentary."
    )

    if monitor.empty:
        return

    st.divider()

    instrument_options = {
        (
            f"{row['Sector']} · "
            f"{row['Bond']} · "
            f"{row['ISIN']}"
        ): row[
            "ISIN"
        ]
        for _, row in monitor.iterrows()
    }

    selected_label = st.selectbox(
        "Inspect bond history",
        options=list(
            instrument_options
        ),
        key="sovereign_observations_selected_bond",
    )

    selected_isin = instrument_options[
        selected_label
    ]

    selected_instrument = next(
        instrument
        for instrument in filtered_instruments
        if instrument.isin == selected_isin
    )

    render_bond_history(
        instrument=selected_instrument,
        history=history,
    )

    st.divider()

    st.markdown(
        """
        **Data treatment**

        RepoLens displays persisted observations as sourced and does not invent
        missing historical prices, yields or spreads. PUBLIC REFERENCE denotes
        permitted public reference data and does not imply an executable quote.

        Sovereign-spread commentary is rule-based and derived from the observed
        BTP-Bund, OAT-Bund and Bono-Bund fields. It is descriptive, not a
        forecast or trade recommendation.
        """
    )


main()