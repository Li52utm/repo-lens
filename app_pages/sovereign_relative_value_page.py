from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.bond_analytics import (
    BondValidationError,
    dirty_price_from_yield,
)
from src.repo_adjusted_relative_value import (
    RepoAdjustedRelativeValueValidationError,
    RepoFundingLegInput,
    analyse_repo_adjusted_relative_value,
)
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

    with st.container(
        border=True
    ):
        st.markdown(
            '<div class="section-label">Trade builder</div>',
            unsafe_allow_html=True,
        )

        selector_left, selector_right = st.columns(
            2
        )

        with selector_left:
            anchor_label = st.selectbox(
                "Anchor instrument",
                options=labels,
                index=labels.index(
                    default_anchor_label
                ),
                key="rv_anchor_instrument",
            )

        with selector_right:
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

        st.caption(
            f"Anchor: {anchor_instrument.display_name} · "
            f"{anchor_instrument.isin}    |    "
            f"Hedge: {hedge_instrument.display_name} · "
            f"{hedge_instrument.isin}"
        )

        setup_one, setup_two, setup_three = st.columns(
            3
        )

        with setup_one:
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

        with setup_two:
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

        with setup_three:
            settlement_date = st.date_input(
                "Settlement date",
                value=default_settlement,
                min_value=earliest_settlement,
                max_value=latest_settlement,
                key="rv_settlement_date",
            )

        st.caption(
            f"Hedge direction is automatically {hedge_direction.value} "
            "to create the opposite leg."
        )

        anchor_german_yield = latest_german_yield(
            benchmark_data=benchmark_data,
            tenor_years=anchor_instrument.benchmark_tenor_years,
        )

        hedge_german_yield = latest_german_yield(
            benchmark_data=benchmark_data,
            tenor_years=hedge_instrument.benchmark_tenor_years,
        )

        yield_left, yield_right = st.columns(
            2
        )

        with yield_left:
            st.markdown(
                "**Anchor market input**"
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

                st.metric(
                    "Official anchor yield",
                    f"{anchor_yield_percent:.3f}%",
                    border=True,
                )

                st.caption(
                    "Official German daily benchmark reference data."
                )

        with yield_right:
            st.markdown(
                "**Hedge market input**"
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

                st.metric(
                    "Official hedge yield",
                    f"{hedge_yield_percent:.3f}%",
                    border=True,
                )

                st.caption(
                    "Official German daily benchmark reference data."
                )

        st.divider()

        repo_overlay_enabled = st.checkbox(
            "Overlay repo funding economics",
            value=True,
            key="rv_repo_overlay_enabled",
            help=(
                "Add explicit desk / broker specific-repo and matched GC "
                "inputs to the DV01-neutral cash-bond trade."
            ),
        )

        anchor_specific_repo_rate_percent = 0.0
        anchor_gc_repo_rate_percent = 0.0
        anchor_haircut_percent = 0.0
        hedge_specific_repo_rate_percent = 0.0
        hedge_gc_repo_rate_percent = 0.0
        hedge_haircut_percent = 0.0
        repo_days = 30
        repo_day_count_basis = 360

        if repo_overlay_enabled:
            st.markdown(
                "**Repo funding overlay inputs**"
            )

            repo_setup_one, repo_setup_two = st.columns(
                2
            )

            with repo_setup_one:
                repo_days = st.number_input(
                    "Matched repo term (days)",
                    min_value=1,
                    max_value=366,
                    value=30,
                    step=1,
                    key="rv_repo_days",
                    help=(
                        "The same contractual repo horizon is applied to "
                        "both legs. RepoLens does not compare mismatched terms."
                    ),
                )

            with repo_setup_two:
                repo_day_count_basis = st.selectbox(
                    "Repo day-count basis",
                    options=[
                        360,
                        365,
                    ],
                    index=0,
                    format_func=lambda value: (
                        f"Actual/{value}"
                    ),
                    key="rv_repo_day_count_basis",
                )

            repo_anchor_column, repo_hedge_column = st.columns(
                2
            )

            with repo_anchor_column:
                st.markdown(
                    f"**Anchor funding · {anchor_instrument.isin}**"
                )

                anchor_repo_rate_columns = st.columns(
                    2
                )

                with anchor_repo_rate_columns[0]:
                    anchor_specific_repo_rate_percent = (
                        st.number_input(
                            "Anchor specific repo (%)",
                            min_value=-20.0,
                            max_value=30.0,
                            value=2.00,
                            step=0.01,
                            format="%.4f",
                            key=(
                                "rv_anchor_specific_repo_"
                                f"{anchor_instrument.isin}"
                            ),
                            help=(
                                "Explicit desk / broker specific-repo input."
                            ),
                        )
                    )

                with anchor_repo_rate_columns[1]:
                    anchor_gc_repo_rate_percent = (
                        st.number_input(
                            "Anchor matched GC (%)",
                            min_value=-20.0,
                            max_value=30.0,
                            value=2.25,
                            step=0.01,
                            format="%.4f",
                            key=(
                                "rv_anchor_gc_repo_"
                                f"{anchor_instrument.isin}"
                            ),
                            help=(
                                "Matched secured GC reference. This is not €STR."
                            ),
                        )
                    )

                anchor_haircut_percent = st.number_input(
                    "Anchor haircut (%)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=0.00,
                    step=0.10,
                    format="%.3f",
                    key=(
                        "rv_anchor_haircut_"
                        f"{anchor_instrument.isin}"
                    ),
                )

            with repo_hedge_column:
                st.markdown(
                    f"**Hedge funding · {hedge_instrument.isin}**"
                )

                hedge_repo_rate_columns = st.columns(
                    2
                )

                with hedge_repo_rate_columns[0]:
                    hedge_specific_repo_rate_percent = (
                        st.number_input(
                            "Hedge specific repo (%)",
                            min_value=-20.0,
                            max_value=30.0,
                            value=2.00,
                            step=0.01,
                            format="%.4f",
                            key=(
                                "rv_hedge_specific_repo_"
                                f"{hedge_instrument.isin}"
                            ),
                            help=(
                                "Explicit desk / broker specific-repo input."
                            ),
                        )
                    )

                with hedge_repo_rate_columns[1]:
                    hedge_gc_repo_rate_percent = (
                        st.number_input(
                            "Hedge matched GC (%)",
                            min_value=-20.0,
                            max_value=30.0,
                            value=2.25,
                            step=0.01,
                            format="%.4f",
                            key=(
                                "rv_hedge_gc_repo_"
                                f"{hedge_instrument.isin}"
                            ),
                            help=(
                                "Matched secured GC reference. This is not €STR."
                            ),
                        )
                    )

                hedge_haircut_percent = st.number_input(
                    "Hedge haircut (%)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=0.00,
                    step=0.10,
                    format="%.3f",
                    key=(
                        "rv_hedge_haircut_"
                        f"{hedge_instrument.isin}"
                    ),
                )

            st.caption(
                "Repo inputs are explicit desk / broker assumptions. "
                "RepoLens does not manufacture executable specific-repo or GC quotes."
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

        repo_adjusted_analysis = None

        if repo_overlay_enabled:
            anchor_dirty_price = dirty_price_from_yield(
                bond=(
                    anchor_instrument
                    .to_fixed_rate_bond()
                ),
                settlement_date=settlement_date,
                yield_to_maturity=(
                    float(
                        anchor_yield_percent
                    )
                    / 100.0
                ),
            )

            hedge_dirty_price = dirty_price_from_yield(
                bond=(
                    hedge_instrument
                    .to_fixed_rate_bond()
                ),
                settlement_date=settlement_date,
                yield_to_maturity=(
                    float(
                        hedge_yield_percent
                    )
                    / 100.0
                ),
            )

            repo_adjusted_analysis = (
                analyse_repo_adjusted_relative_value(
                    anchor=RepoFundingLegInput(
                        isin=anchor_instrument.isin,
                        direction=position.anchor_direction,
                        face_value_eur=(
                            position.anchor_notional_eur
                        ),
                        dirty_price_per_100=(
                            anchor_dirty_price
                        ),
                        haircut_percent=float(
                            anchor_haircut_percent
                        ),
                        specific_repo_rate_percent=float(
                            anchor_specific_repo_rate_percent
                        ),
                        gc_repo_rate_percent=float(
                            anchor_gc_repo_rate_percent
                        ),
                        repo_days=int(
                            repo_days
                        ),
                        day_count_basis=int(
                            repo_day_count_basis
                        ),
                    ),
                    hedge=RepoFundingLegInput(
                        isin=hedge_instrument.isin,
                        direction=position.hedge_direction,
                        face_value_eur=(
                            position.hedge_notional_eur
                        ),
                        dirty_price_per_100=(
                            hedge_dirty_price
                        ),
                        haircut_percent=float(
                            hedge_haircut_percent
                        ),
                        specific_repo_rate_percent=float(
                            hedge_specific_repo_rate_percent
                        ),
                        gc_repo_rate_percent=float(
                            hedge_gc_repo_rate_percent
                        ),
                        repo_days=int(
                            repo_days
                        ),
                        day_count_basis=int(
                            repo_day_count_basis
                        ),
                    ),
                )
            )
    except (
        RelativeValueValidationError,
        RepoAdjustedRelativeValueValidationError,
        BondValidationError,
    ) as error:
        st.error(
            "RepoLens could not construct the relative-value position "
            "and funding overlay."
        )

        st.code(
            str(
                error
            )
        )

        st.stop()

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

    if repo_adjusted_analysis is not None:
        st.markdown(
            '<div class="section-label">Repo funding overlay</div>',
            unsafe_allow_html=True,
        )

        funding_columns = st.columns(
            5
        )

        funding_columns[0].metric(
            "Anchor specialness",
            (
                f"{repo_adjusted_analysis.anchor.specialness_bp:+.2f} bp"
            ),
            delta=(
                f"{repo_adjusted_analysis.anchor.specific_repo_rate_percent:.4f}% "
                "specific"
            ),
            delta_color="off",
            border=True,
        )

        funding_columns[1].metric(
            "Hedge specialness",
            (
                f"{repo_adjusted_analysis.hedge.specialness_bp:+.2f} bp"
            ),
            delta=(
                f"{repo_adjusted_analysis.hedge.specific_repo_rate_percent:.4f}% "
                "specific"
            ),
            delta_color="off",
            border=True,
        )

        funding_columns[2].metric(
            "Specialness differential",
            (
                f"{repo_adjusted_analysis.anchor_minus_hedge_specialness_bp:+.2f} bp"
            ),
            delta="Anchor minus hedge",
            delta_color="off",
            border=True,
        )

        funding_columns[3].metric(
            "Net funding overlay vs GC",
            format_euro(
                repo_adjusted_analysis
                .net_signed_financing_impact_vs_gc_eur,
                decimals=2,
            ),
            delta=(
                f"{repo_adjusted_analysis.repo_days} day matched term"
            ),
            delta_color="off",
            border=True,
        )

        funding_columns[4].metric(
            "Net overlay / €1m anchor",
            format_euro(
                repo_adjusted_analysis
                .net_signed_financing_impact_per_eur_1m_anchor_face,
                decimals=2,
            ),
            delta="Normalised to anchor face",
            delta_color="off",
            border=True,
        )

        funding_leg_frame = pd.DataFrame(
            [
                {
                    "Leg": "Anchor",
                    "ISIN": (
                        repo_adjusted_analysis
                        .anchor
                        .isin
                    ),
                    "Direction": (
                        repo_adjusted_analysis
                        .anchor
                        .direction
                        .value
                    ),
                    "Face value (€)": (
                        repo_adjusted_analysis
                        .anchor
                        .face_value_eur
                    ),
                    "Dirty price": (
                        repo_adjusted_analysis
                        .anchor
                        .dirty_price_per_100
                    ),
                    "Specific repo (%)": (
                        repo_adjusted_analysis
                        .anchor
                        .specific_repo_rate_percent
                    ),
                    "GC (%)": (
                        repo_adjusted_analysis
                        .anchor
                        .gc_repo_rate_percent
                    ),
                    "Specialness (bp)": (
                        repo_adjusted_analysis
                        .anchor
                        .specialness_bp
                    ),
                    "Signed funding impact (€)": (
                        repo_adjusted_analysis
                        .anchor
                        .signed_financing_impact_vs_gc_eur
                    ),
                },
                {
                    "Leg": "Hedge",
                    "ISIN": (
                        repo_adjusted_analysis
                        .hedge
                        .isin
                    ),
                    "Direction": (
                        repo_adjusted_analysis
                        .hedge
                        .direction
                        .value
                    ),
                    "Face value (€)": (
                        repo_adjusted_analysis
                        .hedge
                        .face_value_eur
                    ),
                    "Dirty price": (
                        repo_adjusted_analysis
                        .hedge
                        .dirty_price_per_100
                    ),
                    "Specific repo (%)": (
                        repo_adjusted_analysis
                        .hedge
                        .specific_repo_rate_percent
                    ),
                    "GC (%)": (
                        repo_adjusted_analysis
                        .hedge
                        .gc_repo_rate_percent
                    ),
                    "Specialness (bp)": (
                        repo_adjusted_analysis
                        .hedge
                        .specialness_bp
                    ),
                    "Signed funding impact (€)": (
                        repo_adjusted_analysis
                        .hedge
                        .signed_financing_impact_vs_gc_eur
                    ),
                },
            ]
        )

        st.dataframe(
            funding_leg_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Face value (€)": (
                    st.column_config.NumberColumn(
                        "Face value",
                        format="€%,.0f",
                    )
                ),
                "Dirty price": (
                    st.column_config.NumberColumn(
                        "Dirty price",
                        format="%.4f",
                    )
                ),
                "Specific repo (%)": (
                    st.column_config.NumberColumn(
                        "Specific repo",
                        format="%.4f%%",
                    )
                ),
                "GC (%)": (
                    st.column_config.NumberColumn(
                        "GC",
                        format="%.4f%%",
                    )
                ),
                "Specialness (bp)": (
                    st.column_config.NumberColumn(
                        "Specialness",
                        format="%+.2f bp",
                    )
                ),
                "Signed funding impact (€)": (
                    st.column_config.NumberColumn(
                        "Signed funding impact",
                        format="€%,.2f",
                    )
                ),
            },
        )

        if (
            repo_adjusted_analysis
            .net_signed_financing_impact_vs_gc_eur
            > 0.0
        ):
            st.success(
                "On the entered repo assumptions, specific-collateral funding "
                "improves this DV01-neutral trade versus funding both legs at "
                "their matched GC references."
            )
        elif (
            repo_adjusted_analysis
            .net_signed_financing_impact_vs_gc_eur
            < 0.0
        ):
            st.warning(
                "On the entered repo assumptions, specific-collateral funding "
                "detracts from this DV01-neutral trade versus funding both legs "
                "at their matched GC references."
            )
        else:
            st.info(
                "On the entered repo assumptions, the two signed funding "
                "effects offset versus matched GC."
            )

        st.caption(
            "For a long collateral leg, cheaper specific funding versus GC "
            "adds value. For a short collateral leg, collateral scarcity is "
            "applied with the opposite sign because obtaining the bond is an "
            "economic cost to the short. This is a funding overlay, not an "
            "executable trade recommendation."
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

            When the repo overlay is enabled, RepoLens keeps cash-bond RV and
            financing RV separate. A positive net funding overlay means the
            entered specific-repo economics improve the trade versus matched
            GC funding over the selected repo horizon; a negative value means
            they detract from it.
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

            Specific-repo rates, GC references and haircuts are explicit
            desk / broker inputs and are not executable quotes supplied by
            RepoLens.

            Hedge ratios, DV01, spread measures, scenario P&L, specialness and
            the signed specific-versus-GC funding overlay are RepoLens-derived
            research analytics.
            """
        )


main()