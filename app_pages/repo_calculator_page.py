from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.bond_analytics import (
    BondValidationError,
    accrued_interest,
    coupon_period_dates,
)
from src.repo_analytics import (
    RepoTradeInput,
    RepoValidationError,
    analyse_discount_security_carry_to_maturity,
    analyse_financed_bond_carry,
    calculate_repo_trade,
    required_collateral_market_value,
    required_face_value,
)
from src.sovereign_instrument_catalog import (
    all_instruments,
    master_record_by_isin,
)
from src.sovereign_instruments import SovereignInstrument
from src.sovereign_money_market import (
    MONEY_MARKET_INSTRUMENTS,
    SovereignDiscountSecurity,
)


def format_euro(
    value: float,
    decimals: int = 0,
) -> str:
    """
    Format a euro amount.
    """
    return f"€{value:,.{decimals}f}"


def coupon_instrument_label(
    instrument: SovereignInstrument,
) -> str:
    """
    Create a compact desk-style coupon-bond selector label.
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


def money_market_instrument_label(
    instrument: SovereignDiscountSecurity,
    as_of_date: date,
) -> str:
    """
    Create a compact desk-style money-market collateral label.
    """
    bucket = instrument.remaining_maturity_bucket(
        as_of_date
    )

    return (
        f"{instrument.country_code} · "
        f"{bucket.value} · "
        f"{instrument.security_type.value} · "
        f"{instrument.maturity_date.strftime('%b-%Y')}"
    )


def default_coupon_instrument_index(
    instruments: tuple[
        SovereignInstrument,
        ...,
    ],
) -> int:
    """
    Prefer a short German coupon instrument for repo work.
    """
    for index, instrument in enumerate(
        instruments
    ):
        if (
            instrument.country_code == "DE"
            and instrument.benchmark_tenor_years == 2
        ):
            return index

    return 0


def main() -> None:
    """
    Render the RepoLens Repo Calculator.
    """
    st.markdown(
        """
        <div class="repolens-kicker">
            Secured funding and collateral economics
        </div>
        <div class="repolens-title">
            Repo Calculator
        </div>
        <div class="repolens-subtitle">
            Select sovereign collateral or use manual terms, then translate
            a broker repo quote into cash raised, haircut, repo interest,
            repurchase price and GC specialness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        coupon_catalogue = tuple(
            sorted(
                all_instruments(),
                key=lambda instrument: (
                    instrument.maturity_date,
                    instrument.country_code,
                    instrument.isin,
                ),
            )
        )

        money_market_catalogue = tuple(
            sorted(
                MONEY_MARKET_INSTRUMENTS,
                key=lambda instrument: (
                    instrument.maturity_date,
                    instrument.country_code,
                    instrument.isin,
                ),
            )
        )

    except RuntimeError as error:
        st.error(
            "RepoLens could not load the sovereign collateral universe."
        )
        st.code(
            str(
                error
            )
        )
        st.stop()

    if (
        not coupon_catalogue
        and not money_market_catalogue
    ):
        st.error(
            "The RepoLens sovereign collateral universe is empty."
        )
        st.stop()

    st.markdown(
        '<div class="section-label">Collateral selection</div>',
        unsafe_allow_html=True,
    )

    selected_coupon_instrument: (
        SovereignInstrument
        | None
    ) = None

    selected_money_market_instrument: (
        SovereignDiscountSecurity
        | None
    ) = None

    derived_accrued_interest: (
        float
        | None
    ) = None

    with st.container(
        border=True
    ):
        collateral_mode = st.radio(
            "Collateral source",
            options=[
                "RepoLens coupon bond",
                "RepoLens money market",
                "Manual collateral",
            ],
            horizontal=True,
            key="repo_collateral_mode",
        )

        if collateral_mode == "RepoLens coupon bond":
            if not coupon_catalogue:
                st.warning(
                    "No coupon bonds are available in the RepoLens catalogue."
                )
                st.stop()

            instruments_by_label = {
                coupon_instrument_label(
                    instrument
                ): instrument
                for instrument
                in coupon_catalogue
            }

            labels = list(
                instruments_by_label
            )

            selected_label = st.selectbox(
                "Collateral instrument",
                options=labels,
                index=(
                    default_coupon_instrument_index(
                        coupon_catalogue
                    )
                ),
                key="repo_selected_coupon_instrument",
            )

            selected_coupon_instrument = (
                instruments_by_label[
                    selected_label
                ]
            )

            record = master_record_by_isin(
                selected_coupon_instrument.isin
            )

            st.markdown(
                f"**{selected_coupon_instrument.display_name}**"
            )

            st.caption(
                f"{selected_coupon_instrument.isin} · "
                f"{selected_coupon_instrument.country.value} · "
                f"{selected_coupon_instrument.security_type.value} · "
                f"Maturity "
                f"{selected_coupon_instrument.maturity_date.strftime('%d %b %Y')} · "
                f"{record.benchmark_status.value}"
            )

        elif collateral_mode == "RepoLens money market":
            if not money_market_catalogue:
                st.warning(
                    "No money-market collateral is available."
                )
                st.stop()

            as_of_date = date.today()

            instruments_by_label = {
                money_market_instrument_label(
                    instrument=instrument,
                    as_of_date=as_of_date,
                ): instrument
                for instrument
                in money_market_catalogue
            }

            labels = list(
                instruments_by_label
            )

            selected_label = st.selectbox(
                "Money-market collateral",
                options=labels,
                index=0,
                key="repo_selected_money_market_instrument",
            )

            selected_money_market_instrument = (
                instruments_by_label[
                    selected_label
                ]
            )

            bucket = (
                selected_money_market_instrument
                .remaining_maturity_bucket(
                    as_of_date
                )
            )

            days_to_maturity = (
                selected_money_market_instrument
                .days_to_maturity(
                    as_of_date
                )
            )

            st.markdown(
                f"**{selected_money_market_instrument.display_name}**"
            )

            st.caption(
                f"{selected_money_market_instrument.isin} · "
                f"{selected_money_market_instrument.country} · "
                f"{selected_money_market_instrument.security_type.value} · "
                f"{bucket.value} remaining maturity · "
                f"{days_to_maturity:,} days to maturity"
            )

            st.caption(
                "Zero coupon · Redemption 100.00 · "
                f"Interest convention Actual/"
                f"{selected_money_market_instrument.interest_day_count_basis} · "
                f"{selected_money_market_instrument.data_status.value}"
            )

        else:
            st.caption(
                "Manual mode remains available for collateral that is not "
                "in the RepoLens reference universe."
            )

    st.markdown(
        '<div class="section-label">Transaction inputs</div>',
        unsafe_allow_html=True,
    )

    with st.container(
        border=True
    ):
        collateral_column, repo_column = st.columns(
            2
        )

        with collateral_column:
            st.markdown(
                "**Collateral economics**"
            )

            face_value_eur = st.number_input(
                "Face value (€)",
                min_value=1_000.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="repo_face_value",
            )

            clean_price = st.number_input(
                "Price / 100",
                min_value=0.0,
                value=100.0,
                step=0.01,
                format="%.4f",
                key="repo_clean_price",
                help=(
                    "Explicit desk / broker market input. "
                    "RepoLens does not invent an executable collateral price."
                ),
            )

            purchase_date = st.date_input(
                "Purchase date",
                value=date.today(),
                key="repo_purchase_date",
            )

            if (
                selected_coupon_instrument is not None
                and purchase_date
                >= selected_coupon_instrument.maturity_date
            ):
                st.error(
                    "Purchase date must be before the selected bond's maturity."
                )
                st.stop()

            if (
                selected_money_market_instrument is not None
                and purchase_date
                >= selected_money_market_instrument.maturity_date
            ):
                st.error(
                    "Purchase date must be before the selected "
                    "money-market security's maturity."
                )
                st.stop()

            manual_accrued = (
                collateral_mode
                == "Manual collateral"
            )

            if selected_coupon_instrument is not None:
                try:
                    bond = (
                        selected_coupon_instrument
                        .to_fixed_rate_bond()
                    )

                    derived_accrued_interest = (
                        accrued_interest(
                            bond=bond,
                            settlement_date=purchase_date,
                        )
                    )

                    previous_coupon, next_coupon = (
                        coupon_period_dates(
                            bond=bond,
                            settlement_date=purchase_date,
                        )
                    )

                    days_to_maturity = (
                        selected_coupon_instrument.maturity_date
                        - purchase_date
                    ).days

                    st.caption(
                        "RepoLens contractual schedule: "
                        f"previous coupon "
                        f"{previous_coupon.strftime('%d %b %Y')} · "
                        f"next coupon "
                        f"{next_coupon.strftime('%d %b %Y')} · "
                        f"{days_to_maturity:,} days to maturity"
                    )

                    accrued_mode = st.radio(
                        "Accrued-interest source",
                        options=[
                            "RepoLens schedule",
                            "Desk override",
                        ],
                        horizontal=True,
                        key="repo_accrued_mode",
                    )

                    manual_accrued = (
                        accrued_mode
                        == "Desk override"
                    )

                except BondValidationError as error:
                    st.warning(
                        "RepoLens could not derive accrued interest "
                        "from the selected contractual schedule."
                    )
                    st.code(
                        str(
                            error
                        )
                    )
                    manual_accrued = True

            if selected_money_market_instrument is not None:
                accrued_interest_per_100 = 0.0

                bucket = (
                    selected_money_market_instrument
                    .remaining_maturity_bucket(
                        purchase_date
                    )
                )

                days_to_maturity = (
                    selected_money_market_instrument
                    .days_to_maturity(
                        purchase_date
                    )
                )

                st.metric(
                    "Accrued interest / 100",
                    "0.0000",
                    delta="Zero-coupon discount security",
                    delta_color="off",
                    border=True,
                )

                st.caption(
                    f"{bucket.value} bucket · "
                    f"{days_to_maturity:,} days remaining · "
                    "No coupon accrual"
                )

                manual_accrued = False

            elif manual_accrued:
                accrued_interest_per_100 = (
                    st.number_input(
                        "Accrued interest / 100",
                        min_value=0.0,
                        value=(
                            float(
                                derived_accrued_interest
                            )
                            if derived_accrued_interest
                            is not None
                            else 0.0
                        ),
                        step=0.01,
                        format="%.4f",
                        key="repo_accrued_manual",
                    )
                )

                st.caption(
                    "Accrued interest is an explicit desk / broker input."
                )

            else:
                accrued_interest_per_100 = float(
                    derived_accrued_interest
                    if derived_accrued_interest
                    is not None
                    else 0.0
                )

                st.metric(
                    "RepoLens accrued / 100",
                    f"{accrued_interest_per_100:,.4f}",
                    delta="Derived from contractual coupon schedule",
                    delta_color="off",
                    border=True,
                )

            interim_income_eur = st.number_input(
                "Interim coupon / income (€)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                format="%.2f",
                key="repo_interim_income",
                disabled=(
                    selected_money_market_instrument
                    is not None
                ),
                help=(
                    "Classic repo income paid on coupon collateral during "
                    "the transaction can be recorded here as an equivalent "
                    "manufactured payment. Zero-coupon money-market "
                    "securities have no coupon income."
                ),
            )

        with repo_column:
            st.markdown(
                "**Repo terms**"
            )

            repurchase_date = st.date_input(
                "Repurchase date",
                value=(
                    purchase_date
                    + timedelta(
                        days=7
                    )
                ),
                min_value=(
                    purchase_date
                    + timedelta(
                        days=1
                    )
                ),
                key="repo_repurchase_date",
            )

            selected_maturity_date: (
                date
                | None
            ) = None

            if selected_coupon_instrument is not None:
                selected_maturity_date = (
                    selected_coupon_instrument
                    .maturity_date
                )

            if selected_money_market_instrument is not None:
                selected_maturity_date = (
                    selected_money_market_instrument
                    .maturity_date
                )

            if (
                selected_maturity_date
                is not None
                and repurchase_date
                >= selected_maturity_date
            ):
                st.warning(
                    "The repurchase date reaches or passes collateral "
                    "maturity. This calculator currently requires the repo "
                    "to end before collateral maturity."
                )

            rate_left, rate_right = st.columns(
                2
            )

            with rate_left:
                repo_rate_percent = st.number_input(
                    "Specific repo rate (%)",
                    min_value=-20.0,
                    max_value=30.0,
                    value=2.00,
                    step=0.01,
                    format="%.4f",
                    key="repo_rate",
                    help="Explicit broker / desk repo quote.",
                )

            with rate_right:
                haircut_percent = st.number_input(
                    "Haircut (%)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=0.00,
                    step=0.10,
                    format="%.3f",
                    key="repo_haircut",
                )

            default_day_count_basis = 360

            if selected_money_market_instrument is not None:
                default_day_count_basis = (
                    selected_money_market_instrument
                    .interest_day_count_basis
                )

            day_count_options = [
                360,
                365,
            ]

            day_count_basis = st.selectbox(
                "Money-market day-count basis",
                options=day_count_options,
                index=day_count_options.index(
                    default_day_count_basis
                ),
                format_func=lambda value: (
                    f"Actual/{value}"
                ),
                key="repo_day_count_basis",
            )

            compare_with_gc = st.checkbox(
                "Compare specific repo quote with GC",
                value=True,
                key="repo_compare_gc",
            )

            gc_repo_rate_percent: (
                float
                | None
            ) = None

            if compare_with_gc:
                gc_repo_rate_percent = st.number_input(
                    "GC repo rate (%)",
                    min_value=-20.0,
                    max_value=30.0,
                    value=2.25,
                    step=0.01,
                    format="%.4f",
                    key="repo_gc_rate",
                    help="Explicit desk / broker GC reference input.",
                )

    try:
        trade = RepoTradeInput(
            face_value_eur=float(
                face_value_eur
            ),
            clean_price_per_100=float(
                clean_price
            ),
            accrued_interest_per_100=float(
                accrued_interest_per_100
            ),
            repo_rate_percent=float(
                repo_rate_percent
            ),
            haircut_percent=float(
                haircut_percent
            ),
            purchase_date=purchase_date,
            repurchase_date=repurchase_date,
            day_count_basis=int(
                day_count_basis
            ),
            gc_repo_rate_percent=(
                float(
                    gc_repo_rate_percent
                )
                if gc_repo_rate_percent
                is not None
                else None
            ),
            interim_income_eur=float(
                interim_income_eur
            ),
        )

        result = calculate_repo_trade(
            trade
        )

    except RepoValidationError as error:
        st.error(
            "RepoLens could not calculate this repo transaction."
        )
        st.code(
            str(
                error
            )
        )
        st.stop()

    if (
        selected_maturity_date
        is not None
        and repurchase_date
        >= selected_maturity_date
    ):
        st.stop()

    st.markdown(
        '<div class="section-label">Transaction economics</div>',
        unsafe_allow_html=True,
    )

    economics_top = st.columns(
        4
    )

    economics_top[0].metric(
        "Dirty / full price",
        f"{result.dirty_price_per_100:,.4f}",
        delta=(
            f"Price {clean_price:,.4f} + accrued "
            f"{accrued_interest_per_100:,.4f}"
        ),
        delta_color="off",
        border=True,
    )

    economics_top[1].metric(
        "Collateral market value",
        format_euro(
            result.collateral_market_value_eur
        ),
        delta="Before haircut",
        delta_color="off",
        border=True,
    )

    economics_top[2].metric(
        "Cash advanced",
        format_euro(
            result.purchase_price_eur
        ),
        delta=f"{haircut_percent:.3f}% haircut",
        delta_color="off",
        border=True,
    )

    economics_top[3].metric(
        "Repo term",
        f"{result.repo_days} days",
        delta=f"Actual/{day_count_basis}",
        delta_color="off",
        border=True,
    )

    economics_bottom = st.columns(
        4
    )

    economics_bottom[0].metric(
        "Haircut amount",
        format_euro(
            result.haircut_amount_eur
        ),
        border=True,
    )

    economics_bottom[1].metric(
        "Repo interest",
        format_euro(
            result.repo_interest_eur,
            decimals=2,
        ),
        delta=f"{repo_rate_percent:.4f}% p.a.",
        delta_color="off",
        border=True,
    )

    economics_bottom[2].metric(
        "Repurchase price",
        format_euro(
            result.repurchase_price_eur,
            decimals=2,
        ),
        delta="Purchase price + repo interest",
        delta_color="off",
        border=True,
    )

    economics_bottom[3].metric(
        "Manufactured payment",
        format_euro(
            result.manufactured_payment_eur,
            decimals=2,
        ),
        delta="Interim income, if any",
        delta_color="off",
        border=True,
    )

    if selected_coupon_instrument is not None:
        st.markdown(
            '<div class="section-label">Financing-adjusted bond carry</div>',
            unsafe_allow_html=True,
        )

        try:
            carry_analysis = analyse_financed_bond_carry(
                bond=(
                    selected_coupon_instrument
                    .to_fixed_rate_bond()
                ),
                trade=trade,
            )

        except RepoValidationError as error:
            st.warning(
                "RepoLens could not calculate financed bond carry "
                "for the selected inputs."
            )
            st.code(
                str(
                    error
                )
            )

        else:
            unchanged_scenario = next(
                scenario
                for scenario
                in carry_analysis.scenarios
                if scenario.yield_shock_bp == 0.0
            )

            carry_columns = st.columns(
                4
            )

            carry_columns[0].metric(
                "Implied start yield",
                f"{carry_analysis.start_yield_percent:.3f}%",
                delta="Solved from desk clean price",
                delta_color="off",
                border=True,
            )

            carry_columns[1].metric(
                "Unchanged-yield P&L",
                format_euro(
                    unchanged_scenario
                    .financing_adjusted_pnl_eur,
                    decimals=2,
                ),
                delta=(
                    f"{format_euro(carry_analysis.coupon_income_eur, decimals=2)} "
                    "coupon income"
                ),
                delta_color="off",
                border=True,
            )

            carry_columns[2].metric(
                "P&L per €1m face",
                format_euro(
                    unchanged_scenario
                    .financing_adjusted_pnl_per_eur_1m_face,
                    decimals=2,
                ),
                delta="Unchanged exit yield",
                delta_color="off",
                border=True,
            )

            breakeven_move = (
                carry_analysis
                .breakeven_yield_move_bp
            )

            breakeven_yield = (
                carry_analysis
                .breakeven_exit_yield_percent
            )

            carry_columns[3].metric(
                "Breakeven yield move",
                (
                    f"{breakeven_move:+.2f} bp"
                    if breakeven_move is not None
                    else "N/A"
                ),
                delta=(
                    f"Exit yield {breakeven_yield:.3f}%"
                    if breakeven_yield is not None
                    else "No valid breakeven solve"
                ),
                delta_color="off",
                border=True,
            )

            scenario_frame = pd.DataFrame(
                [
                    {
                        "Yield shock (bp)": scenario.yield_shock_bp,
                        "Exit yield (%)": scenario.exit_yield_percent,
                        "Exit dirty price": (
                            scenario.exit_dirty_price_per_100
                        ),
                        "Coupon income (€)": scenario.coupon_income_eur,
                        "Gross bond P&L (€)": scenario.gross_bond_pnl_eur,
                        "Repo interest (€)": scenario.repo_interest_eur,
                        "Financing-adjusted P&L (€)": (
                            scenario.financing_adjusted_pnl_eur
                        ),
                        "P&L / €1m face (€)": (
                            scenario
                            .financing_adjusted_pnl_per_eur_1m_face
                        ),
                    }
                    for scenario
                    in carry_analysis.scenarios
                ]
            )

            st.dataframe(
                scenario_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Yield shock (bp)": st.column_config.NumberColumn(
                        "Yield shock",
                        format="%+.0f bp",
                    ),
                    "Exit yield (%)": st.column_config.NumberColumn(
                        "Exit yield",
                        format="%.3f%%",
                    ),
                    "Exit dirty price": st.column_config.NumberColumn(
                        "Exit dirty",
                        format="%.4f",
                    ),
                    "Coupon income (€)": st.column_config.NumberColumn(
                        "Coupon income",
                        format="€%,.2f",
                    ),
                    "Gross bond P&L (€)": st.column_config.NumberColumn(
                        "Gross bond P&L",
                        format="€%,.2f",
                    ),
                    "Repo interest (€)": st.column_config.NumberColumn(
                        "Repo interest",
                        format="€%,.2f",
                    ),
                    "Financing-adjusted P&L (€)": (
                        st.column_config.NumberColumn(
                            "Financing-adjusted P&L",
                            format="€%,.2f",
                        )
                    ),
                    "P&L / €1m face (€)": st.column_config.NumberColumn(
                        "P&L / €1m face",
                        format="€%,.2f",
                    ),
                },
            )

            st.caption(
                "Scenario P&L = exit bond market value + coupon income "
                "minus starting collateral market value minus repo interest. "
                "Yield shocks are applied to the yield implied by the "
                "entered clean price. Transaction costs and haircut "
                "opportunity cost are not included."
            )

    if selected_money_market_instrument is not None:
        st.markdown(
            '<div class="section-label">Financing-adjusted discount carry</div>',
            unsafe_allow_html=True,
        )

        try:
            discount_carry = (
                analyse_discount_security_carry_to_maturity(
                    face_value_eur=float(
                        face_value_eur
                    ),
                    price_per_100=float(
                        clean_price
                    ),
                    redemption_value_per_100=(
                        selected_money_market_instrument
                        .redemption_value_per_100
                    ),
                    purchase_date=purchase_date,
                    maturity_date=(
                        selected_money_market_instrument
                        .maturity_date
                    ),
                    repo_rate_percent=float(
                        repo_rate_percent
                    ),
                    haircut_percent=float(
                        haircut_percent
                    ),
                    day_count_basis=int(
                        day_count_basis
                    ),
                    gc_repo_rate_percent=(
                        float(
                            gc_repo_rate_percent
                        )
                        if gc_repo_rate_percent
                        is not None
                        else None
                    ),
                )
            )

        except RepoValidationError as error:
            st.warning(
                "RepoLens could not calculate financing-to-maturity "
                "economics for the selected discount security."
            )
            st.code(
                str(
                    error
                )
            )

        else:
            carry_columns = st.columns(
                4
            )

            carry_columns[0].metric(
                "Gross pull-to-par",
                format_euro(
                    discount_carry
                    .gross_pull_to_par_eur,
                    decimals=2,
                ),
                delta="Before financing cost",
                delta_color="off",
                border=True,
            )

            carry_columns[1].metric(
                "Funding cost to maturity",
                format_euro(
                    discount_carry
                    .financing_cost_to_maturity_eur,
                    decimals=2,
                ),
                delta=(
                    f"Assumes {repo_rate_percent:.4f}% "
                    "can be maintained"
                ),
                delta_color="off",
                border=True,
            )

            carry_columns[2].metric(
                "Financing-adjusted pull-to-par",
                format_euro(
                    discount_carry
                    .financing_adjusted_pull_to_par_eur,
                    decimals=2,
                ),
                delta=(
                    f"{format_euro(discount_carry.financing_adjusted_pull_to_par_per_eur_1m_face, decimals=2)} "
                    "per €1m face"
                ),
                delta_color="off",
                border=True,
            )

            carry_columns[3].metric(
                "Breakeven repo rate",
                (
                    f"{discount_carry.breakeven_repo_rate_percent:.4f}%"
                ),
                delta="Net pull-to-par = €0",
                delta_color="off",
                border=True,
            )

            carry_second_row = st.columns(
                3
            )

            carry_second_row[0].metric(
                "Annualised net carry",
                (
                    f"{discount_carry.financing_adjusted_annualised_return_percent:+.3f}%"
                ),
                delta="Scenario annualised on selected day-count",
                delta_color="off",
                border=True,
            )

            carry_second_row[1].metric(
                "Days to maturity",
                f"{discount_carry.days_to_maturity}",
                border=True,
            )

            if (
                discount_carry
                .financing_benefit_vs_gc_to_maturity_eur
                is not None
            ):
                carry_second_row[2].metric(
                    "Benefit vs GC to maturity",
                    format_euro(
                        discount_carry
                        .financing_benefit_vs_gc_to_maturity_eur,
                        decimals=2,
                    ),
                    delta=(
                        "Assumes both repo rates remain unchanged "
                        "through maturity"
                    ),
                    delta_color="off",
                    border=True,
                )
            else:
                carry_second_row[2].metric(
                    "Benefit vs GC to maturity",
                    "N/A",
                    delta="Enable GC comparison to calculate",
                    delta_color="off",
                    border=True,
                )

            st.caption(
                "Financing-to-maturity is a scenario: RepoLens assumes the "
                "entered repo rate can be maintained or rolled unchanged "
                "until redemption. It excludes haircut opportunity cost, "
                "transaction costs, margin changes and future repo-rate changes."
            )

    if selected_money_market_instrument is not None:
        st.markdown(
            '<div class="section-label">Discount-security economics</div>',
            unsafe_allow_html=True,
        )

        discount_columns = st.columns(
            3
        )

        redemption_value = (
            selected_money_market_instrument
            .redemption_value_eur(
                face_value_eur=float(
                    face_value_eur
                )
            )
        )

        pull_to_par = (
            selected_money_market_instrument
            .pull_to_par_eur(
                face_value_eur=float(
                    face_value_eur
                ),
                price_per_100=float(
                    clean_price
                ),
            )
        )

        discount_columns[0].metric(
            "Redemption value",
            format_euro(
                redemption_value
            ),
            delta="Contractual value at maturity",
            delta_color="off",
            border=True,
        )

        discount_columns[1].metric(
            "Gross pull-to-par",
            format_euro(
                pull_to_par,
                decimals=2,
            ),
            delta="Before financing and costs",
            delta_color="off",
            border=True,
        )

        discount_columns[2].metric(
            "Remaining maturity",
            (
                selected_money_market_instrument
                .remaining_maturity_bucket(
                    purchase_date
                )
                .value
            ),
            delta=(
                f"{selected_money_market_instrument.days_to_maturity(purchase_date)} "
                "days"
            ),
            delta_color="off",
            border=True,
        )

    if (
        result.specialness_bp is not None
        and result.financing_benefit_vs_gc_eur
        is not None
    ):
        st.markdown(
            '<div class="section-label">GC comparison</div>',
            unsafe_allow_html=True,
        )

        special_columns = st.columns(
            3
        )

        special_columns[0].metric(
            "GC rate",
            f"{gc_repo_rate_percent:.4f}%",
            border=True,
        )

        special_columns[1].metric(
            "Specialness",
            f"{result.specialness_bp:+,.2f} bp",
            delta="GC minus specific repo",
            delta_color="off",
            border=True,
        )

        special_columns[2].metric(
            "Financing benefit vs GC",
            format_euro(
                result.financing_benefit_vs_gc_eur,
                decimals=2,
            ),
            delta="Over the selected repo term",
            delta_color="off",
            border=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Cash-to-collateral sizing</div>',
        unsafe_allow_html=True,
    )

    with st.container(
        border=True
    ):
        sizing_left, sizing_middle, sizing_right = (
            st.columns(
                3
            )
        )

        with sizing_left:
            target_cash_eur = st.number_input(
                "Target cash to raise (€)",
                min_value=1_000.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="repo_target_cash",
            )

        required_market_value = (
            required_collateral_market_value(
                target_cash_eur=float(
                    target_cash_eur
                ),
                haircut_percent=float(
                    haircut_percent
                ),
            )
        )

        required_nominal = required_face_value(
            target_cash_eur=float(
                target_cash_eur
            ),
            dirty_price=(
                result
                .dirty_price_per_100
            ),
            haircut_percent=float(
                haircut_percent
            ),
        )

        with sizing_middle:
            st.metric(
                "Required collateral value",
                format_euro(
                    required_market_value
                ),
                border=True,
            )

        with sizing_right:
            st.metric(
                "Required face value",
                format_euro(
                    required_nominal
                ),
                border=True,
            )

    st.divider()

    methodology_left, methodology_right = st.columns(
        2
    )

    with methodology_left:
        st.subheader(
            "Repo mechanics"
        )

        st.markdown(
            """
            **Purchase price** is collateral market value after the
            configured haircut.

            **Repo interest** is simple interest on purchase price over
            the actual number of calendar days using the selected
            360- or 365-day money-market basis.

            **Repurchase price** equals purchase price plus repo interest.
            Negative repo rates are supported.

            Coupon-bond carry scenarios reprice the bond at repo maturity,
            include coupon income during the horizon and deduct repo interest.
            The breakeven yield is the exit yield at which that
            financing-adjusted horizon P&L is zero.

            Money-market discount securities are modelled separately from
            coupon bonds. Their accrued interest is zero. Financing-to-maturity
            carry uses contractual redemption and assumes the entered repo
            rate can be maintained unchanged through maturity.
            """
        )

    with methodology_right:
        st.subheader(
            "Data classification"
        )

        st.markdown(
            """
            Selected instrument terms come from the RepoLens sovereign
            reference universe.

            Price, repo rate, GC rate, haircut and any desk override are
            explicit **desk / broker inputs**.

            Schedule-derived accrued interest, remaining-maturity buckets,
            cash, repo-interest, specialness, pull-to-par, financed carry,
            breakeven yield and scenario P&L are **RepoLens-derived analytics**.

            Manual collateral mode remains available and unchanged for
            instruments outside the RepoLens reference universe.

            This is an analytical calculator, not a trade confirmation,
            settlement, margin-maintenance or regulatory reporting system.
            """
        )


main()