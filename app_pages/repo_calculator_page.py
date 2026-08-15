from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.bond_analytics import (
    BondValidationError,
    accrued_interest,
    coupon_period_dates,
)
from src.repo_analytics import (
    RepoTradeInput,
    RepoValidationError,
    calculate_repo_trade,
    required_collateral_market_value,
    required_face_value,
)
from src.sovereign_instrument_catalog import (
    all_instruments,
    master_record_by_isin,
)
from src.sovereign_instruments import SovereignInstrument


def format_euro(value: float, decimals: int = 0) -> str:
    return f"€{value:,.{decimals}f}"


def instrument_label(instrument: SovereignInstrument) -> str:
    coupon_percent = instrument.annual_coupon_rate * 100.0
    return (
        f"{instrument.country_code} · "
        f"{instrument.benchmark_tenor_years}Y · "
        f"{coupon_percent:.2f}% "
        f"{instrument.security_type.value} · "
        f"{instrument.maturity_date.strftime('%b-%Y')}"
    )


def default_instrument_index(
    instruments: tuple[SovereignInstrument, ...],
) -> int:
    for index, instrument in enumerate(instruments):
        if (
            instrument.country_code == "DE"
            and instrument.benchmark_tenor_years == 2
        ):
            return index
    return 0


def main() -> None:
    st.markdown(
        """
        <div class="repolens-kicker">
            Secured funding and collateral economics
        </div>
        <div class="repolens-title">
            Repo Calculator
        </div>
        <div class="repolens-subtitle">
            Select RepoLens collateral or use manual terms, then translate
            a broker repo quote into cash raised, haircut, repo interest,
            repurchase price and GC specialness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        catalogue = tuple(
            sorted(
                all_instruments(),
                key=lambda instrument: (
                    instrument.maturity_date,
                    instrument.country_code,
                    instrument.isin,
                ),
            )
        )
    except RuntimeError as error:
        st.error("RepoLens could not load the sovereign collateral universe.")
        st.code(str(error))
        st.stop()

    if not catalogue:
        st.error("The RepoLens sovereign collateral universe is empty.")
        st.stop()

    st.markdown(
        '<div class="section-label">Collateral selection</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        collateral_mode = st.radio(
            "Collateral source",
            options=[
                "RepoLens instrument",
                "Manual collateral",
            ],
            horizontal=True,
            key="repo_collateral_mode",
        )

        selected_instrument: SovereignInstrument | None = None
        derived_accrued_interest: float | None = None

        if collateral_mode == "RepoLens instrument":
            instruments_by_label = {
                instrument_label(instrument): instrument
                for instrument in catalogue
            }
            labels = list(instruments_by_label)

            selected_label = st.selectbox(
                "Collateral instrument",
                options=labels,
                index=default_instrument_index(catalogue),
                key="repo_selected_instrument",
            )

            selected_instrument = instruments_by_label[selected_label]
            record = master_record_by_isin(selected_instrument.isin)

            st.markdown(f"**{selected_instrument.display_name}**")
            st.caption(
                f"{selected_instrument.isin} · "
                f"{selected_instrument.country.value} · "
                f"{selected_instrument.security_type.value} · "
                f"Maturity {selected_instrument.maturity_date.strftime('%d %b %Y')} · "
                f"{record.benchmark_status.value}"
            )
        else:
            st.caption(
                "Manual mode is available for collateral that is not yet "
                "in the RepoLens instrument catalogue."
            )

    st.markdown(
        '<div class="section-label">Transaction inputs</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        collateral_column, repo_column = st.columns(2)

        with collateral_column:
            st.markdown("**Collateral economics**")

            face_value_eur = st.number_input(
                "Face value (€)",
                min_value=1_000.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="repo_face_value",
            )

            clean_price = st.number_input(
                "Clean price / 100",
                min_value=0.0,
                value=100.0,
                step=0.01,
                format="%.4f",
                key="repo_clean_price",
                help=(
                    "Explicit desk / broker market input. "
                    "RepoLens does not invent an executable bond price."
                ),
            )

            purchase_date = st.date_input(
                "Purchase date",
                value=date.today(),
                key="repo_purchase_date",
            )

            manual_accrued = collateral_mode == "Manual collateral"

            if (
                selected_instrument is not None
                and purchase_date >= selected_instrument.maturity_date
            ):
                st.error(
                    "Purchase date must be before the selected bond's maturity."
                )
                st.stop()

            if selected_instrument is not None:
                try:
                    bond = selected_instrument.to_fixed_rate_bond()

                    derived_accrued_interest = accrued_interest(
                        bond=bond,
                        settlement_date=purchase_date,
                    )

                    previous_coupon, next_coupon = coupon_period_dates(
                        bond=bond,
                        settlement_date=purchase_date,
                    )

                    days_to_maturity = (
                        selected_instrument.maturity_date - purchase_date
                    ).days

                    st.caption(
                        "RepoLens contractual schedule: "
                        f"previous coupon {previous_coupon.strftime('%d %b %Y')} · "
                        f"next coupon {next_coupon.strftime('%d %b %Y')} · "
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

                    manual_accrued = accrued_mode == "Desk override"

                except BondValidationError as error:
                    st.warning(
                        "RepoLens could not derive accrued interest from "
                        "the selected contractual schedule."
                    )
                    st.code(str(error))
                    manual_accrued = True

            if manual_accrued:
                accrued_interest_per_100 = st.number_input(
                    "Accrued interest / 100",
                    min_value=0.0,
                    value=(
                        float(derived_accrued_interest)
                        if derived_accrued_interest is not None
                        else 0.0
                    ),
                    step=0.01,
                    format="%.4f",
                    key="repo_accrued_manual",
                )
                st.caption(
                    "Accrued interest is an explicit desk / broker input."
                )
            else:
                accrued_interest_per_100 = float(
                    derived_accrued_interest
                    if derived_accrued_interest is not None
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
                help=(
                    "Classic repo income paid on collateral during the "
                    "transaction can be recorded here as an equivalent "
                    "manufactured payment."
                ),
            )

        with repo_column:
            st.markdown("**Repo terms**")

            repurchase_date = st.date_input(
                "Repurchase date",
                value=purchase_date + timedelta(days=7),
                min_value=purchase_date + timedelta(days=1),
                key="repo_repurchase_date",
            )

            if (
                selected_instrument is not None
                and repurchase_date >= selected_instrument.maturity_date
            ):
                st.warning(
                    "The selected repurchase date reaches or passes bond "
                    "maturity. This v2 calculator is intended for repo terms "
                    "ending before collateral maturity."
                )

            rate_left, rate_right = st.columns(2)

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

            day_count_basis = st.selectbox(
                "Money-market day-count basis",
                options=[360, 365],
                index=0,
                format_func=lambda value: f"Actual/{value}",
                key="repo_day_count_basis",
            )

            compare_with_gc = st.checkbox(
                "Compare specific repo quote with GC",
                value=True,
                key="repo_compare_gc",
            )

            gc_repo_rate_percent: float | None = None

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
            face_value_eur=float(face_value_eur),
            clean_price_per_100=float(clean_price),
            accrued_interest_per_100=float(accrued_interest_per_100),
            repo_rate_percent=float(repo_rate_percent),
            haircut_percent=float(haircut_percent),
            purchase_date=purchase_date,
            repurchase_date=repurchase_date,
            day_count_basis=int(day_count_basis),
            gc_repo_rate_percent=(
                float(gc_repo_rate_percent)
                if gc_repo_rate_percent is not None
                else None
            ),
            interim_income_eur=float(interim_income_eur),
        )

        result = calculate_repo_trade(trade)

    except RepoValidationError as error:
        st.error("RepoLens could not calculate this repo transaction.")
        st.code(str(error))
        st.stop()

    if (
        selected_instrument is not None
        and repurchase_date >= selected_instrument.maturity_date
    ):
        st.stop()

    st.markdown(
        '<div class="section-label">Transaction economics</div>',
        unsafe_allow_html=True,
    )

    economics_top = st.columns(4)

    economics_top[0].metric(
        "Dirty price",
        f"{result.dirty_price_per_100:,.4f}",
        delta=(
            f"Clean {clean_price:,.4f} + accrued "
            f"{accrued_interest_per_100:,.4f}"
        ),
        delta_color="off",
        border=True,
    )

    economics_top[1].metric(
        "Collateral market value",
        format_euro(result.collateral_market_value_eur),
        delta="Before haircut",
        delta_color="off",
        border=True,
    )

    economics_top[2].metric(
        "Cash advanced",
        format_euro(result.purchase_price_eur),
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

    economics_bottom = st.columns(4)

    economics_bottom[0].metric(
        "Haircut amount",
        format_euro(result.haircut_amount_eur),
        border=True,
    )

    economics_bottom[1].metric(
        "Repo interest",
        format_euro(result.repo_interest_eur, decimals=2),
        delta=f"{repo_rate_percent:.4f}% p.a.",
        delta_color="off",
        border=True,
    )

    economics_bottom[2].metric(
        "Repurchase price",
        format_euro(result.repurchase_price_eur, decimals=2),
        delta="Purchase price + repo interest",
        delta_color="off",
        border=True,
    )

    economics_bottom[3].metric(
        "Manufactured payment",
        format_euro(result.manufactured_payment_eur, decimals=2),
        delta="Interim income, if any",
        delta_color="off",
        border=True,
    )

    if (
        result.specialness_bp is not None
        and result.financing_benefit_vs_gc_eur is not None
    ):
        st.markdown(
            '<div class="section-label">GC comparison</div>',
            unsafe_allow_html=True,
        )

        special_columns = st.columns(3)

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

    with st.container(border=True):
        sizing_left, sizing_middle, sizing_right = st.columns(3)

        with sizing_left:
            target_cash_eur = st.number_input(
                "Target cash to raise (€)",
                min_value=1_000.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="repo_target_cash",
            )

        required_market_value = required_collateral_market_value(
            target_cash_eur=float(target_cash_eur),
            haircut_percent=float(haircut_percent),
        )

        required_nominal = required_face_value(
            target_cash_eur=float(target_cash_eur),
            dirty_price=result.dirty_price_per_100,
            haircut_percent=float(haircut_percent),
        )

        with sizing_middle:
            st.metric(
                "Required collateral value",
                format_euro(required_market_value),
                border=True,
            )

        with sizing_right:
            st.metric(
                "Required face value",
                format_euro(required_nominal),
                border=True,
            )

    st.divider()

    methodology_left, methodology_right = st.columns(2)

    with methodology_left:
        st.subheader("Repo mechanics")
        st.markdown(
            """
            **Purchase price** is collateral market value after the
            configured haircut.

            **Repo interest** is simple interest on purchase price over
            the actual number of calendar days using the selected
            360- or 365-day money-market basis.

            **Repurchase price** equals purchase price plus repo interest.
            Negative repo rates are supported.
            """
        )

    with methodology_right:
        st.subheader("Data classification")
        st.markdown(
            """
            Selected instrument terms come from the RepoLens sovereign
            reference catalogue.

            Clean price, repo rate, GC rate, haircut and any desk override
            are explicit **desk / broker inputs**.

            Schedule-derived accrued interest and all cash, repo-interest,
            specialness and sizing outputs are **RepoLens-derived analytics**.

            This is an analytical calculator, not a trade confirmation,
            settlement, margin-maintenance or regulatory reporting system.
            """
        )


main()