from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.repo_analytics import (
    RepoTradeInput,
    RepoValidationError,
    calculate_repo_trade,
    required_collateral_market_value,
    required_face_value,
)


def format_euro(
    value: float,
    decimals: int = 0,
) -> str:
    """
    Format a euro amount.
    """
    return f"€{value:,.{decimals}f}"


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
            Translate a bond price and broker repo quote into cash raised,
            haircut, repo interest, repurchase price and GC specialness.
        </div>
        """,
        unsafe_allow_html=True,
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
                "**Collateral**"
            )

            face_value_eur = st.number_input(
                "Face value (€)",
                min_value=1_000.0,
                value=10_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="repo_face_value",
            )

            price_left, price_right = st.columns(
                2
            )

            with price_left:
                clean_price = st.number_input(
                    "Clean price / 100",
                    min_value=0.0,
                    value=100.0,
                    step=0.01,
                    format="%.4f",
                    key="repo_clean_price",
                )

            with price_right:
                accrued_interest = st.number_input(
                    "Accrued / 100",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.4f",
                    key="repo_accrued",
                )

            interim_income_eur = st.number_input(
                "Interim coupon / income (€)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                format="%.2f",
                key="repo_interim_income",
                help=(
                    "For classic repo, an equivalent manufactured "
                    "payment is returned to the collateral provider."
                ),
            )

        with repo_column:
            st.markdown(
                "**Repo terms**"
            )

            today = date.today()

            date_left, date_right = st.columns(
                2
            )

            with date_left:
                purchase_date = st.date_input(
                    "Purchase date",
                    value=today,
                    key="repo_purchase_date",
                )

            with date_right:
                repurchase_date = st.date_input(
                    "Repurchase date",
                    value=(
                        today
                        + timedelta(
                            days=7
                        )
                    ),
                    key="repo_repurchase_date",
                )

            rate_left, rate_right = st.columns(
                2
            )

            with rate_left:
                repo_rate_percent = st.number_input(
                    "Repo rate (%)",
                    min_value=-20.0,
                    max_value=30.0,
                    value=2.00,
                    step=0.01,
                    format="%.4f",
                    key="repo_rate",
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
                options=[
                    360,
                    365,
                ],
                index=0,
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
                accrued_interest
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
                if gc_repo_rate_percent is not None
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

    st.markdown(
        '<div class="section-label">Transaction economics</div>',
        unsafe_allow_html=True,
    )

    economics_top = st.columns(
        4
    )

    economics_top[0].metric(
        "Dirty price",
        f"{result.dirty_price_per_100:,.4f}",
        delta=(
            f"Clean {clean_price:,.4f} + accrued "
            f"{accrued_interest:,.4f}"
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

    if (
        result.specialness_bp is not None
        and result.financing_benefit_vs_gc_eur is not None
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
            delta="For collateral provider over this term",
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
        sizing_left, sizing_middle, sizing_right = st.columns(
            3
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

        required_market_value = required_collateral_market_value(
            target_cash_eur=float(
                target_cash_eur
            ),
            haircut_percent=float(
                haircut_percent
            ),
        )

        required_nominal = required_face_value(
            target_cash_eur=float(
                target_cash_eur
            ),
            dirty_price=result.dirty_price_per_100,
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
            **Purchase price** is the collateral market value after the
            configured haircut.

            **Repo interest** uses simple interest on the purchase price
            over the actual number of calendar days using the selected
            360- or 365-day money-market basis.

            **Repurchase price** equals purchase price plus repo interest.
            Negative repo rates are supported.
            """
        )

    with methodology_right:
        st.subheader(
            "Data classification"
        )

        st.markdown(
            """
            Bond price, accrued interest, repo rate, GC rate, haircut and
            interim income are explicit **desk / broker inputs**.

            Calculated cash, haircut amount, repo interest, specialness and
            required collateral are **RepoLens-derived analytics**.

            This page is an analytical calculator, not a trade confirmation,
            settlement, margin-maintenance or regulatory reporting system.
            """
        )


main()