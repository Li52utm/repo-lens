from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

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
from src.repo_adjusted_carry import (
    RepoAdjustedCarryValidationError,
    analyse_repo_adjusted_bond_carry,
)
from src.repo_market_state import (
    GCReference,
    RepoClearingType,
    RepoCounterpartySegment,
    RepoMarketStateValidationError,
    RepoQuoteSourceType,
    RepoSpecialnessResult,
    SpecificRepoQuote,
    compare_specific_to_gc,
)
from src.repo_specialness_history import (
    RepoSpecialnessHistoryValidationError,
    analyse_specialness_history,
    observation_from_result,
)
from src.repo_specialness_store import (
    RepoSpecialnessStoreError,
    append_repo_specialness_record,
    history_observations_for_market,
    load_repo_specialness_records,
    stored_record_from_market_state,
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


def format_optional_bp(
    value: float | None,
    decimals: int = 2,
) -> str:
    """
    Format an optional basis-point value.
    """
    if value is None:
        return "N/A"

    return f"{value:+,.{decimals}f} bp"


def format_optional_z_score(
    value: float | None,
) -> str:
    """
    Format an optional z-score.
    """
    if value is None:
        return "N/A"

    return f"{value:+.2f}σ"


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

    structured_specialness: (
        RepoSpecialnessResult
        | None
    ) = None

    structured_specific_quote: (
        SpecificRepoQuote
        | None
    ) = None

    structured_gc_reference: (
        GCReference
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

            specific_quote_source_name = (
                "Interactive desk / broker input"
            )
            gc_quote_source_name = (
                "Interactive desk / broker input"
            )

            specific_quote_source_type = (
                RepoQuoteSourceType.DESK_INPUT
            )
            gc_quote_source_type = (
                RepoQuoteSourceType.DESK_INPUT
            )

            quote_timestamp_mode = (
                "Input capture time"
            )
            explicit_specific_quote_date = None
            explicit_specific_quote_time = None
            explicit_gc_quote_date = None
            explicit_gc_quote_time = None

            specific_quote_venue = ""
            gc_quote_venue = ""
            gc_basket_name = ""
            specific_clearing_type = RepoClearingType.UNSPECIFIED
            gc_clearing_type = RepoClearingType.UNSPECIFIED
            specific_counterparty_segment = (
                RepoCounterpartySegment.UNSPECIFIED
            )
            gc_counterparty_segment = (
                RepoCounterpartySegment.UNSPECIFIED
            )

            if compare_with_gc:
                gc_repo_rate_percent = st.number_input(
                    "GC repo rate (%)",
                    min_value=-20.0,
                    max_value=30.0,
                    value=2.25,
                    step=0.01,
                    format="%.4f",
                    key="repo_gc_rate",
                    help=(
                        "Explicit desk / broker GC reference input. "
                        "This is not €STR."
                    ),
                )

                with st.expander(
                    "GC / specific quote provenance"
                ):
                    provenance_left, provenance_right = (
                        st.columns(
                            2
                        )
                    )

                    with provenance_left:
                        specific_quote_source_name = (
                            st.text_input(
                                "Specific quote source",
                                value=(
                                    "Interactive desk / broker input"
                                ),
                                key=(
                                    "repo_specific_quote_source"
                                ),
                            )
                        )

                        specific_quote_source_type = (
                            st.selectbox(
                                "Specific source type",
                                options=list(
                                    RepoQuoteSourceType
                                ),
                                index=0,
                                format_func=lambda value: (
                                    value.value
                                    .replace(
                                        "_",
                                        " ",
                                    )
                                    .title()
                                ),
                                key=(
                                    "repo_specific_source_type"
                                ),
                            )
                        )

                    with provenance_right:
                        gc_quote_source_name = (
                            st.text_input(
                                "GC reference source",
                                value=(
                                    "Interactive desk / broker input"
                                ),
                                key=(
                                    "repo_gc_quote_source"
                                ),
                            )
                        )

                        gc_quote_source_type = (
                            st.selectbox(
                                "GC source type",
                                options=list(
                                    RepoQuoteSourceType
                                ),
                                index=0,
                                format_func=lambda value: (
                                    value.value
                                    .replace(
                                        "_",
                                        " ",
                                    )
                                    .title()
                                ),
                                key=(
                                    "repo_gc_source_type"
                                ),
                            )
                        )

                    st.markdown(
                        "**Market context**"
                    )

                    context_left, context_right = st.columns(2)

                    with context_left:
                        specific_quote_venue = st.text_input(
                            "Specific quote venue",
                            value="",
                            placeholder="e.g. broker, CCP venue, bilateral",
                            key="repo_specific_quote_venue",
                        )

                    with context_right:
                        gc_quote_venue = st.text_input(
                            "GC reference venue",
                            value="",
                            placeholder="e.g. Eurex, desk composite",
                            key="repo_gc_quote_venue",
                        )

                    gc_basket_name = st.text_input(
                        "GC basket / reference identity",
                        value="",
                        placeholder="e.g. EUR sovereign GC, GC Pooling basket",
                        key="repo_gc_basket_name",
                        help=(
                            "Name the GC basket or reference explicitly. "
                            "RepoLens does not assume all GC observations "
                            "refer to the same collateral pool."
                        ),
                    )

                    clearing_left, clearing_right = st.columns(2)

                    with clearing_left:
                        specific_clearing_type = st.selectbox(
                            "Specific clearing type",
                            options=list(RepoClearingType),
                            index=0,
                            format_func=lambda item: (
                                item.value.replace("_", " ").title()
                            ),
                            key="repo_specific_clearing_type",
                        )

                    with clearing_right:
                        gc_clearing_type = st.selectbox(
                            "GC clearing type",
                            options=list(RepoClearingType),
                            index=0,
                            format_func=lambda item: (
                                item.value.replace("_", " ").title()
                            ),
                            key="repo_gc_clearing_type",
                        )

                    segment_left, segment_right = st.columns(2)

                    with segment_left:
                        specific_counterparty_segment = st.selectbox(
                            "Specific market segment",
                            options=list(RepoCounterpartySegment),
                            index=0,
                            format_func=lambda item: (
                                item.value.replace("_", " ").title()
                            ),
                            key="repo_specific_counterparty_segment",
                        )

                    with segment_right:
                        gc_counterparty_segment = st.selectbox(
                            "GC market segment",
                            options=list(RepoCounterpartySegment),
                            index=0,
                            format_func=lambda item: (
                                item.value.replace("_", " ").title()
                            ),
                            key="repo_gc_counterparty_segment",
                        )

                    st.caption(
                        "Venue, clearing and market-segment fields are "
                        "comparison context. RepoLens flags differences "
                        "rather than assuming the observations are identical."
                    )

                    st.divider()

                    quote_timestamp_mode = (
                        st.radio(
                            "Quote timestamp treatment",
                            options=[
                                "Input capture time",
                                "Explicit quote time",
                            ],
                            index=0,
                            horizontal=True,
                            key=(
                                "repo_quote_timestamp_mode"
                            ),
                            help=(
                                "Use input capture time only when the entered "
                                "rates represent the current observed market. "
                                "Use explicit quote time for broker, venue or "
                                "desk observations received earlier."
                            ),
                        )
                    )

                    if (
                        quote_timestamp_mode
                        == "Explicit quote time"
                    ):
                        timestamp_now_utc = (
                            datetime.now(
                                timezone.utc
                            )
                        )

                        specific_time_left, specific_time_right = (
                            st.columns(
                                2
                            )
                        )

                        with specific_time_left:
                            explicit_specific_quote_date = (
                                st.date_input(
                                    "Specific quote date (UTC)",
                                    value=(
                                        timestamp_now_utc.date()
                                    ),
                                    key=(
                                        "repo_specific_quote_date"
                                    ),
                                )
                            )

                        with specific_time_right:
                            explicit_specific_quote_time = (
                                st.time_input(
                                    "Specific quote time (UTC)",
                                    value=(
                                        timestamp_now_utc
                                        .replace(
                                            second=0,
                                            microsecond=0,
                                        )
                                        .timetz()
                                        .replace(
                                            tzinfo=None
                                        )
                                    ),
                                    step=60,
                                    key=(
                                        "repo_specific_quote_time"
                                    ),
                                )
                            )

                        gc_time_left, gc_time_right = (
                            st.columns(
                                2
                            )
                        )

                        with gc_time_left:
                            explicit_gc_quote_date = (
                                st.date_input(
                                    "GC quote date (UTC)",
                                    value=(
                                        timestamp_now_utc.date()
                                    ),
                                    key=(
                                        "repo_gc_quote_date"
                                    ),
                                )
                            )

                        with gc_time_right:
                            explicit_gc_quote_time = (
                                st.time_input(
                                    "GC quote time (UTC)",
                                    value=(
                                        timestamp_now_utc
                                        .replace(
                                            second=0,
                                            microsecond=0,
                                        )
                                        .timetz()
                                        .replace(
                                            tzinfo=None
                                        )
                                    ),
                                    step=60,
                                    key=(
                                        "repo_gc_quote_time"
                                    ),
                                )
                            )

                        st.caption(
                            "Explicit timestamps are interpreted as UTC. "
                            "RepoLens preserves separate specific and GC quote "
                            "times and reports their time gap."
                        )

                    else:
                        st.caption(
                            "RepoLens will stamp both observations with the "
                            "current UTC calculator capture time. This should "
                            "only be used when the entered rates represent the "
                            "market observed now."
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

        selected_collateral_isin: (
            str
            | None
        ) = None

        selected_collateral_currency: (
            str
            | None
        ) = None

        if selected_coupon_instrument is not None:
            selected_collateral_isin = (
                selected_coupon_instrument.isin
            )
            selected_collateral_currency = (
                selected_coupon_instrument.currency
            )

        elif selected_money_market_instrument is not None:
            selected_collateral_isin = (
                selected_money_market_instrument.isin
            )
            selected_collateral_currency = (
                selected_money_market_instrument.currency
            )

        if (
            compare_with_gc
            and gc_repo_rate_percent is not None
            and selected_collateral_isin is not None
            and selected_collateral_currency is not None
        ):
            input_capture_timestamp = (
                datetime.now(
                    timezone.utc
                )
            )

            if (
                quote_timestamp_mode
                == "Explicit quote time"
                and explicit_specific_quote_date
                is not None
                and explicit_specific_quote_time
                is not None
                and explicit_gc_quote_date
                is not None
                and explicit_gc_quote_time
                is not None
            ):
                specific_observation_timestamp = (
                    datetime.combine(
                        explicit_specific_quote_date,
                        explicit_specific_quote_time,
                        tzinfo=timezone.utc,
                    )
                )

                gc_observation_timestamp = (
                    datetime.combine(
                        explicit_gc_quote_date,
                        explicit_gc_quote_time,
                        tzinfo=timezone.utc,
                    )
                )

                if (
                    specific_observation_timestamp
                    > input_capture_timestamp
                    or gc_observation_timestamp
                    > input_capture_timestamp
                ):
                    raise RepoMarketStateValidationError(
                        "Explicit quote timestamps must not be later than "
                        "the current UTC input-capture time."
                    )

            else:
                specific_observation_timestamp = (
                    input_capture_timestamp
                )
                gc_observation_timestamp = (
                    input_capture_timestamp
                )

            structured_specific_quote = (
                SpecificRepoQuote(
                    isin=(
                        selected_collateral_isin
                    ),
                    currency=(
                        selected_collateral_currency
                    ),
                    repo_days=result.repo_days,
                    rate_percent=float(
                        repo_rate_percent
                    ),
                    quote_timestamp=(
                        specific_observation_timestamp
                    ),
                    source_name=(
                        specific_quote_source_name
                    ),
                    source_type=(
                        specific_quote_source_type
                    ),
                    venue=(
                        specific_quote_venue.strip()
                        or None
                    ),
                    clearing_type=(
                        specific_clearing_type
                    ),
                    counterparty_segment=(
                        specific_counterparty_segment
                    ),
                )
            )

            structured_gc_reference = (
                GCReference(
                    currency=(
                        selected_collateral_currency
                    ),
                    repo_days=result.repo_days,
                    rate_percent=float(
                        gc_repo_rate_percent
                    ),
                    quote_timestamp=(
                        gc_observation_timestamp
                    ),
                    source_name=(
                        gc_quote_source_name
                    ),
                    source_type=(
                        gc_quote_source_type
                    ),
                    basket_name=(
                        gc_basket_name.strip()
                        or None
                    ),
                    venue=(
                        gc_quote_venue.strip()
                        or None
                    ),
                    clearing_type=(
                        gc_clearing_type
                    ),
                    counterparty_segment=(
                        gc_counterparty_segment
                    ),
                )
            )

            structured_specialness = (
                compare_specific_to_gc(
                    specific_quote=(
                        structured_specific_quote
                    ),
                    gc_reference=(
                        structured_gc_reference
                    ),
                    purchase_price_eur=(
                        result.purchase_price_eur
                    ),
                    day_count_basis=int(
                        day_count_basis
                    ),
                )
            )

    except (
        RepoValidationError,
        RepoMarketStateValidationError,
    ) as error:
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

    if (
        selected_coupon_instrument is not None
        and gc_repo_rate_percent is not None
    ):
        st.markdown(
            '<div class="section-label">Repo-adjusted carry vs GC</div>',
            unsafe_allow_html=True,
        )

        try:
            repo_adjusted_carry = (
                analyse_repo_adjusted_bond_carry(
                    bond=(
                        selected_coupon_instrument
                        .to_fixed_rate_bond()
                    ),
                    trade=trade,
                    gc_repo_rate_percent=float(
                        gc_repo_rate_percent
                    ),
                )
            )

        except RepoAdjustedCarryValidationError as error:
            st.warning(
                "RepoLens could not calculate the specific-versus-GC "
                "carry comparison for the selected inputs."
            )
            st.code(
                str(
                    error
                )
            )

        else:
            repo_adjusted_columns = st.columns(
                4
            )

            repo_adjusted_columns[0].metric(
                "Specific-funded carry",
                format_euro(
                    repo_adjusted_carry
                    .unchanged_yield_specific_pnl_eur,
                    decimals=2,
                ),
                delta=(
                    f"{repo_adjusted_carry.specific_repo_rate_percent:.4f}% "
                    "specific repo"
                ),
                delta_color="off",
                border=True,
            )

            repo_adjusted_columns[1].metric(
                "GC-funded carry",
                format_euro(
                    repo_adjusted_carry
                    .unchanged_yield_gc_pnl_eur,
                    decimals=2,
                ),
                delta=(
                    f"{repo_adjusted_carry.gc_repo_rate_percent:.4f}% "
                    "GC reference"
                ),
                delta_color="off",
                border=True,
            )

            repo_adjusted_columns[2].metric(
                "Funding edge vs GC",
                format_euro(
                    repo_adjusted_carry
                    .financing_advantage_vs_gc_eur,
                    decimals=2,
                ),
                delta=(
                    f"{repo_adjusted_carry.specialness_bp:+.2f} bp "
                    "GC minus specific"
                ),
                delta_color="off",
                border=True,
            )

            repo_adjusted_columns[3].metric(
                "Breakeven-yield advantage",
                (
                    f"{repo_adjusted_carry.breakeven_yield_advantage_bp:+.2f} bp"
                    if (
                        repo_adjusted_carry
                        .breakeven_yield_advantage_bp
                        is not None
                    )
                    else "N/A"
                ),
                delta=(
                    "Extra adverse yield move absorbed by cheaper funding"
                    if (
                        repo_adjusted_carry
                        .breakeven_yield_advantage_bp
                        is not None
                    )
                    else "No valid paired breakeven solve"
                ),
                delta_color="off",
                border=True,
            )

            repo_adjusted_second_row = st.columns(
                4
            )

            repo_adjusted_second_row[0].metric(
                "Specific repo interest",
                format_euro(
                    repo_adjusted_carry
                    .specific_repo_interest_eur,
                    decimals=2,
                ),
                delta=f"{repo_adjusted_carry.repo_days} day horizon",
                delta_color="off",
                border=True,
            )

            repo_adjusted_second_row[1].metric(
                "GC repo interest",
                format_euro(
                    repo_adjusted_carry
                    .gc_repo_interest_eur,
                    decimals=2,
                ),
                delta="Same collateral and haircut",
                delta_color="off",
                border=True,
            )

            repo_adjusted_second_row[2].metric(
                "Specific breakeven move",
                (
                    f"{repo_adjusted_carry.specific_breakeven_yield_move_bp:+.2f} bp"
                    if (
                        repo_adjusted_carry
                        .specific_breakeven_yield_move_bp
                        is not None
                    )
                    else "N/A"
                ),
                delta=(
                    f"Exit yield "
                    f"{repo_adjusted_carry.specific_breakeven_exit_yield_percent:.3f}%"
                    if (
                        repo_adjusted_carry
                        .specific_breakeven_exit_yield_percent
                        is not None
                    )
                    else "No valid solve"
                ),
                delta_color="off",
                border=True,
            )

            repo_adjusted_second_row[3].metric(
                "GC breakeven move",
                (
                    f"{repo_adjusted_carry.gc_breakeven_yield_move_bp:+.2f} bp"
                    if (
                        repo_adjusted_carry
                        .gc_breakeven_yield_move_bp
                        is not None
                    )
                    else "N/A"
                ),
                delta=(
                    f"Exit yield "
                    f"{repo_adjusted_carry.gc_breakeven_exit_yield_percent:.3f}%"
                    if (
                        repo_adjusted_carry
                        .gc_breakeven_exit_yield_percent
                        is not None
                    )
                    else "No valid solve"
                ),
                delta_color="off",
                border=True,
            )

            repo_rate_scenario_frame = pd.DataFrame(
                [
                    {
                        "Repo shock (bp)": (
                            scenario.repo_rate_shock_bp
                        ),
                        "Specific repo (%)": (
                            scenario
                            .shocked_specific_repo_rate_percent
                        ),
                        "Specialness vs GC (bp)": (
                            scenario.specialness_vs_gc_bp
                        ),
                        "Unchanged-yield carry (€)": (
                            scenario.financing_adjusted_pnl_eur
                        ),
                        "Carry / €1m face (€)": (
                            scenario
                            .financing_adjusted_pnl_per_eur_1m_face
                        ),
                        "Funding edge vs GC (€)": (
                            scenario
                            .financing_advantage_vs_gc_eur
                        ),
                        "Funding edge / €1m (€)": (
                            scenario
                            .financing_advantage_vs_gc_per_eur_1m_face
                        ),
                    }
                    for scenario
                    in repo_adjusted_carry.repo_rate_scenarios
                ]
            )

            st.dataframe(
                repo_rate_scenario_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Repo shock (bp)": (
                        st.column_config.NumberColumn(
                            "Repo shock",
                            format="%+.0f bp",
                        )
                    ),
                    "Specific repo (%)": (
                        st.column_config.NumberColumn(
                            "Specific repo",
                            format="%.4f%%",
                        )
                    ),
                    "Specialness vs GC (bp)": (
                        st.column_config.NumberColumn(
                            "Specialness vs GC",
                            format="%+.2f bp",
                        )
                    ),
                    "Unchanged-yield carry (€)": (
                        st.column_config.NumberColumn(
                            "Unchanged-yield carry",
                            format="€%,.2f",
                        )
                    ),
                    "Carry / €1m face (€)": (
                        st.column_config.NumberColumn(
                            "Carry / €1m face",
                            format="€%,.2f",
                        )
                    ),
                    "Funding edge vs GC (€)": (
                        st.column_config.NumberColumn(
                            "Funding edge vs GC",
                            format="€%,.2f",
                        )
                    ),
                    "Funding edge / €1m (€)": (
                        st.column_config.NumberColumn(
                            "Funding edge / €1m",
                            format="€%,.2f",
                        )
                    ),
                },
            )

            st.caption(
                "Specific-funded and GC-funded carry use the same bond, "
                "price, accrued interest, haircut, dates and day-count basis. "
                "Only the financing rate changes. The repo-rate sensitivity "
                "table holds the cash-bond yield unchanged and shocks only the "
                "specific repo rate, isolating funding sensitivity. Transaction "
                "costs, fail charges, variation margin, haircut opportunity "
                "cost and future repo roll paths are excluded."
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

        displayed_specialness_bp = (
            structured_specialness.specialness_bp
            if structured_specialness is not None
            else result.specialness_bp
        )

        special_columns = st.columns(
            4
        )

        special_columns[0].metric(
            "Specific repo",
            f"{repo_rate_percent:.4f}%",
            border=True,
        )

        special_columns[1].metric(
            "GC reference",
            f"{gc_repo_rate_percent:.4f}%",
            border=True,
        )

        special_columns[2].metric(
            "Specialness",
            f"{displayed_specialness_bp:+,.2f} bp",
            delta="GC minus specific repo",
            delta_color="off",
            border=True,
        )

        displayed_financing_benefit = (
            structured_specialness.financing_benefit_vs_gc_eur
            if (
                structured_specialness is not None
                and structured_specialness
                .financing_benefit_vs_gc_eur
                is not None
            )
            else result.financing_benefit_vs_gc_eur
        )

        special_columns[3].metric(
            "Financing benefit vs GC",
            format_euro(
                displayed_financing_benefit,
                decimals=2,
            ),
            delta="Over the selected repo term",
            delta_color="off",
            border=True,
        )

        if structured_specialness is not None:
            st.caption(
                f"Matched structured comparison · "
                f"{structured_specialness.isin} · "
                f"{structured_specialness.currency} · "
                f"{structured_specialness.repo_days}-day term · "
                f"specific source: "
                f"{structured_specialness.specific_source_name} · "
                f"GC source: "
                f"{structured_specialness.gc_source_name}"
            )

            timestamp_columns = st.columns(
                3
            )

            timestamp_columns[0].metric(
                "Specific quote time",
                (
                    structured_specialness
                    .specific_quote_timestamp
                    .strftime("%Y-%m-%d %H:%M UTC")
                ),
                border=True,
            )

            timestamp_columns[1].metric(
                "GC quote time",
                (
                    structured_specialness
                    .gc_quote_timestamp
                    .strftime("%Y-%m-%d %H:%M UTC")
                ),
                border=True,
            )

            timestamp_columns[2].metric(
                "Quote time gap",
                (
                    f"{structured_specialness.quote_time_difference_seconds:,.0f} sec"
                ),
                delta=(
                    quote_timestamp_mode
                ),
                delta_color="off",
                border=True,
            )

            comparison_context = (
                structured_specialness.comparison_context
            )

            if comparison_context is not None:
                st.markdown(
                    '<div class="section-label">Comparison quality</div>',
                    unsafe_allow_html=True,
                )

                comparison_columns = st.columns(4)

                def context_label(
                    value: bool | None,
                ) -> str:
                    if value is True:
                        return "Matched"
                    if value is False:
                        return "Different"
                    return "Unknown"

                comparison_columns[0].metric(
                    "Venue",
                    context_label(
                        comparison_context.same_venue
                    ),
                    border=True,
                )

                comparison_columns[1].metric(
                    "Clearing",
                    context_label(
                        comparison_context.same_clearing_type
                    ),
                    border=True,
                )

                comparison_columns[2].metric(
                    "Market segment",
                    context_label(
                        comparison_context.same_counterparty_segment
                    ),
                    border=True,
                )

                comparison_columns[3].metric(
                    "GC basket",
                    (
                        "Identified"
                        if comparison_context.gc_basket_identified
                        else "Unspecified"
                    ),
                    border=True,
                )

                if comparison_context.is_fully_context_matched:
                    st.success(
                        "The quote pair is fully matched on the optional "
                        "market-context dimensions captured by RepoLens."
                    )
                else:
                    st.warning(
                        "Specialness is arithmetically valid, but the quote "
                        "pair is not fully context-matched."
                    )

                    for warning in comparison_context.warnings:
                        st.caption(
                            f"• {warning}"
                        )

                st.caption(
                    "Currency and repo term remain hard matching requirements. "
                    "Venue, clearing type, market segment and GC basket identity "
                    "are comparison-quality dimensions."
                )

            st.markdown(
                '<div class="section-label">Historical specialness context</div>',
                unsafe_allow_html=True,
            )

            try:
                stored_specialness_records = (
                    load_repo_specialness_records()
                )

                matched_history = (
                    history_observations_for_market(
                        records=(
                            stored_specialness_records
                        ),
                        isin=(
                            structured_specialness.isin
                        ),
                        currency=(
                            structured_specialness.currency
                        ),
                        repo_days=(
                            structured_specialness.repo_days
                        ),
                        before_timestamp=(
                            structured_specialness
                            .specific_quote_timestamp
                        ),
                    )
                )

                historical_analysis = None

                if matched_history:
                    historical_analysis = (
                        analyse_specialness_history(
                            historical_observations=(
                                matched_history
                            ),
                            current_observation=(
                                observation_from_result(
                                    structured_specialness
                                )
                            ),
                        )
                    )

            except (
                RepoSpecialnessStoreError,
                RepoSpecialnessHistoryValidationError,
                OSError,
            ) as error:
                st.warning(
                    "Historical specialness context is unavailable because "
                    "RepoLens could not read or validate the saved observation history."
                )
                st.code(
                    str(
                        error
                    )
                )

            else:
                if historical_analysis is None:
                    st.info(
                        "No earlier saved observations match this exact "
                        "ISIN, currency and repo term yet. Save clean market "
                        "observations over time to build instrument-specific context."
                    )

                else:
                    history_columns = st.columns(
                        5
                    )

                    history_columns[0].metric(
                        "Historical median",
                        format_optional_bp(
                            historical_analysis
                            .historical_median_bp
                        ),
                        delta=(
                            f"{historical_analysis.historical_observation_count} "
                            "matched observations"
                        ),
                        delta_color="off",
                        border=True,
                    )

                    history_columns[1].metric(
                        "Current percentile",
                        (
                            f"{historical_analysis.historical_percentile:.1f}%"
                        ),
                        delta=(
                            "Empirical rank vs saved history"
                        ),
                        delta_color="off",
                        border=True,
                    )

                    history_columns[2].metric(
                        "Z-score",
                        format_optional_z_score(
                            historical_analysis.z_score
                        ),
                        delta=(
                            "Vs historical mean / volatility"
                            if historical_analysis.z_score
                            is not None
                            else "Historical volatility is zero"
                        ),
                        delta_color="off",
                        border=True,
                    )

                    history_columns[3].metric(
                        "Change vs previous",
                        format_optional_bp(
                            historical_analysis
                            .change_vs_previous_bp
                        ),
                        delta=(
                            historical_analysis
                            .previous_timestamp
                            .strftime("%Y-%m-%d %H:%M UTC")
                            if historical_analysis
                            .previous_timestamp
                            is not None
                            else "No previous observation"
                        ),
                        delta_color="off",
                        border=True,
                    )

                    history_columns[4].metric(
                        "Positive specialness share",
                        (
                            f"{historical_analysis.positive_specialness_share_percent:.1f}%"
                        ),
                        delta="Historical observations > 0 bp",
                        delta_color="off",
                        border=True,
                    )

                    st.caption(
                        "Historical distribution · "
                        f"mean "
                        f"{historical_analysis.historical_mean_bp:+.2f} bp · "
                        f"range "
                        f"{historical_analysis.historical_min_bp:+.2f} to "
                        f"{historical_analysis.historical_max_bp:+.2f} bp · "
                        f"population σ "
                        f"{historical_analysis.historical_std_bp:.2f} bp. "
                        "RepoLens does not apply a universal threshold to label "
                        "an issue special; these metrics provide matched historical context."
                    )

            with st.container(
                border=True
            ):
                save_left, save_right = st.columns(
                    [
                        3,
                        1,
                    ],
                    vertical_alignment="center",
                )

                with save_left:
                    st.markdown(
                        "**Historical specialness observation**"
                    )
                    st.caption(
                        "Nothing is written automatically. Save only when "
                        "the displayed specific-repo and GC inputs represent "
                        "an observation you want included in RepoLens history."
                    )

                with save_right:
                    save_market_observation = st.button(
                        "Save market observation",
                        key="repo_save_market_observation",
                        type="primary",
                        width="stretch",
                    )

                if save_market_observation:
                    if (
                        structured_specific_quote is None
                        or structured_gc_reference is None
                    ):
                        st.error(
                            "RepoLens could not reconstruct the structured "
                            "quote pair for persistence."
                        )
                    else:
                        try:
                            stored_record = (
                                stored_record_from_market_state(
                                    specific_quote=(
                                        structured_specific_quote
                                    ),
                                    gc_reference=(
                                        structured_gc_reference
                                    ),
                                    result=(
                                        structured_specialness
                                    ),
                                )
                            )

                            append_repo_specialness_record(
                                stored_record
                            )

                        except (
                            RepoSpecialnessStoreError,
                            OSError,
                        ) as error:
                            st.error(
                                "RepoLens could not save this market observation."
                            )
                            st.code(
                                str(
                                    error
                                )
                            )

                        else:
                            st.success(
                                "Market observation saved to "
                                "data/market/repo_specialness_history.csv."
                            )
                            st.caption(
                                f"Saved {stored_record.isin} · "
                                f"{stored_record.currency} · "
                                f"{stored_record.repo_days}-day · "
                                f"{stored_record.specialness_bp:+.2f} bp "
                                "specialness."
                            )

        else:
            st.caption(
                "Manual collateral has no structured RepoLens ISIN/currency "
                "identity, so this view uses the transaction calculator's "
                "GC-minus-specific calculation only."
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

            When a GC reference is supplied for coupon collateral, RepoLens
            also compares unchanged-yield carry under the specific repo rate
            with the same bond funded at GC. The difference isolates the
            financing advantage or disadvantage of the specific collateral.
            Repo-rate shocks hold the cash-bond yield unchanged so funding
            sensitivity is not mixed with outright duration risk.

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
            explicit **desk / broker inputs**. For catalogue collateral,
            RepoLens also records user-supplied GC and specific-quote
            provenance and requires the comparison to match on currency
            and repo term.

            Historical repo observations are persisted only when the user
            explicitly selects **Save market observation**. Streamlit reruns
            do not automatically write market history. Quote timestamps can
            use the current UTC input-capture time or explicit UTC observation
            times for the specific and GC legs. Historical context is matched
            strictly by ISIN, currency and repo term; the current observation
            is excluded from its own historical distribution.

            Venue, clearing type, market segment and GC basket identity are
            stored and surfaced as comparison-quality metadata. RepoLens flags
            context mismatches instead of treating different market settings
            as automatically equivalent.

            Schedule-derived accrued interest, remaining-maturity buckets,
            cash, repo-interest, specialness, pull-to-par, financed carry,
            specific-versus-GC carry, funding-edge and breakeven-yield
            comparisons, and scenario P&L are **RepoLens-derived analytics**.

            Manual collateral mode remains available and unchanged for
            instruments outside the RepoLens reference universe.

            This is an analytical calculator, not a trade confirmation,
            settlement, margin-maintenance or regulatory reporting system.
            """
        )


main()