from __future__ import annotations

from datetime import date

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
)


SELECTED_INSTRUMENT_SESSION_KEY = (
    "repolens_selected_sovereign_instrument_isin"
)

PICKER_COUNTRY_SESSION_KEY = (
    "repolens_instrument_picker_country"
)

PICKER_SECURITY_TYPE_SESSION_KEY = (
    "repolens_instrument_picker_security_type"
)


def benchmark_status_label(
    status: BenchmarkStatus,
) -> str:
    """
    Convert internal benchmark status into readable desk terminology.
    """
    labels = {
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

    return labels[
        status
    ]


def coupon_frequency_label(
    frequency: int,
) -> str:
    """
    Convert coupon frequency into readable text.
    """
    labels = {
        1: "Annual",
        2: "Semi-annual",
        4: "Quarterly",
    }

    return labels.get(
        frequency,
        f"{frequency} payments/year",
    )


def compact_date(
    value: date,
) -> str:
    """
    Format dates compactly for the instrument browser.
    """
    return value.strftime(
        "%d %b %Y"
    )


def default_instrument() -> SovereignInstrument:
    """
    Return the preferred initial RepoLens Bond Terminal instrument.

    Germany 10Y remains the default because the existing application
    has an approved exact German benchmark observation for that sector.
    """
    instruments = all_instruments()

    matches = tuple(
        instrument
        for instrument in instruments
        if (
            instrument.country
            == SovereignCountry.GERMANY
            and instrument.benchmark_tenor_years
            == 10
        )
    )

    if matches:
        return matches[0]

    if not instruments:
        raise RuntimeError(
            "The sovereign instrument catalogue is empty."
        )

    return instruments[0]


def resolve_selected_instrument() -> SovereignInstrument:
    """
    Resolve the instrument currently stored in Streamlit session state.
    """
    instruments = all_instruments()

    if not instruments:
        raise RuntimeError(
            "The sovereign instrument catalogue is empty."
        )

    instruments_by_isin = {
        instrument.isin: instrument
        for instrument in instruments
    }

    selected_isin = st.session_state.get(
        SELECTED_INSTRUMENT_SESSION_KEY
    )

    if (
        selected_isin is not None
        and selected_isin in instruments_by_isin
    ):
        return instruments_by_isin[
            selected_isin
        ]

    instrument = default_instrument()

    st.session_state[
        SELECTED_INSTRUMENT_SESSION_KEY
    ] = instrument.isin

    return instrument


def instrument_search_text(
    instrument: SovereignInstrument,
) -> str:
    """
    Create searchable text for one catalogue instrument.
    """
    record = master_record_by_isin(
        instrument.isin
    )

    return " ".join(
        [
            instrument.isin,
            instrument.display_name,
            instrument.country.value,
            instrument.country_code,
            instrument.security_type.value,
            str(
                instrument.benchmark_tenor_years
            ),
            str(
                instrument.annual_coupon_rate
                * 100.0
            ),
            benchmark_status_label(
                record.benchmark_status
            ),
        ]
    ).lower()


def filtered_instruments(
    country_filter: str,
    security_type_filter: str,
    search_text: str,
) -> tuple[
    SovereignInstrument,
    ...,
]:
    """
    Filter the complete catalogue for the large instrument browser.
    """
    instruments = all_instruments()

    normalised_search = (
        search_text
        .strip()
        .lower()
    )

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
                security_type_filter == "All"
                or instrument.security_type.value
                == security_type_filter
            )
            and (
                not normalised_search
                or normalised_search
                in instrument_search_text(
                    instrument
                )
            )
        )
    )

    return tuple(
        sorted(
            selected,
            key=lambda instrument: (
                instrument.country.value,
                instrument.maturity_date,
                instrument.isin,
            ),
        )
    )


def render_selected_instrument_card(
    instrument: SovereignInstrument,
) -> None:
    """
    Render the currently selected instrument in the sidebar.
    """
    record = master_record_by_isin(
        instrument.isin
    )

    status = benchmark_status_label(
        record.benchmark_status
    )

    coupon_percent = (
        instrument.annual_coupon_rate
        * 100.0
    )

    st.markdown(
        f"""
        <div class="instrument-selection-card">
            <div class="instrument-selection-kicker">
                Selected instrument
            </div>

            <div class="instrument-selection-name">
                {instrument.display_name}
            </div>

            <div class="instrument-selection-meta">
                {instrument.isin}
                &nbsp;·&nbsp;
                {instrument.benchmark_tenor_years}Y sector
                &nbsp;·&nbsp;
                {status}
            </div>

            <div class="instrument-selection-detail">
                Coupon {coupon_percent:.3f}%
                &nbsp;·&nbsp;
                Maturity {compact_date(instrument.maturity_date)}
                &nbsp;·&nbsp;
                {coupon_frequency_label(instrument.coupon_frequency)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_instrument_option(
    instrument: SovereignInstrument,
) -> None:
    """
    Render one readable instrument card inside the browser.
    """
    record = master_record_by_isin(
        instrument.isin
    )

    status = benchmark_status_label(
        record.benchmark_status
    )

    coupon_percent = (
        instrument.annual_coupon_rate
        * 100.0
    )

    st.markdown(
        f"""
        <div class="instrument-browser-row">
            <div class="instrument-browser-country">
                {instrument.country_code}
                &nbsp;·&nbsp;
                {instrument.benchmark_tenor_years}Y
                &nbsp;·&nbsp;
                {instrument.security_type.value}
            </div>

            <div class="instrument-browser-name">
                {instrument.display_name}
            </div>

            <div class="instrument-browser-meta">
                {instrument.isin}
                &nbsp;·&nbsp;
                Coupon {coupon_percent:.3f}%
                &nbsp;·&nbsp;
                Maturity {compact_date(instrument.maturity_date)}
                &nbsp;·&nbsp;
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog(
    "Sovereign instrument universe",
    width="large",
)
def show_instrument_browser() -> None:
    """
    Render a large searchable catalogue and allow one bond selection.
    """
    instruments = all_instruments()

    if not instruments:
        st.error(
            "No sovereign instruments are available."
        )

        return

    st.caption(
        "Browse RepoLens sovereign reference data. "
        "Selecting a bond changes the active Bond Terminal instrument."
    )

    country_options = [
        "All",
        *[
            country.value
            for country
            in SovereignCountry
        ],
    ]

    security_type_options = [
        "All",
        *sorted(
            {
                instrument
                .security_type
                .value
                for instrument
                in instruments
            }
        ),
    ]

    filter_column_one, filter_column_two = (
        st.columns(
            2
        )
    )

    with filter_column_one:
        country_filter = st.selectbox(
            "Country",
            options=country_options,
            key=PICKER_COUNTRY_SESSION_KEY,
        )

    with filter_column_two:
        security_type_filter = st.selectbox(
            "Security type",
            options=security_type_options,
            key=PICKER_SECURITY_TYPE_SESSION_KEY,
        )

    search_text = st.text_input(
        "Search",
        placeholder=(
            "Search ISIN, issuer, bond name, maturity sector..."
        ),
        key="repolens_instrument_picker_search",
    )

    matches = filtered_instruments(
        country_filter=country_filter,
        security_type_filter=(
            security_type_filter
        ),
        search_text=search_text,
    )

    st.caption(
        f"{len(matches)} instrument"
        f"{'' if len(matches) == 1 else 's'} shown"
    )

    if not matches:
        st.info(
            "No instruments match the current filters."
        )

        return

    currently_selected_isin = (
        st.session_state.get(
            SELECTED_INSTRUMENT_SESSION_KEY
        )
    )

    for instrument in matches:
        row_left, row_right = st.columns(
            [
                5,
                1,
            ],
            vertical_alignment="center",
        )

        with row_left:
            render_instrument_option(
                instrument
            )

        with row_right:
            is_selected = (
                instrument.isin
                == currently_selected_isin
            )

            button_label = (
                "Selected"
                if is_selected
                else "Select"
            )

            if st.button(
                button_label,
                key=(
                    "repolens_instrument_select_"
                    f"{instrument.isin}"
                ),
                disabled=is_selected,
                type=(
                    "secondary"
                    if is_selected
                    else "primary"
                ),
                width="stretch",
            ):
                st.session_state[
                    SELECTED_INSTRUMENT_SESSION_KEY
                ] = instrument.isin

                st.rerun()

        st.divider()


def sovereign_instrument_picker() -> SovereignInstrument:
    """
    Render the compact sidebar control and return the selected bond.

    Instrument browsing occurs inside a large Streamlit dialog so long
    sovereign bond names do not need to fit inside the sidebar.
    """
    instrument = (
        resolve_selected_instrument()
    )

    render_selected_instrument_card(
        instrument
    )

    if st.button(
        "Browse bond universe",
        icon=":material/search:",
        width="stretch",
        key="repolens_open_instrument_browser",
    ):
        show_instrument_browser()

    return instrument