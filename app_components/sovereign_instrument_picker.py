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
    Convert catalogue benchmark status into readable desk terminology.
    """
    labels = {
        BenchmarkStatus.PRIMARY_BENCHMARK: "Primary benchmark",
        BenchmarkStatus.REFERENCE_BOND: "Reference bond",
        BenchmarkStatus.OFF_THE_RUN: "Off-the-run",
    }

    return labels[status]


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
    Format a date for compact instrument metadata.
    """
    return value.strftime("%d %b %Y")


def default_instrument() -> SovereignInstrument:
    """
    Return Germany 10Y as the default terminal instrument when available.
    """
    instruments = all_instruments()

    if not instruments:
        raise RuntimeError(
            "The sovereign instrument catalogue is empty."
        )

    for instrument in instruments:
        if (
            instrument.country
            == SovereignCountry.GERMANY
            and instrument.benchmark_tenor_years
            == 10
        ):
            return instrument

    return instruments[0]


def resolve_selected_instrument() -> SovereignInstrument:
    """
    Resolve the currently selected catalogue instrument.
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

    if selected_isin in instruments_by_isin:
        return instruments_by_isin[selected_isin]

    instrument = default_instrument()

    st.session_state[
        SELECTED_INSTRUMENT_SESSION_KEY
    ] = instrument.isin

    return instrument


def instrument_search_text(
    instrument: SovereignInstrument,
) -> str:
    """
    Return searchable catalogue text for one instrument.
    """
    record = master_record_by_isin(
        instrument.isin
    )

    values = [
        instrument.isin,
        instrument.display_name,
        instrument.country.value,
        instrument.country_code,
        instrument.security_type.value,
        str(instrument.benchmark_tenor_years),
        f"{instrument.annual_coupon_rate * 100.0:.3f}",
        benchmark_status_label(
            record.benchmark_status
        ),
    ]

    return " ".join(values).lower()


def filtered_instruments(
    country_filter: str,
    security_type_filter: str,
    search_text: str,
) -> tuple[SovereignInstrument, ...]:
    """
    Filter the full sovereign catalogue.
    """
    normalised_search = (
        search_text
        .strip()
        .lower()
    )

    selected = tuple(
        instrument
        for instrument in all_instruments()
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
    Render the active instrument without Markdown interpreting HTML
    indentation as a code block.
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

    html = (
        '<div class="instrument-selection-card">'
        '<div class="instrument-selection-kicker">'
        'Selected instrument'
        '</div>'
        '<div class="instrument-selection-name">'
        f'{instrument.display_name}'
        '</div>'
        '<div class="instrument-selection-meta">'
        f'{instrument.isin} · '
        f'{instrument.benchmark_tenor_years}Y sector · '
        f'{status}'
        '</div>'
        '<div class="instrument-selection-detail">'
        f'Coupon {coupon_percent:.3f}% · '
        f'Maturity {compact_date(instrument.maturity_date)} · '
        f'{coupon_frequency_label(instrument.coupon_frequency)}'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_instrument_option(
    instrument: SovereignInstrument,
) -> None:
    """
    Render one bond inside the expanded browser.
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

    html = (
        '<div class="instrument-browser-row">'
        '<div class="instrument-browser-country">'
        f'{instrument.country_code} · '
        f'{instrument.benchmark_tenor_years}Y · '
        f'{instrument.security_type.value}'
        '</div>'
        '<div class="instrument-browser-name">'
        f'{instrument.display_name}'
        '</div>'
        '<div class="instrument-browser-meta">'
        f'{instrument.isin} · '
        f'Coupon {coupon_percent:.3f}% · '
        f'Maturity {compact_date(instrument.maturity_date)} · '
        f'{status}'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


@st.dialog(
    "Sovereign instrument universe",
    width="large",
)
def show_instrument_browser() -> None:
    """
    Render the expanded searchable instrument universe.
    """
    instruments = all_instruments()

    if not instruments:
        st.error(
            "No sovereign instruments are available."
        )
        return

    st.caption(
        "Browse the RepoLens sovereign universe and select "
        "the instrument to load into the Bond Terminal."
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
                instrument.security_type.value
                for instrument
                in instruments
            }
        ),
    ]

    country_column, security_column = st.columns(2)

    with country_column:
        country_filter = st.selectbox(
            "Country",
            options=country_options,
            key=PICKER_COUNTRY_SESSION_KEY,
        )

    with security_column:
        security_type_filter = st.selectbox(
            "Security type",
            options=security_type_options,
            key=PICKER_SECURITY_TYPE_SESSION_KEY,
        )

    search_text = st.text_input(
        "Search instrument",
        placeholder=(
            "Search name, ISIN, tenor, country or security type"
        ),
        key="repolens_instrument_picker_search",
    )

    matches = filtered_instruments(
        country_filter=country_filter,
        security_type_filter=security_type_filter,
        search_text=search_text,
    )

    st.caption(
        f"{len(matches)} "
        f"{'instrument' if len(matches) == 1 else 'instruments'}"
    )

    if not matches:
        st.info(
            "No instruments match the current filters."
        )
        return

    selected_isin = st.session_state.get(
        SELECTED_INSTRUMENT_SESSION_KEY
    )

    for instrument in matches:
        information_column, action_column = st.columns(
            [5, 1],
            vertical_alignment="center",
        )

        with information_column:
            render_instrument_option(
                instrument
            )

        with action_column:
            is_selected = (
                instrument.isin
                == selected_isin
            )

            if is_selected:
                st.button(
                    "Selected",
                    key=(
                        "repolens_selected_"
                        f"{instrument.isin}"
                    ),
                    disabled=True,
                    width="stretch",
                )

            else:
                if st.button(
                    "Select",
                    key=(
                        "repolens_select_"
                        f"{instrument.isin}"
                    ),
                    type="primary",
                    width="stretch",
                ):
                    st.session_state[
                        SELECTED_INSTRUMENT_SESSION_KEY
                    ] = instrument.isin

                    st.rerun()

        st.divider()


def sovereign_instrument_picker() -> SovereignInstrument:
    """
    Render the compact Bond Terminal instrument selector.

    The complete catalogue opens in a large dialog rather than being
    forced into a narrow sidebar selectbox.
    """
    instrument = resolve_selected_instrument()

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