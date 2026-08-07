from __future__ import annotations

import hmac

import streamlit as st


APP_TITLE = "RepoLens"

APP_SUBTITLE = (
    "European Repo, Sovereign Relative Value "
    "and Event Risk Intelligence"
)


st.set_page_config(
    page_title="RepoLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_css() -> None:
    """
    Apply shared RepoLens styling across every application page.
    """
    st.markdown(
        """
        <style>
        /* ---------------------------------------------------------
           APPLICATION SHELL
        --------------------------------------------------------- */

        .stApp {
            font-feature-settings:
                "kern" 1,
                "liga" 1;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1.35rem;
            padding-bottom: 4rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }


        /* ---------------------------------------------------------
           SIDEBAR
        --------------------------------------------------------- */

        section[data-testid="stSidebar"] {
            width: 420px !important;
            min-width: 420px !important;
            border-right:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.18
                );
        }

        section[data-testid="stSidebar"]
        > div {
            width: 420px !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarContent"] {
            padding-top: 1rem;
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }

        section[data-testid="stSidebar"]
        h2 {
            font-size: 1.55rem;
            font-weight: 780;
            letter-spacing: -0.025em;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"]
        p {
            line-height: 1.5;
        }


        /* ---------------------------------------------------------
           SIDEBAR NAVIGATION
        --------------------------------------------------------- */

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] {
            margin-bottom: 0.75rem;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"]
        a {
            border-radius: 0.6rem;
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"]
        a:hover {
            background:
                rgba(
                    128,
                    128,
                    128,
                    0.10
                );
        }


        /* ---------------------------------------------------------
           FORM CONTROLS
        --------------------------------------------------------- */

        section[data-testid="stSidebar"]
        [data-testid="stSelectbox"],
        section[data-testid="stSidebar"]
        [data-testid="stNumberInput"],
        section[data-testid="stSidebar"]
        [data-testid="stTextInput"],
        section[data-testid="stSidebar"]
        [data-testid="stDateInput"] {
            margin-bottom: 0.65rem;
        }

        div[data-baseweb="select"] > div {
            min-height: 2.85rem;
            border-radius: 0.65rem;
        }

        div[data-baseweb="input"] {
            border-radius: 0.65rem;
        }

        ul[role="listbox"] {
            max-width: 600px !important;
        }

        ul[role="listbox"] li {
            min-height: 2.7rem;
            white-space: normal !important;
            line-height: 1.35;
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
        }

        div[role="radiogroup"] {
            gap: 0.35rem;
        }


        /* ---------------------------------------------------------
           EXPANDERS
        --------------------------------------------------------- */

        [data-testid="stExpander"] {
            border:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.22
                );
            border-radius: 0.75rem;
            overflow: hidden;
        }

        [data-testid="stExpander"]
        details summary {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }


        /* ---------------------------------------------------------
           BUTTONS
        --------------------------------------------------------- */

        .stButton > button {
            min-height: 2.65rem;
            border-radius: 0.65rem;
            font-weight: 650;
        }

        section[data-testid="stSidebar"]
        .stButton > button {
            min-height: 2.75rem;
        }


        /* ---------------------------------------------------------
           METRICS
        --------------------------------------------------------- */

        [data-testid="stMetric"] {
            border:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.20
                );
            border-radius: 0.8rem;
            padding: 0.95rem 1rem;
            background:
                rgba(
                    128,
                    128,
                    128,
                    0.035
                );
            min-height: 7rem;
        }

        [data-testid="stMetric"]:hover {
            border-color:
                rgba(
                    128,
                    128,
                    128,
                    0.34
                );
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.015em;
        }

        [data-testid="stMetricValue"] {
            font-weight: 720;
            letter-spacing: -0.025em;
        }


        /* ---------------------------------------------------------
           REPOLENS TYPOGRAPHY
        --------------------------------------------------------- */

        .repolens-kicker {
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            opacity: 0.58;
            margin-bottom: 0.4rem;
        }

        .repolens-title {
            font-size: 2.35rem;
            font-weight: 780;
            letter-spacing: -0.035em;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }

        .repolens-subtitle {
            max-width: 950px;
            font-size: 1rem;
            line-height: 1.55;
            opacity: 0.68;
            margin-bottom: 1.5rem;
        }

        .section-label {
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.58;
            margin-top: 0.8rem;
            margin-bottom: 0.4rem;
        }


        /* ---------------------------------------------------------
           STATUS PANELS
        --------------------------------------------------------- */

        .status-box {
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
            margin-top: 0.5rem;
            margin-bottom: 1.25rem;
            border:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.22
                );
            line-height: 1.55;
        }

        .status-normal {
            background:
                rgba(
                    34,
                    139,
                    94,
                    0.10
                );
        }

        .status-monitor {
            background:
                rgba(
                    214,
                    149,
                    0,
                    0.12
                );
        }

        .status-high {
            background:
                rgba(
                    200,
                    55,
                    55,
                    0.12
                );
        }

        .status-event {
            background:
                rgba(
                    70,
                    105,
                    190,
                    0.12
                );
        }


        /* ---------------------------------------------------------
           RISK / INFORMATION CARDS
        --------------------------------------------------------- */

        .risk-card {
            border:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.20
                );
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            background:
                rgba(
                    128,
                    128,
                    128,
                    0.035
                );
        }

        .risk-card-title {
            font-size: 0.74rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.62;
            margin-bottom: 0.4rem;
        }

        .risk-card-value {
            font-size: 1.5rem;
            font-weight: 740;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }

        .small-note {
            font-size: 0.82rem;
            line-height: 1.45;
            opacity: 0.66;
        }


        /* ---------------------------------------------------------
           DATAFRAMES
        --------------------------------------------------------- */

        [data-testid="stDataFrame"] {
            border-radius: 0.7rem;
            overflow: hidden;
        }


        /* ---------------------------------------------------------
           DIVIDERS
        --------------------------------------------------------- */

        hr {
            margin-top: 1.65rem !important;
            margin-bottom: 1.65rem !important;
            opacity: 0.45;
        }


        /* ---------------------------------------------------------
           SIDEBAR BRAND PANEL
        --------------------------------------------------------- */

        .repolens-sidebar-brand {
            border:
                1px solid rgba(
                    128,
                    128,
                    128,
                    0.18
                );
            border-radius: 0.8rem;
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
            background:
                rgba(
                    128,
                    128,
                    128,
                    0.025
                );
        }

        .repolens-sidebar-brand-name {
            font-size: 1.1rem;
            font-weight: 780;
            letter-spacing: -0.025em;
            margin-bottom: 0.25rem;
        }

        .repolens-sidebar-brand-subtitle {
            font-size: 0.78rem;
            line-height: 1.4;
            opacity: 0.62;
        }


        /* ---------------------------------------------------------
           RESPONSIVE
        --------------------------------------------------------- */

        @media (
            max-width: 900px
        ) {
            section[data-testid="stSidebar"] {
                width: 340px !important;
                min-width: 340px !important;
            }

            section[data-testid="stSidebar"]
            > div {
                width: 340px !important;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .repolens-title {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_optional_password() -> str | None:
    """
    Read the optional dashboard password from Streamlit secrets.
    """
    try:
        configured_password = (
            st.secrets.get(
                "APP_PASSWORD"
            )
        )
    except FileNotFoundError:
        return None

    if configured_password is None:
        return None

    password_text = str(
        configured_password
    ).strip()

    return (
        password_text
        or None
    )


def require_password() -> None:
    """
    Require authentication only when a password is configured.
    """
    configured_password = (
        read_optional_password()
    )

    if configured_password is None:
        return

    if st.session_state.get(
        "repolens_authenticated",
        False,
    ):
        return

    st.markdown(
        """
        <div class="repolens-kicker">
            Restricted research dashboard
        </div>
        <div class="repolens-title">
            RepoLens
        </div>
        <div class="repolens-subtitle">
            European repo, sovereign relative-value
            and event-risk intelligence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    entered_password = st.text_input(
        "Password",
        type="password",
        key="repolens_password_input",
    )

    if st.button(
        "Open RepoLens",
        type="primary",
        width="stretch",
    ):
        if hmac.compare_digest(
            entered_password,
            configured_password,
        ):
            st.session_state[
                "repolens_authenticated"
            ] = True

            st.rerun()

        st.error(
            "Incorrect password."
        )

    st.stop()


def render_shared_sidebar() -> None:
    """
    Render shared sidebar context and application controls.
    """
    st.sidebar.markdown(
        f"""
        <div class="repolens-sidebar-brand">
            <div class="repolens-sidebar-brand-name">
                {APP_TITLE}
            </div>
            <div class="repolens-sidebar-brand-subtitle">
                {APP_SUBTITLE}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "**Research scope**"
    )

    st.sidebar.caption(
        "Euro money-market funding, policy transmission, "
        "sovereign valuation, relative value and portfolio risk."
    )

    st.sidebar.markdown(
        "**Primary official sources**"
    )

    st.sidebar.caption(
        "European Central Bank, Deutsche Bundesbank "
        "and sovereign reference-data sources."
    )

    st.sidebar.markdown(
        "**Data classification**"
    )

    st.sidebar.caption(
        "Official observations, explicit desk inputs "
        "and RepoLens-derived analytics are kept separate."
    )

    configured_password = (
        read_optional_password()
    )

    if configured_password is not None:
        st.sidebar.divider()

        if st.sidebar.button(
            "Lock dashboard",
            width="stretch",
        ):
            st.session_state[
                "repolens_authenticated"
            ] = False

            st.rerun()


def main() -> None:
    """
    Run the shared RepoLens application router.
    """
    apply_custom_css()
    require_password()

    pages = {
        "Euro Funding": [
            st.Page(
                "app_pages/morning_sheet_page.py",
                title="Morning Sheet",
                icon=":material/space_dashboard:",
                default=True,
            ),
            st.Page(
                "app_pages/risk_monitor_page.py",
                title="Risk Monitor",
                icon=":material/shield:",
            ),
        ],
        "Sovereign Markets": [
            st.Page(
                "app_pages/sovereign_bond_terminal_page.py",
                title="Bond Terminal",
                icon=":material/account_balance:",
            ),
            st.Page(
                "app_pages/sovereign_relative_value_page.py",
                title="Relative Value Monitor",
                icon=":material/compare_arrows:",
            ),
            st.Page(
                "app_pages/sovereign_portfolio_page.py",
                title="Portfolio Risk Book",
                icon=":material/menu_book:",
            ),
        ],
    }

    selected_page = st.navigation(
        pages,
        position="sidebar",
    )

    render_shared_sidebar()

    selected_page.run()


if __name__ == "__main__":
    main()