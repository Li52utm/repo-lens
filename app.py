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
    Apply the shared RepoLens workstation shell.

    The shell deliberately avoids a fixed main-column max width so the
    analytics area reclaims the viewport when Streamlit's sidebar is
    collapsed.
    """
    st.markdown(
        """
        <style>
        :root {
            --repolens-sidebar-width: 360px;
            --repolens-border: rgba(148, 163, 184, 0.18);
            --repolens-border-strong: rgba(148, 163, 184, 0.30);
            --repolens-panel: rgba(15, 23, 42, 0.34);
            --repolens-panel-hover: rgba(30, 41, 59, 0.46);
            --repolens-muted: rgba(226, 232, 240, 0.62);
        }

        /* ---------------------------------------------------------
           APPLICATION SHELL
        --------------------------------------------------------- */

        .stApp {
            font-feature-settings: "kern" 1, "liga" 1;
        }

        [data-testid="stAppViewContainer"] {
            width: 100%;
        }

        [data-testid="stMain"] {
            min-width: 0 !important;
            width: 100% !important;
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            width: 100% !important;
            max-width: none !important;
            padding-top: 1.15rem !important;
            padding-bottom: 4rem !important;
            padding-left: clamp(1rem, 1.6vw, 2rem) !important;
            padding-right: clamp(1rem, 1.6vw, 2rem) !important;
        }

        /* ---------------------------------------------------------
           SIDEBAR
        --------------------------------------------------------- */

        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: var(--repolens-sidebar-width) !important;
            min-width: var(--repolens-sidebar-width) !important;
            max-width: var(--repolens-sidebar-width) !important;
            border-right: 1px solid var(--repolens-border);
        }

        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            width: var(--repolens-sidebar-width) !important;
            min-width: var(--repolens-sidebar-width) !important;
            max-width: var(--repolens-sidebar-width) !important;
        }

        /*
        Do not assign a fixed width to the collapsed sidebar.
        Streamlit can then reduce it to its own small navigation rail
        instead of leaving a 360-420px blank gutter.
        */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 0 !important;
            max-width: none !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarContent"] {
            padding-top: 0.85rem;
            padding-left: 0.15rem;
            padding-right: 0.15rem;
        }

        section[data-testid="stSidebar"] p {
            line-height: 1.42;
        }

        /* ---------------------------------------------------------
           SIDEBAR NAVIGATION
        --------------------------------------------------------- */

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] {
            margin-bottom: 0.55rem;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a {
            border-radius: 0.55rem;
            margin: 0.1rem 0;
        }

        section[data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(148, 163, 184, 0.10);
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
            margin-bottom: 0.5rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] {
            border-radius: 0.6rem;
        }

        div[data-baseweb="select"] > div {
            min-height: 2.6rem;
        }

        ul[role="listbox"] {
            max-width: min(760px, 88vw) !important;
        }

        ul[role="listbox"] li {
            min-height: 2.55rem;
            white-space: normal !important;
            line-height: 1.3;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }

        div[role="radiogroup"] {
            gap: 0.25rem;
        }

        /* ---------------------------------------------------------
           EXPANDERS / BUTTONS
        --------------------------------------------------------- */

        [data-testid="stExpander"] {
            border: 1px solid var(--repolens-border);
            border-radius: 0.7rem;
            overflow: hidden;
        }

        .stButton > button {
            min-height: 2.55rem;
            border-radius: 0.6rem;
            font-weight: 650;
        }

        /* ---------------------------------------------------------
           METRICS
        --------------------------------------------------------- */

        [data-testid="stMetric"] {
            border: 1px solid var(--repolens-border);
            border-radius: 0.72rem;
            padding: 0.85rem 0.9rem;
            background: var(--repolens-panel);
            min-height: 6.6rem;
            overflow: visible !important;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--repolens-border-strong);
            background: var(--repolens-panel-hover);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.76rem;
            font-weight: 650;
            letter-spacing: 0.01em;
        }

        [data-testid="stMetricValue"] {
            font-size: clamp(1.45rem, 1.85vw, 2.2rem) !important;
            font-weight: 720;
            letter-spacing: -0.025em;
            overflow: visible !important;
        }

        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] p {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: nowrap !important;
        }

        [data-testid="stMetricDelta"] {
            max-width: 100%;
        }

        /* ---------------------------------------------------------
           REPOLENS TYPOGRAPHY
        --------------------------------------------------------- */

        .repolens-kicker,
        .section-label {
            font-size: 0.71rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.58;
        }

        .repolens-kicker {
            margin-bottom: 0.35rem;
        }

        .section-label {
            margin-top: 0.7rem;
            margin-bottom: 0.35rem;
        }

        .repolens-title {
            font-size: clamp(2rem, 2.5vw, 2.65rem);
            font-weight: 780;
            letter-spacing: -0.035em;
            line-height: 1.03;
            margin-bottom: 0.3rem;
        }

        .repolens-subtitle {
            max-width: 1100px;
            font-size: 0.98rem;
            line-height: 1.5;
            opacity: 0.68;
            margin-bottom: 1.35rem;
        }

        /* ---------------------------------------------------------
           STATUS / INFORMATION PANELS
        --------------------------------------------------------- */

        .status-box,
        .risk-card,
        .repolens-sidebar-brand {
            border: 1px solid var(--repolens-border);
            border-radius: 0.72rem;
        }

        .status-box {
            padding: 0.9rem 1rem;
            margin-top: 0.45rem;
            margin-bottom: 1.1rem;
            line-height: 1.48;
        }

        .status-normal {
            background: rgba(34, 139, 94, 0.10);
        }

        .status-monitor {
            background: rgba(214, 149, 0, 0.12);
        }

        .status-high {
            background: rgba(200, 55, 55, 0.12);
        }

        .status-event {
            background: rgba(70, 105, 190, 0.12);
        }

        .risk-card {
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            background: var(--repolens-panel);
        }

        .risk-card-title {
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.62;
            margin-bottom: 0.35rem;
        }

        .risk-card-value {
            font-size: 1.45rem;
            font-weight: 740;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .small-note {
            font-size: 0.8rem;
            line-height: 1.42;
            opacity: 0.66;
        }

        /* ---------------------------------------------------------
           TABLES / DIVIDERS
        --------------------------------------------------------- */

        [data-testid="stDataFrame"] {
            border-radius: 0.65rem;
            overflow: hidden;
        }

        hr {
            margin-top: 1.45rem !important;
            margin-bottom: 1.45rem !important;
            opacity: 0.4;
        }

        /* ---------------------------------------------------------
           COMPACT SIDEBAR BRAND
        --------------------------------------------------------- */

        .repolens-sidebar-brand {
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.65rem;
            background: rgba(15, 23, 42, 0.24);
        }

        .repolens-sidebar-brand-name {
            font-size: 1.02rem;
            font-weight: 780;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }

        .repolens-sidebar-brand-subtitle {
            font-size: 0.73rem;
            line-height: 1.35;
            opacity: 0.58;
        }

        /* ---------------------------------------------------------
           RESPONSIVE
        --------------------------------------------------------- */

        @media (max-width: 1100px) {
            :root {
                --repolens-sidebar-width: 330px;
            }

            [data-testid="stMainBlockContainer"],
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
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
    Render compact shared context without crowding page controls.
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

    st.sidebar.caption(
        "Official market observations, explicit desk/broker inputs "
        "and RepoLens-derived analytics are labelled separately."
    )

    configured_password = read_optional_password()

    if configured_password is not None:
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
                "app_pages/repo_calculator_page.py",
                title="Repo Calculator",
                icon=":material/calculate:",
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