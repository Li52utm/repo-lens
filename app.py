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
    Apply shared styling across every RepoLens page.
    """
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
            background: rgba(128, 128, 128, 0.035);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.82rem;
        }

        .repolens-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.62;
            margin-bottom: 0.35rem;
        }

        .repolens-title {
            font-size: 2.15rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .repolens-subtitle {
            font-size: 1rem;
            opacity: 0.70;
            margin-bottom: 1.25rem;
        }

        .section-label {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            opacity: 0.60;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
        }

        .status-box {
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
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
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: rgba(128, 128, 128, 0.035);
        }

        .risk-card-title {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.35rem;
        }

        .risk-card-value {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .small-note {
            font-size: 0.82rem;
            opacity: 0.67;
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
        configured_password = st.secrets.get(
            "APP_PASSWORD"
        )
    except FileNotFoundError:
        return None

    if configured_password is None:
        return None

    password_text = str(
        configured_password
    ).strip()

    return password_text or None


def require_password() -> None:
    """
    Require authentication only when a password is configured.
    """
    configured_password = read_optional_password()

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
            Enter the dashboard password to continue.
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
    Render shared sidebar context and dashboard controls.
    """
    st.sidebar.markdown(
        f"## {APP_TITLE}"
    )

    st.sidebar.caption(
        APP_SUBTITLE
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        "**Research scope**"
    )

    st.sidebar.write(
        "Euro money-market funding, policy transmission, "
        "sovereign valuation and relative-value intelligence."
    )

    st.sidebar.markdown(
        "**Primary data sources**"
    )

    st.sidebar.write(
        "European Central Bank and Deutsche Bundesbank"
    )

    st.sidebar.markdown(
        "**Classification**"
    )

    st.sidebar.write(
        "Official and desk-supplied inputs with "
        "RepoLens-derived analytics"
    )

    configured_password = read_optional_password()

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
    Run the shared RepoLens router.
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