"""LLMOps Studio Reusable UI Widgets.

Exposes standard Streamlit component functions to maintain consistent typography,
gradient styling, and future implementation alerts.
"""

import streamlit as st


def render_header(title: str, subtitle: str) -> None:
    """Renders the standard premium page header with gradient text.

    Args:
        title: Page main heading.
        subtitle: Sub-heading or context description.
    """
    st.markdown(
        f'<h1 class="gradient-text">{title}</h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color: #9CA3AF; font-size: 1.15rem; margin-top: -8px; margin-bottom: 24px;">{subtitle}</p>', 
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_future_phase_box(package_name: str, description: str) -> None:
    """Renders a styled card detailing future architectural integrations.

    Args:
        package_name: The backend package name (e.g. backend/training/).
        description: Explanatory text of how future logic links with current skeletons.
    """
    st.info(
        f"**Future Integration Layer**: `{package_name}`\n\n"
        f"{description}\n\n"
        f"In subsequent phases, frontend callbacks here will invoke services injected via "
        f"`backend/core/dependencies.py` to route payloads defined in `backend/schemas/` "
        f"to physical executor modules."
    )


def render_metric_card(title: str, value: str, trend: str = "", status: str = "active") -> None:
    """Displays a custom styled metric container block.

    Args:
        title: Metric label.
        value: Numeric or string representation.
        trend: Sub-metric helper context (e.g., '+12%').
        status: Color status classification ('active', 'success', 'pending').
    """
    badge_class = "badge-active"
    if status == "success":
        badge_class = "badge-success"
    elif status == "pending":
        badge_class = "badge-pending"

    st.markdown(
        f"""
        <div class="custom-card">
            <div class="custom-card-title">{title}</div>
            <div class="custom-card-value">{value}</div>
            <div style="margin-top: 8px;">
                <span class="{badge_class}">{trend}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
