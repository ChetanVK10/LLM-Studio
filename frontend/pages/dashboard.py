"""LLMOps Studio Dashboard Page.

Displays a high-level operational overview, system health status, and directory layout stats.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box, render_metric_card
from backend.core.config import get_settings

settings = get_settings()

# 1. Page Config
st.set_page_config(page_title="Dashboard - LLMOps Studio", layout="wide")

# Apply custom styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Render Header
render_header("LLMOps Studio", "Central Operations and Pipeline Dashboard")

# 3. Render System Overview Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Fine-Tuned Checkpoints",
        value="1 Active",
        trend="v0.1.0-alpha",
        status="active"
    )

with col2:
    render_metric_card(
        title="Registered Datasets",
        value="1 Ingested",
        trend="1k rows (processed)",
        status="success"
    )

with col3:
    render_metric_card(
        title="Active Jobs",
        value="0 Running",
        trend="Idle",
        status="success"
    )

with col4:
    render_metric_card(
        title="Metrics Logs Size",
        value="12.5 KB",
        trend="5 Rotating logs",
        status="active"
    )

st.write("")

# 4. Operations Overview
st.subheader("System Architecture Map")
st.markdown(
    """
    LLMOps Studio implements a highly-decoupled architecture designed for enterprise-grade scalability.
    
    - **Frontend Layer**: Presentation dashboard (Streamlit) communicating strictly via services.
    - **Service Orchestration Layer**: Interfaces that map validation requests and delegate tasks to pipeline drivers.
    - **Physical Engines**: Pluggable executor units for QLoRA, generation scorers, and hardware benchmarks.
    - **Metadata & Registry Catalog**: Cryptographic checkpoint indexes mapping physical outputs to configurations.
    """
)

# 5. Future roadmap integration documentation
st.subheader("Integration Architecture Blueprint")
render_future_phase_box(
    package_name="backend/core/",
    description="This dashboard displays summaries computed by reading file metadata and service layers. "
                "In future releases, this screen will load live execution telemetry from a background process "
                "using the RunLifecycleManager and retrieve checkpoints stats from ModelCatalog."
)
