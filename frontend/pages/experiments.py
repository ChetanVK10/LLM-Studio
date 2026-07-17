"""LLMOps Studio Experiments Page.

Tracks experiment metadata, hyperparameters histories, and runs logs comparisons.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_experiment_tracker

# 1. Config
st.set_page_config(page_title="Experiments - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Experiment Tracking", "Log training runs, compare hyperparameter records, and check metrics histories")

# 3. Present runs
st.subheader("Historical Tracking Runs")

# Placeholder comparison tables
mock_runs = [
    {
        "Run ID": "run_qlora_llama_001",
        "Experiment Name": "llama3-8b-alpaca-qlora",
        "Status": "Completed",
        "Learning Rate": 0.0002,
        "LoRA Rank (r)": 16,
        "LoRA Alpha": 32,
        "Training Loss": 0.452,
        "Started At": "2026-07-17 19:15:00"
    }
]

st.table(mock_runs)

st.write("")

# 4. Sync details
st.subheader("Remote Tracking Configurations")
st.text_input("Weights & Biases Workspace", value="llmops-studio-team", disabled=True)
st.text_input("MLflow Run Server URI", value="http://localhost:5000", disabled=True)

st.write("")

# 5. Future roadmap integration
render_future_phase_box(
    package_name="backend/experiment_tracking/",
    description="This view links with backend/experiment_tracking/ modules. "
                "In subsequent phases, training loops will use RunLifecycleManager "
                "to initiate run tracking contexts, log parameters (such as learning_rate), "
                "record epoch step metrics data via ExperimentMetricsStore, and sync log lines "
                "with remote tools via WandbIntegration."
)
