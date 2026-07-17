"""LLMOps Studio Settings Page.

Displays active system environment configurations, directories writability, and parameters profiles.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()

# 1. Config
st.set_page_config(page_title="Settings - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("System Settings", "Inspect active environment configuration loads, directories mappings, and fallback parameters")

# 3. Environment Summary
st.subheader("Runtime Environment Profile")

col1, col2 = st.columns(2)
with col1:
    st.text_input("Active App Environment (APP_ENV)", value=settings.app_env, disabled=True)
    st.text_input("Application Name", value=settings.app_name, disabled=True)
    st.text_input("Application Version", value=settings.app_version, disabled=True)
with col2:
    st.text_input("Hugging Face API Status", value="CONFIGURED (MOCK)" if settings.huggingface_api_key else "MISSING", disabled=True)
    st.text_input("OpenAI API Status", value="CONFIGURED (MOCK)" if settings.openai_api_key else "MISSING", disabled=True)
    st.text_input("Default model name", value=settings.default_model_name, disabled=True)

st.write("")

# 4. Storage paths mapping
st.subheader("Physical Directories Mappings")

path_data = [
    {"Directory Target": "Workspace Root Path", "Path": str(settings.workspace_root)},
    {"Directory Target": "Data Cache & Processing Path", "Path": str(settings.data_dir)},
    {"Directory Target": "Artifacts Checkpoint Outputs Path", "Path": str(settings.artifact_dir)},
    {"Directory Target": "System Logs Storage Path", "Path": str(settings.log_dir)},
    {"Directory Target": "Foundation Models Cache Path", "Path": str(settings.model_dir)}
]
st.table(path_data)

st.write("")

# 5. Future roadmap integration
render_future_phase_box(
    package_name="configs/",
    description="This settings view loads configuration properties initialized through backend/core/config.py. "
                "In subsequent development phases, environment overrides or yaml configurations changes "
                "will reload dynamically and adapt local GPU profiles and active training queues."
)
