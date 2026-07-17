"""LLMOps Studio Model Registry Page.

Displays fine-tuned model checkpoint catalog records, adapter paths, and version controls.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_registry_service

# 1. Config
st.set_page_config(page_title="Model Registry - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Model Registry", "Version checkpoints, manage catalog entries, and resolve adapter paths")

# 3. Retrieve registry list from services
registry_service = get_registry_service()
checkpoints = registry_service.list_checkpoints()

# 4. Display Checkpoint catalog
st.subheader("Registered Model Adapters")
if not checkpoints:
    st.info("No fine-tuned checkpoints cataloged in database registry yet.")
else:
    registry_table = []
    for cp in checkpoints:
        registry_table.append({
            "Checkpoint ID": cp.checkpoint_id,
            "Model Label": cp.model_name,
            "Base Foundation": cp.base_model,
            "Target Path": cp.path,
            "Tags": ", ".join(cp.tags),
            "Registered At": cp.registered_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    st.table(registry_table)

st.write("")

# 5. Registry operations placeholders
st.subheader("Adapter Checkpoint Operations")
col1, col2 = st.columns(2)
with col1:
    st.selectbox("Select Checkpoint Target", ["qlora_llama_001"])
    st.text_input("Assign Version Tag", placeholder="e.g. production-v1")
with col2:
    st.selectbox("Promote Model Target Environment", ["Development", "Staging", "Production"])
    st.button("Apply Tag & Promote Version", disabled=True)

st.write("")

# 6. Future roadmap integration
render_future_phase_box(
    package_name="backend/registry/",
    description="This view links with backend/services/registry_service.py. "
                "In future releases, this screen will load details using ModelCatalog, "
                "read checkpoint config files from CheckpointMetadataManager, and compress/export weight "
                "packages for API integrations using CheckpointStorageManager."
)
