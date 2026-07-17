"""LLMOps Studio Datasets Page.

Manages data profile registers, displays preprocessing pipelines configurations, and splits.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_dataset_service

# 1. Config
st.set_page_config(page_title="Datasets - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Dataset Management", "Ingest, preprocess, and partition training and evaluation datasets")

# 3. Retrieve datasets from service layer
dataset_service = get_dataset_service()
datasets = dataset_service.list_datasets()

# 4. Present Datasets list
st.subheader("Registered Datasets Catalog")
if not datasets:
    st.info("No datasets cataloged in database registry yet.")
else:
    # Prepare clean dictionary mapping for display table
    data_table = []
    for ds in datasets:
        data_table.append({
            "Dataset Name": ds.dataset_name,
            "Total Samples": ds.num_samples,
            "Target Splits": ds.split_ratio,
            "Schema Columns": ", ".join(ds.features),
            "Ingestion Date": ds.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    st.table(data_table)

st.write("")

# 5. Preprocessing controls placeholder
st.subheader("Ingest Custom Dataset")
col1, col2 = st.columns(2)
with col1:
    st.text_input("Dataset Identifier Key", placeholder="e.g. alpaca-cleaned-10k")
    st.text_input("Raw File Path (JSON/Parquet)", placeholder="e.g. data/raw/alpaca.json")
with col2:
    st.selectbox("Tokenizer Matching Profile", ["meta-llama/Llama-3-8b", "mistralai/Mistral-7B-v0.1"])
    st.text_input("Split Percentages (Train/Val/Test)", value="80/10/10")

st.button("Queue Dataset Processing", disabled=True)

st.write("")

# 6. Future roadmap integration
render_future_phase_box(
    package_name="backend/datasets/",
    description="This view links with backend/services/dataset_service.py. "
                "In future phases, the 'Queue Dataset Processing' trigger will call DatasetProcessor "
                "to clean files, generate vocabulary alignments, write outputs to parquet files under data/processed/ "
                "and populate DatasetMetadata schemas."
)
