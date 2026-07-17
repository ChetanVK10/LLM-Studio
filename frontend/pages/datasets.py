"""LLMOps Studio Datasets Page.

Manages data ingestion (local uploads & Hugging Face Hub), runs schemas validation,
profiles dataset statistics, and executes random splits.
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from backend.core.config import get_settings
from backend.core.dependencies import get_dataset_service, get_storage_manager
from backend.core.exceptions import DatasetError, DatasetValidationError
from backend.schemas.datasets import DatasetSplitConfig
from frontend.components.ui import render_header, render_metric_card

settings = get_settings()

# 1. Config
st.set_page_config(page_title="Dataset Manager - LLMOps Studio", layout="wide")

# Apply custom styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Dataset Management", "Ingest, validate, profile, and partition training and evaluation datasets")

dataset_service = get_dataset_service()
storage_manager = get_storage_manager()

# 3. Sidebar / Split ratios controls
st.sidebar.markdown("### Dataset Split Configuration")
train_pct = st.sidebar.slider("Train Split (%)", 0, 100, 80, step=5)
val_pct = st.sidebar.slider("Validation Split (%)", 0, 100 - train_pct, 10, step=5)
test_pct = 100 - train_pct - val_pct
st.sidebar.info(f"Splits Ratio: Train {train_pct}% / Val {val_pct}% / Test {test_pct}%")

random_seed = st.sidebar.number_input("Random Seed for Splits", value=42, step=1)

split_config = DatasetSplitConfig(
    train_ratio=float(train_pct) / 100.0,
    val_ratio=float(val_pct) / 100.0,
    test_ratio=float(test_pct) / 100.0,
    random_seed=random_seed
)

# 4. Ingest Custom Dataset Section
st.subheader("📥 Ingest Custom Dataset")
ingestion_mode = st.radio("Ingestion Source", ["Local File Upload", "Hugging Face Hub Ingestion"], horizontal=True)

if ingestion_mode == "Local File Upload":
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV, JSON, or JSONL dataset file", 
            type=["csv", "json", "jsonl"]
        )
    with col2:
        custom_name = st.text_input(
            "Dataset Logical Name (optional)", 
            placeholder="e.g. alpaca-clean"
        )
        
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        # Resolve ext
        file_ext = "." + file_name.split(".")[-1]
        display_name = custom_name.strip() if custom_name.strip() else file_name.split(".")[0]
        
        if st.button("Ingest Local Dataset"):
            with st.spinner("Processing file ingestion pipeline..."):
                try:
                    meta = dataset_service.import_local_dataset(
                        name=display_name,
                        content=file_bytes,
                        file_extension=file_ext,
                        split_config=split_config
                    )
                    st.success(f"✓ Dataset '{display_name}' ingested successfully! ID: `{meta.dataset_id[:8]}`")
                    st.rerun()
                except DatasetValidationError as ve:
                    st.error(f"❌ Dataset Validation Failed: {ve}")
                except DatasetError as de:
                    st.error(f"❌ Ingestion Error: {de}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error during parsing: {e}")

else:  # Hugging Face
    col1, col2 = st.columns(2)
    with col1:
        hf_repo_id = st.text_input(
            "Hugging Face Repository ID", 
            placeholder="e.g. tatsu-lab/alpaca"
        )
    with col2:
        hf_split = st.text_input("HF Dataset Split Target", value="train")
        
    if st.button("Import from Hugging Face Hub"):
        if not hf_repo_id.strip():
            st.warning("Please enter a Hugging Face Repository ID.")
        else:
            with st.spinner(f"Ingesting records from Hugging Face: {hf_repo_id}..."):
                try:
                    meta = dataset_service.import_huggingface_dataset(
                        repo_id=hf_repo_id,
                        split=hf_split,
                        split_config=split_config
                    )
                    st.success(f"✓ HF Dataset imported successfully! ID: `{meta.dataset_id[:8]}`")
                    st.rerun()
                except DatasetValidationError as ve:
                    st.error(f"❌ HF Dataset Validation Failed: {ve}")
                except DatasetError as de:
                    st.error(f"❌ Import Error: {de}")
                except Exception as e:
                    st.error(f"❌ Unexpected HF Integration Error: {e}")

st.markdown("---")

# 5. Dataset Browser & Profile statistics
datasets = dataset_service.list_datasets()

st.subheader("🔍 Dataset Browser & Statistics")
if not datasets:
    st.info("No registered datasets found in registry catalog database.")
else:
    options = {f"{ds.dataset_name} ({ds.dataset_id[:8]})": ds.dataset_id for ds in datasets}
    selected_label = st.selectbox("Select a registered dataset to inspect", list(options.keys()))
    selected_id = options[selected_label]
    
    meta = dataset_service.get_dataset(selected_id)
    if meta:
        st.markdown(f"#### Profile: **{meta.dataset_name}**")
        st.caption(
            f"ID: `{meta.dataset_id}` | Source: `{meta.source}` | "
            f"Status: `{meta.lifecycle_state.value.upper()}`"
        )
        
        # Grid metrics
        if meta.profile:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Total Samples", f"{meta.profile.rows_count:,}", "📊 Row records count")
            with col2:
                render_metric_card("Estimated Tokens", f"{meta.profile.estimated_tokens:,}", "🪙 Heuristic count")
            with col3:
                render_metric_card("Avg Prompt Len", f"{meta.profile.avg_prompt_length} ch", "✍️ Mean prompt length")
            with col4:
                render_metric_card("Avg Response Len", f"{meta.profile.avg_response_length} ch", "🎯 Mean target length")
                
            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Duplicate records:**")
                st.code(f"{meta.profile.duplicate_count} duplicate rows detected.")
            with col2:
                st.markdown("**Columns detected:**")
                st.code(", ".join(meta.profile.columns))
                
        # Validation Reports Panel
        if meta.validation_report:
            st.markdown("##### Quality Validation Audit Report")
            if meta.validation_report.is_valid:
                st.success("✓ Dataset conforms to all target fine-tuning constraints.")
            else:
                st.error("❌ Dataset contains validation errors. Core execution blocked.")
                
            if meta.validation_report.issues:
                for issue in meta.validation_report.issues:
                    icon = "🔵 [INFO]"
                    if issue.severity.value == "error":
                        icon = "🔴 [ERROR]"
                    elif issue.severity.value == "warning":
                        icon = "🟡 [WARNING]"
                    st.markdown(f"- **{icon}** (Field: `{issue.affected_field}`): {issue.message}")
            else:
                st.info("No validation findings recorded.")
                
        # Preview Samples Dataframe
        if meta.processed_path:
            st.markdown("##### Dataset splits preview (Train partition first 5 rows)")
            train_file_path = f"{meta.processed_path}/train.jsonl"
            try:
                if storage_manager.exists(train_file_path):
                    raw_lines = storage_manager.read_file(train_file_path).decode("utf-8").strip().splitlines()
                    preview_records = [json.loads(line) for line in raw_lines[:5] if line.strip()]
                    if preview_records:
                        df_preview = pd.DataFrame(preview_records)
                        st.dataframe(df_preview, use_container_width=True)
                    else:
                        st.info("Train partition contains no records.")
                else:
                    st.warning("Processed split files could not be located in storage.")
            except Exception as e:
                st.error(f"Error loading preview records: {e}")

st.markdown("---")

# 6. History Table
st.subheader("📜 Ingestion History & Registry Logs")
if not datasets:
    st.info("No logs present in metadata registry.")
else:
    history_data = []
    for ds in datasets:
        history_data.append({
            "Dataset ID": ds.dataset_id[:12] + "...",
            "Name": ds.dataset_name,
            "Source": ds.source,
            "Created Date": ds.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Lifecycle State": ds.lifecycle_state.value.upper(),
            "Version": ds.version
        })
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
