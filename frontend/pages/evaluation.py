"""LLMOps Studio Evaluation Page.

Runs evaluation suites, presents ROUGE, BLEU, classification matrices,
and comparative leaderboards.
"""

import json
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from backend.core.config import get_settings
from backend.core.constants import DatasetLifecycleState
from backend.core.dependencies import get_dataset_service, get_evaluation_service, get_storage_manager, get_training_service
from backend.core.exceptions import EvaluationError
from backend.schemas.evaluation import EvaluationConfig, TaskType
from frontend.components.ui import render_header, render_metric_card

settings = get_settings()

# 1. Config
st.set_page_config(page_title="Evaluation - LLMOps Studio", layout="wide")

# Apply custom styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Model Evaluation", "Evaluate generation quality, classification performance, and rank checkpoints on leaderboards")

eval_service = get_evaluation_service()
training_service = get_training_service()
dataset_service = get_dataset_service()
storage_manager = get_storage_manager()

# Load registries
checkpoints = training_service.list_checkpoints()
datasets = dataset_service.list_datasets()
ready_datasets = [ds for ds in datasets if ds.lifecycle_state == DatasetLifecycleState.READY]

tab1, tab2, tab3 = st.tabs(["⚙️ Run Evaluation", "📈 Results Grid", "🏆 Benchmark Leaderboard"])

with tab1:
    st.subheader("Launch New Model Evaluation")
    
    if not checkpoints:
        st.warning(
            "⚠️ No model checkpoints are registered. Please fine-tune a model first on "
            "the [Fine-Tuning](training) page."
        )
    elif not ready_datasets:
        st.warning(
            "⚠️ No datasets are READY. Please process a dataset first on "
            "the [Dataset Management](datasets) page."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. Target Model & Dataset")
            checkpoint_options = {f"{ck.model_name} ({ck.checkpoint_id[:8]})": ck.checkpoint_id for ck in checkpoints}
            selected_model_label = st.selectbox("Select model checkpoint", list(checkpoint_options.keys()))
            selected_model_id = checkpoint_options[selected_model_label]

            dataset_options = {f"{ds.dataset_name} ({ds.dataset_id[:8]})": ds.dataset_id for ds in ready_datasets}
            selected_ds_label = st.selectbox("Select evaluation dataset", list(dataset_options.keys()))
            selected_ds_id = dataset_options[selected_ds_label]
            
            task_type = st.radio(
                "Task Category Type",
                options=[TaskType.GENERATION, TaskType.CLASSIFICATION],
                format_func=lambda x: x.value.title()
            )

        with col2:
            st.markdown("#### 2. Inference Parameters")
            max_new_tokens = st.number_input("Max New Tokens", min_value=1, value=128, step=32)
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
            top_p = st.slider("Top-p Sampling", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
            batch_size = st.number_input("Batch Size", min_value=1, value=4, step=1)
            seed = st.number_input("Random Seed", value=42)

        if st.button("Start Pipeline Evaluation"):
            eval_id = f"eval-{uuid.uuid4().hex[:12]}"
            config = EvaluationConfig(
                evaluation_id=eval_id,
                model_id=selected_model_id,
                dataset_id=selected_ds_id,
                task_type=task_type,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                random_seed=seed
            )
            
            try:
                with st.spinner("Processing inferences and calculating scores..."):
                    result = eval_service.run_evaluation(config)
                st.success(f"✓ Evaluation completed! ID: `{result.evaluation_id}`")
                
                # Show scores cards immediately
                st.markdown("##### Scores Output:")
                m_cols = st.columns(4)
                idx = 0
                for m, val in result.metrics.model_dump().items():
                    if val is not None:
                        with m_cols[idx % 4]:
                            render_metric_card(m.upper(), f"{val:.4f}", "Computed score")
                        idx += 1
            except EvaluationError as ee:
                st.error(f"❌ Evaluation Failure: {ee}")
            except Exception as e:
                st.error(f"❌ System error during pipeline evaluation: {e}")

with tab2:
    st.subheader("Evaluation Results Directory")
    evaluations = eval_service.list_evaluations()
    
    if not evaluations:
        st.info("No evaluations records stored in registry yet.")
    else:
        # Select evaluation run
        eval_options = {f"{e.evaluation_id} ({e.created_at.strftime('%Y-%m-%d %H:%M')})": e.evaluation_id for e in evaluations}
        selected_eval_label = st.selectbox("Select evaluation run", list(eval_options.keys()))
        selected_eval_id = eval_options[selected_eval_label]
        
        result = eval_service.get_evaluation(selected_eval_id)
        if result:
            st.markdown("#### Evaluation Metric Highlights")
            m_cols = st.columns(4)
            idx = 0
            for m, val in result.metrics.model_dump().items():
                if val is not None:
                    with m_cols[idx % 4]:
                        render_metric_card(m.upper(), f"{val:.4f}", "Computed score")
                    idx += 1
            
            st.write("")
            st.markdown("#### Full Summary Report")
            report_md = eval_service.get_report_markdown(selected_eval_id)
            st.markdown(report_md)
            
            # Download button
            st.download_button(
                "📥 Download Summary Report (MD)",
                data=report_md,
                file_name=f"eval_report_{selected_eval_id}.md",
                mime="text/markdown"
            )

with tab3:
    st.subheader("Benchmarking Leaderboard")
    
    if not ready_datasets:
        st.info("Please register and process datasets to evaluate models.")
    else:
        dataset_options = {f"{ds.dataset_name} ({ds.dataset_id[:8]})": ds.dataset_id for ds in ready_datasets}
        selected_ds_label = st.selectbox("Select dataset for leaderboard", list(dataset_options.keys()))
        selected_ds_id = dataset_options[selected_ds_label]
        
        leaderboard = eval_service.get_leaderboard(selected_ds_id)
        
        if not leaderboard:
            st.info("No leaderboard entries registered for this dataset yet. Run evaluations to compile comparisons.")
        else:
            df_leaderboard = pd.DataFrame(leaderboard)
            st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)
            
            # Draw st.bar_chart of scores comparison
            st.markdown("##### Visual Performance Comparison")
            chart_df = df_leaderboard[["model_name", "metric_value"]].set_index("model_name")
            st.bar_chart(chart_df)
