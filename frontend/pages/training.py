"""LLMOps Studio Training Page.

Configures QLoRA hyperparameter controls, starts training pipelines,
monitors optimization losses steps, and shows checkpoints histories.
"""

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from backend.core.config import get_settings
from backend.core.constants import DatasetLifecycleState
from backend.core.dependencies import get_dataset_service, get_storage_manager, get_training_service
from backend.core.exceptions import DatasetValidationError, HyperparameterError, TrainingError
from backend.schemas.training import HyperparametersSchema, LoraConfigSchema, TrainingJobConfig, TrainingJobState
from frontend.components.ui import render_header, render_metric_card

settings = get_settings()

# 1. Config
st.set_page_config(page_title="Fine-Tuning - LLMOps Studio", layout="wide")

# Apply custom styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Fine-Tuning (QLoRA)", "Setup hyperparameters, apply low-rank quantization adapters, and monitor training steps")

training_service = get_training_service()
dataset_service = get_dataset_service()
storage_manager = get_storage_manager()

# 3. Load active datasets to display in selection box
datasets = dataset_service.list_datasets()
ready_datasets = [ds for ds in datasets if ds.lifecycle_state == DatasetLifecycleState.READY]

# Resolve active job ID in session state
if "active_job_id" not in st.session_state:
    st.session_state.active_job_id = None

# Tab containers
tab1, tab2, tab3 = st.tabs(["⚙️ Configure Training", "📈 Active Job Monitor", "📜 Training History"])

with tab1:
    st.subheader("Launch New Fine-Tuning Job")
    
    if not ready_datasets:
        st.warning(
            "⚠️ No datasets are READY for training. Please process a dataset first on "
            "the [Dataset Management](datasets) page."
        )
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Foundation Model & Target split")
            base_model = st.text_input("Base Foundation Model", value=settings.default_model_name)
            
            # Map ready dataset display names to IDs
            dataset_options = {f"{ds.dataset_name} ({ds.dataset_id[:8]})": ds.dataset_id for ds in ready_datasets}
            selected_ds_label = st.selectbox("Select ready dataset", list(dataset_options.keys()))
            selected_ds_id = dataset_options[selected_ds_label]
            
            st.markdown("#### 2. LoRA Adapter Configuration")
            lora_r = st.number_input("LoRA Rank (r)", min_value=1, value=8, step=4)
            lora_alpha = st.number_input("LoRA Scaling (alpha)", min_value=1, value=16, step=8)
            lora_dropout = st.slider("LoRA Dropout Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
            target_modules = st.multiselect(
                "Target layer modules",
                options=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                default=["q_proj", "v_proj"]
            )
            
        with col2:
            st.markdown("#### 3. Hyperparameters & Optimization")
            epochs = st.number_input("Training Epochs", min_value=1, value=3, step=1)
            batch_size = st.number_input("Batch Size per Device", min_value=1, value=4, step=1)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=2e-4, format="%.6f")
            warmup_ratio = st.slider("Warmup Steps Ratio", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, value=0.01, format="%.4f")
            logging_steps = st.number_input("Logging Steps Frequency", min_value=1, value=5, step=1)
            
        if st.button("Start Fine-Tuning Run"):
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            
            config = TrainingJobConfig(
                job_id=job_id,
                base_model_name=base_model,
                dataset_id=selected_ds_id,
                hyperparameters=HyperparametersSchema(
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay,
                    logging_steps=logging_steps
                ),
                lora_config=LoraConfigSchema(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules
                )
            )
            
            try:
                started_id = training_service.start_training(config)
                st.session_state.active_job_id = started_id
                st.success(f"✓ Job launched successfully! Job ID: `{started_id}`")
                st.rerun()
            except TrainingError as te:
                st.error(f"❌ Launch Denied: {te}")
            except DatasetValidationError as dve:
                st.error(f"❌ Dataset Invalid: {dve}")
            except HyperparameterError as he:
                st.error(f"❌ Invalid Configurations: {he}")
            except Exception as e:
                st.error(f"❌ System failure starting job: {e}")

with tab2:
    st.subheader("Active Pipeline Progress Monitor")
    
    active_id = st.session_state.active_job_id
    
    if not active_id:
        st.info("No fine-tuning job is currently running or selected for monitoring.")
    else:
        st.button("🔄 Refresh Status Logs")
        
        status = training_service.get_job_status(active_id)
        if not status:
            st.warning(f"Unable to locate active status for job ID `{active_id}`.")
        else:
            st.markdown(f"#### Job Run Progress: **{active_id}**")
            
            # State color logic
            state_val = status.state.value.upper()
            if state_val == "RUNNING":
                st.info(f"⏳ Status: **{state_val}** (Elapsed Time: {int(status.elapsed_seconds)}s)")
            elif state_val == "COMPLETED":
                st.success(f"✓ Status: **{state_val}** (Duration: {int(status.elapsed_seconds)}s)")
                st.session_state.active_job_id = None  # Clear active job
            elif state_val == "FAILED":
                st.error(f"❌ Status: **{state_val}** | Error: {status.error_message}")
                st.session_state.active_job_id = None  # Clear active job
                
            # Progress bar
            progress_pct = 0.0
            if status.total_steps > 0:
                progress_pct = float(status.current_step) / float(status.total_steps)
            st.progress(min(1.0, progress_pct))
            
            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Step Progress", f"{status.current_step} / {status.total_steps}", "🏃 Step index counter")
            with col2:
                render_metric_card("Epoch", f"{status.current_epoch:.2f}", "🔄 Epochs completed")
            with col3:
                train_loss_val = f"{status.train_loss:.4f}" if status.train_loss is not None else "N/A"
                render_metric_card("Training Loss", train_loss_val, "📉 Loss value")
            with col4:
                eval_loss_val = f"{status.eval_loss:.4f}" if status.eval_loss is not None else "N/A"
                render_metric_card("Validation Loss", eval_loss_val, "🧪 Eval split loss")

            # Real-time loss curve chart
            st.markdown("##### Loss Curves Mapping")
            checkpoint_dir = f"artifacts/checkpoints/{active_id}"
            stats_file = f"{checkpoint_dir}/run_stats.json"
            
            try:
                log_file = "logs/training.log"
                if storage_manager.exists(log_file):
                    log_text = storage_manager.read_file(log_file).decode("utf-8")
                    # Extract step and loss values from log_text
                    steps = []
                    losses = []
                    for line in log_text.splitlines():
                        if "Step" in line and "Loss:" in line:
                            parts = line.split("|")
                            try:
                                step_str = parts[1].split("Step")[1].split("/")[0].strip()
                                loss_str = parts[2].split("Loss:")[1].strip()
                                steps.append(int(step_str))
                                losses.append(float(loss_str))
                            except Exception:
                                pass
                    if steps and losses:
                        df_loss = pd.DataFrame({"Loss": losses}, index=steps)
                        st.line_chart(df_loss)
                    else:
                        st.caption("Awaiting logged steps to plot curve...")
            except Exception as e:
                st.caption(f"Waiting for loss data: {e}")

            # Console execution outputs log
            st.markdown("##### Output Console Log Feed")
            log_file = "logs/training.log"
            try:
                if storage_manager.exists(log_file):
                    log_content = storage_manager.read_file(log_file).decode("utf-8")
                    st.text_area("Console Log", value=log_content, height=200, disabled=True)
                else:
                    st.text_area("Console Log", value="Console output is loading...", height=200, disabled=True)
            except Exception as e:
                st.error(f"Error reading console log: {e}")

with tab3:
    st.subheader("Training Job Catalog Registry")
    history = training_service.list_training_history()
    
    if not history:
        st.info("No training history registered in SQLite database catalog yet.")
    else:
        history_table = []
        for job in history:
            history_table.append({
                "Job ID": job.job_id,
                "State": job.state.value.upper(),
                "Epochs Completed": job.current_epoch,
                "Final Step": job.current_step,
                "Train Loss": f"{job.train_loss:.4f}" if job.train_loss is not None else "N/A",
                "Created Date": job.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        df_history = pd.DataFrame(history_table)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
    st.write("")
    st.subheader("Model Checkpoints Registry")
    checkpoints = training_service.list_checkpoints()
    if not checkpoints:
        st.info("No model checkpoints registered in catalog yet.")
    else:
        check_table = []
        for ck in checkpoints:
            check_table.append({
                "Checkpoint ID": ck.checkpoint_id,
                "Model Adapter Name": ck.model_name,
                "Base Model Source": ck.base_model,
                "Adapter Storage Path": ck.path,
                "Tags": ", ".join(ck.tags),
                "Registered Time": ck.registered_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        df_check = pd.DataFrame(check_table)
        st.dataframe(df_check, use_container_width=True, hide_index=True)
