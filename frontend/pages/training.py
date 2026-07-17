"""LLMOps Studio Training Page.

Configures QLoRA hyperparameter controls, starts training pipelines, and tracks job status.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_training_service

# 1. Config
st.set_page_config(page_title="Training - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Fine-Tuning (QLoRA)", "Setup hyperparameters, apply low-rank quantization adapters, and start training")

# 3. Fine-tuning setup controls form
st.subheader("Fine-Tuning Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model & Data parameters")
    base_model = st.text_input("Base Foundation Model", value=settings.default_model_name)
    dataset_name = st.selectbox("Training Dataset Key", ["alpaca-cleaned-1k"])

    st.markdown("### QLoRA Matrix parameters")
    lora_r = st.number_input("LoRA Rank (r)", value=settings.lora_r, step=8)
    lora_alpha = st.number_input("LoRA Scaling (alpha)", value=settings.lora_alpha, step=16)
    lora_dropout = st.slider("LoRA Dropout Rate", min_value=0.0, max_value=0.5, value=settings.lora_dropout, step=0.01)

with col2:
    st.markdown("### Optimization parameters")
    epochs = st.number_input("Training Epochs", value=settings.num_train_epochs, min_value=1)
    batch_size = st.number_input("Batch Size per Device", value=settings.batch_size, min_value=1)
    learning_rate = st.number_input("Learning Rate", value=settings.learning_rate, format="%.5f")
    
    st.markdown("### Target modules")
    target_modules = st.multiselect(
        "Layer modules targeted for adaptation",
        options=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        default=settings.target_modules
    )

st.write("")
st.button("Start Fine-Tuning Execution", disabled=True)

st.write("")

# 4. Present Active jobs
st.subheader("Fine-Tuning Jobs Lifecycle Tracker")
training_service = get_training_service()
jobs = training_service.list_active_jobs()

if not jobs:
    st.info("No active fine-tuning jobs are running or queued at present.")
else:
    # Display jobs
    st.write(jobs)

st.write("")

# 5. Future integration details
render_future_phase_box(
    package_name="backend/training/",
    description="This form integrates with backend/services/training_service.py. "
                "In the next phase, clicking 'Start Fine-Tuning Execution' will validate inputs using "
                "the TrainingRequest schema, invoke QLoRATrainer algorithms to initialize bitsandbytes 4-bit "
                "PEFT layers, and stream loss progress to local files under logs/training.log."
)
