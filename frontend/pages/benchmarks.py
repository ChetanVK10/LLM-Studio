"""LLMOps Studio Benchmarks Page.

Compares model speed profiles, peak VRAM allocations, throughput rates, and cost estimators.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_benchmark_service

# 1. Config
st.set_page_config(page_title="Benchmarks - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Model Benchmarking", "Compare latency and throughput constraints side-by-side across multiple model weight checkpoints")

# 3. Retrieve benchmark runs from service layer
bench_service = get_benchmark_service()
runs = bench_service.list_benchmark_runs()

# 4. Display benchmark table
st.subheader("Performance Comparison Matrix")

# Create placeholder data for illustration
mock_comparison_data = [
    {
        "Model Key Name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Throughput (tok/s)": "38.2 tok/s",
        "Avg Latency (ms)": "95.5 ms",
        "Peak VRAM Footprint": "15,840 MB",
        "Avg Cost / 1M Tokens": "$0.180"
    },
    {
        "Model Key Name": "LLaMA-3-8B-QLoRA-Custom (Adapter)",
        "Throughput (tok/s)": "35.5 tok/s",
        "Avg Latency (ms)": "104.2 ms",
        "Peak VRAM Footprint": "12,450 MB",
        "Avg Cost / 1M Tokens": "$0.140"
    }
]

st.table(mock_comparison_data)

st.write("")

# 5. Create new Benchmark Trigger inputs
st.subheader("Launch New Benchmarking Task")
col1, col2 = st.columns(2)
with col1:
    st.multiselect(
        "Select Model Checkpoints to Benchmark",
        options=["meta-llama/Meta-Llama-3-8B-Instruct", "LLaMA-3-8B-QLoRA-Custom"],
        default=["meta-llama/Meta-Llama-3-8B-Instruct", "LLaMA-3-8B-QLoRA-Custom"]
    )
with col2:
    st.selectbox("Select Test Dataset", ["alpaca-cleaned-1k"])
    st.number_input("Inference Concurrency (Batch Size)", value=4, min_value=1)

st.button("Run Benchmark Suite", disabled=True)

st.write("")

# 6. Future roadmap integration
render_future_phase_box(
    package_name="backend/benchmarking/",
    description="This view links with backend/services/benchmark_service.py. "
                "In future releases, triggering benchmarks will invoke ModelBenchmarker "
                "to run batch inferences across target devices, profile peak memory footprint allocation "
                "and tokens processing speeds using PyTorch profilers, and log metrics data to logs/benchmark.log."
)
