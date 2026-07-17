"""LLMOps Studio Evaluation Page.

Runs evaluation suites, presents ROUGE, BERTScore, Exact Match, judge scores, and resource stats.
"""

import streamlit as st

from frontend.components.ui import render_header, render_future_phase_box
from backend.core.config import get_settings

settings = get_settings()
from backend.core.dependencies import get_evaluation_service

# 1. Config
st.set_page_config(page_title="Evaluation - LLMOps Studio", layout="wide")

# Styling
with open(settings.workspace_root / "frontend" / "styles" / "main.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Header
render_header("Model Evaluation", "Evaluate generation quality, system latencies, judge scores, and compute cost profiles")

# 3. Retrieve evaluation runs from services
eval_service = get_evaluation_service()
runs = eval_service.list_evaluation_runs()

# 4. Display evaluation histories
st.subheader("Completed Evaluation Runs")
if not runs:
    st.info("No evaluations run logs cataloged in database registry yet.")
else:
    eval_table = []
    for r in runs:
        eval_table.append({
            "Eval ID": r.eval_id,
            "Model Under Test": r.model_name,
            "Target Dataset": r.dataset_name,
            "ROUGE-L": f"{r.rouge_scores.get('rougeL', 0.0):.4f}",
            "BERTScore F1": f"{r.bertscore:.4f}",
            "Exact Match": f"{r.exact_match:.2%}",
            "Avg Latency": f"{r.latency_ms:.1f} ms",
            "Throughput": f"{r.throughput_tokens_per_sec:.1f} tok/s",
            "LLM Judge Score": f"{r.llm_judge_score:.1f} / 5.0",
            "Estimated Cost": f"${r.cost_estimation_usd:.3f}"
        })
    st.table(eval_table)

st.write("")

# 5. Evaluate parameters panels
st.subheader("Configure New Evaluation Task")
col1, col2 = st.columns(2)
with col1:
    st.selectbox("Select Model to Evaluate", ["LLaMA-3-8B-QLoRA-Custom"])
    st.selectbox("Select Evaluation Dataset", ["alpaca-cleaned-1k"])
with col2:
    st.multiselect(
        "Select Scoring Methods",
        options=["ROUGE", "BERTScore", "Exact Match", "LLM-as-a-Judge", "Cost & Performance Benchmarking"],
        default=["ROUGE", "BERTScore", "Exact Match", "LLM-as-a-Judge", "Cost & Performance Benchmarking"]
    )

st.button("Launch Evaluation Suite", disabled=True)

st.write("")

# 6. Future roadmap integration
render_future_phase_box(
    package_name="backend/evaluation/",
    description="This view links with backend/services/evaluation_service.py. "
                "In subsequent phases, launching evaluations will invoke ModelEvaluator "
                "to calculate generation scores, run system prompts through GPT-4/Claude LLM Judge, "
                "profile latency statistics, and write results to artifacts/evaluations/."
)
