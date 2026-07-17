"""Evaluation Schemas.

Defines Pydantic structures for model evaluation metrics and judges outputs.
"""

from typing import Dict
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Aggregated metrics collected from running an evaluation dataset against a model."""
    eval_id: str = Field(
        ..., 
        description="Unique evaluation task sequence identifier"
    )
    model_name: str = Field(
        ..., 
        description="Name of the model profile evaluated"
    )
    dataset_name: str = Field(
        ..., 
        description="Name of the evaluation benchmark dataset used"
    )
    
    # Text Generation Quality
    rouge_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Computed ROUGE metrics (e.g. rouge1, rouge2, rougeL)"
    )
    bertscore: float = Field(
        0.0, 
        description="BERTScore F1 similarity representation"
    )
    exact_match: float = Field(
        0.0, 
        description="Exact match ratio for precise instruction prompts"
    )
    
    # Hardware/Performance Metrics
    latency_ms: float = Field(
        0.0, 
        description="Average inference request latency in milliseconds"
    )
    throughput_tokens_per_sec: float = Field(
        0.0, 
        description="Model throughput speed in tokens per second"
    )
    
    # Judge & Cost
    llm_judge_score: float = Field(
        0.0, 
        description="Quality metric rating computed via LLM-as-a-Judge (1-5 scale)"
    )
    cost_estimation_usd: float = Field(
        0.0, 
        description="Estimated hardware computing and token cost in USD"
    )
