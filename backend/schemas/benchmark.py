"""Benchmark Schemas.

Defines Pydantic structures for multi-model performance benchmarking summaries.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    """Execution output comparing latencies and resource usage metrics across multiple models."""
    benchmark_id: str = Field(
        ..., 
        description="Unique benchmarking run identifier"
    )
    models: List[str] = Field(
        ..., 
        description="List of compared target model identifiers"
    )
    dataset_name: str = Field(
        ..., 
        description="Benchmark dataset mapping utilized"
    )
    latency_comparison: Dict[str, float] = Field(
        ..., 
        description="Average latency comparison mapping in milliseconds per model"
    )
    throughput_comparison: Dict[str, float] = Field(
        ..., 
        description="Tokens per second throughput rating per model"
    )
    memory_peak_comparison: Dict[str, float] = Field(
        ..., 
        description="Peak VRAM graphics memory footprint footprint comparison (MB)"
    )
    cost_comparison: Dict[str, float] = Field(
        ..., 
        description="Estimated relative dollar cost per 1M generated tokens per model"
    )
