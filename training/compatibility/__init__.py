"""LLMOps Studio Training Compatibility Layer.

Provides lightweight compatibility utilities for Hugging Face TRL / Transformers
training pipelines. Resolves the upstream TRL force-cast issue where `SFTTrainer`
unconditionally mutates trainable parameters to `torch.bfloat16` on quantized models.
"""

from .trl_precision_fix import apply_trl_precision_fix, validate_environment

__all__ = ["apply_trl_precision_fix", "validate_environment"]
