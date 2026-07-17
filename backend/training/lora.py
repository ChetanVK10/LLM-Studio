"""LoRA Configuration Helper.

Bridges schema configurations with Hugging Face PEFT LoraConfig adapters,
incorporating mock fallbacks if dependencies are absent.
"""

import logging
from typing import Any

from backend.schemas.training import LoraConfigSchema

logger = logging.getLogger("app")

try:
    from peft import LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    
    class TaskType:
        """Mock TaskType Enum."""
        CAUSAL_LM = "CAUSAL_LM"
        
    class LoraConfig:
        """Mock LoraConfig class."""
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


def get_lora_config(schema: LoraConfigSchema) -> LoraConfig:
    """Builds a PEFT LoraConfig from a validation schema.

    Args:
        schema: Pydantic LoRA configuration schema.

    Returns:
        Hugging Face PEFT LoraConfig adapter object.
    """
    logger.info(
        f"LoRA Helper: Creating adapter configuration: rank={schema.r}, "
        f"alpha={schema.lora_alpha}, targets={schema.target_modules}"
    )
    
    if PEFT_AVAILABLE:
        return LoraConfig(
            r=schema.r,
            lora_alpha=schema.lora_alpha,
            target_modules=schema.target_modules,
            lora_dropout=schema.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    else:
        logger.warning("PEFT library not imported. Falling back to mocked LoraConfig representation.")
        return LoraConfig(
            r=schema.r,
            lora_alpha=schema.lora_alpha,
            target_modules=schema.target_modules,
            lora_dropout=schema.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
