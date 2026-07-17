"""Unit tests for the LoRA configuration helper.

Validates that schemas map correctly to adapter configuration parameters.
"""

from backend.schemas.training import LoraConfigSchema
from backend.training.lora import get_lora_config


def test_get_lora_config() -> None:
    """Asserts that schema variables translate accurately to LoraConfig fields."""
    schema = LoraConfigSchema(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj"]
    )
    config = get_lora_config(schema)

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.1
    assert config.target_modules == ["q_proj", "v_proj"]
    assert config.bias == "none"
