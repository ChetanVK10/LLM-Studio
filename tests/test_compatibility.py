"""Unit tests for training/compatibility precision layer."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from training.compatibility import apply_trl_precision_fix, validate_environment


class MockModel(nn.Module):
    """Mock PyTorch module simulating quantized base model + LoRA adapters."""
    def __init__(self, is_4bit: bool = True):
        super().__init__()
        self.is_loaded_in_4bit = is_4bit
        # Frozen base weights
        self.base_layer = nn.Linear(32, 32, bias=False)
        for p in self.base_layer.parameters():
            p.requires_grad = False
        # Trainable LoRA adapter weights
        self.lora_A = nn.Parameter(torch.randn(16, 32))
        self.lora_B = nn.Parameter(torch.randn(32, 16))


class MockTrainer:
    """Mock Hugging Face / TRL Trainer."""
    def __init__(self, model: nn.Module, fp16: bool = True, bf16: bool = False):
        self.model = model
        self.args = MagicMock()
        self.args.fp16 = fp16
        self.args.bf16 = bf16


def test_validate_environment_invalid_trainer():
    """Test validation fails on invalid trainer reference."""
    with pytest.raises(ValueError, match="Invalid trainer reference"):
        validate_environment(None, target_dtype=torch.float16)


def test_precision_fix_converts_bfloat16_lora_params():
    """Test that bfloat16 trainable parameters are converted to target dtype (float16)."""
    model = MockModel(is_4bit=True)
    # Simulate TRL mutation to bfloat16
    model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
    model.lora_B.data = model.lora_B.data.to(torch.bfloat16)

    trainer = MockTrainer(model, fp16=True, bf16=False)

    metrics = apply_trl_precision_fix(trainer, target_dtype=torch.float16, verbose=False)

    assert metrics["fix_applied"] is True
    assert metrics["converted"] == 2
    assert metrics["status"] == "PASS"
    assert model.lora_A.dtype == torch.float16
    assert model.lora_B.dtype == torch.float16


def test_precision_fix_preserves_frozen_base_weights():
    """Test that frozen base model parameters are untouched."""
    model = MockModel(is_4bit=True)
    base_dtype_before = model.base_layer.weight.dtype

    # Mutate trainable adapter params
    model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
    model.lora_B.data = model.lora_B.data.to(torch.bfloat16)

    trainer = MockTrainer(model, fp16=True, bf16=False)
    apply_trl_precision_fix(trainer, target_dtype=torch.float16, verbose=False)

    assert model.base_layer.weight.dtype == base_dtype_before
    assert model.base_layer.weight.requires_grad is False


def test_auto_bypass_when_params_already_correct():
    """Test auto-bypass when parameters are already in target precision (e.g. future TRL release)."""
    model = MockModel(is_4bit=True)
    # Already float16
    model.lora_A.data = model.lora_A.data.to(torch.float16)
    model.lora_B.data = model.lora_B.data.to(torch.float16)

    trainer = MockTrainer(model, fp16=True, bf16=False)

    metrics = apply_trl_precision_fix(trainer, target_dtype=torch.float16, verbose=False)

    assert metrics["fix_applied"] is False
    assert metrics["converted"] == 0
    assert metrics["already_correct"] == 2
