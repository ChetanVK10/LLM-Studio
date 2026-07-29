# TRL Precision Compatibility Module (`training/compatibility`)

## Problem Description & Context

When fine-tuning quantized models (4-bit/8-bit QLoRA) using Hugging Face TRL `SFTTrainer`, an upstream force-cast in `trl/trainer/sft_trainer.py` unconditionally converts all trainable parameters (`requires_grad=True`) to `torch.bfloat16` during `SFTTrainer.__init__()`.

### Upstream Code Location

In TRL 1.9.2 (`trl/trainer/sft_trainer.py`, lines 1127–1130):

```python
if _is_quantized_model:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)  # ← Upstream force-cast
```

### Why Tesla T4 GPUs Crash

1. **Hardware Limitation**: NVIDIA Tesla T4 (Turing Architecture, Compute Capability 7.5) lacks native `bfloat16` hardware execution instructions.
2. **GradScaler Failure**: When `fp16=True`, PyTorch `GradScaler` expects `float16` or `float32` gradients. Passing `bfloat16` gradients causes runtime crashes or invalid scaling during `trainer.train()`.

---

## Compatibility Layer Architecture

This module provides a non-invasive, repository-local precision alignment guard that runs **after** `SFTTrainer.__init__()` and **before** `trainer.train()`.

```
[SFTTrainer Init Completed]
           │
           ▼
[apply_trl_precision_fix(trainer, target_dtype)]
           │
           ├─► 1. Environment & Hardware Inspection
           │      └── Evaluate whether fix is needed (fix_required)
           │
           ├─► 2. Targeted Parameter Scan
           │      ├── Convert deviating trainable params: param.data.to(target_dtype)
           │      └── Preserve frozen base model weights
           │
           ├─► 3. Post-Condition Verification
           │      └── Assert 100% parameter precision alignment
           │
           ▼
[trainer.train() runs cleanly with matching FP16/FP32 adapter weights]
```

---

## Integration Guide

### 1. `training/scripts/train.py`

```python
from trl import SFTTrainer
from training.compatibility import apply_trl_precision_fix

# Instantiate SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    callbacks=[metrics_callback],
    **sft_kwargs
)

# Apply precision fixup guard
apply_trl_precision_fix(
    trainer=trainer,
    target_dtype=compute_dtype,
    verbose=True
)

# Launch training safely
trainer.train()
```

### 2. `03_qlora_training.ipynb`

In the training cell immediately following `SFTTrainer` creation:

```python
from training.compatibility import apply_trl_precision_fix

apply_trl_precision_fix(trainer=trainer, target_dtype=compute_dtype, verbose=True)
```

---

## How to Remove When TRL Fixes the Issue

If a future TRL release removes the hardcoded `bfloat16` force-cast:

1. **Automatic Bypass**: This compatibility layer automatically detects if parameters are already in the target precision (`fix_required=False`) and skips parameter conversion with zero side effects.
2. **Complete Removal**: To cleanly remove this module from the repository:
   - Remove the `from training.compatibility import apply_trl_precision_fix` line and `apply_trl_precision_fix(...)` call from `train.py` and `03_qlora_training.ipynb`.
   - Delete the `training/compatibility/` directory.
