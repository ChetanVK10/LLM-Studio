# Known Compatibility Issues & Risk Register — LLMOps Studio

This document records all known library compatibility issues, deprecation warnings, runtime traps, and architectural edge cases identified during system diagnostics and codebase audits.

---

## Issue 1: Unconditional `bfloat16` Adapter Cast in TRL `SFTTrainer` (CRITICAL)

* **Component**: `trl.trainer.SFTTrainer`
* **Affected Pipeline**: `03_qlora_training.ipynb`, `training/scripts/train.py`, `backend/training/trainer.py`
* **Target Hardware**: GPUs without native hardware BFloat16 support (e.g., NVIDIA Tesla T4, Compute Capability 7.5).
* **Severity**: **HIGH (Causes Training Crash)**
* **Empirical Evidence & Source Line**:
  - File: `trl/trainer/sft_trainer.py`, lines `1127–1130`:
    ```python
    if _is_quantized_model:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
    ```
* **Root Cause Analysis**:
  When `_is_quantized_model` is `True` (e.g., base model loaded with `BitsAndBytesConfig(load_in_4bit=True)`), `SFTTrainer.__init__` unconditionally mutates all trainable adapter parameters (`requires_grad=True`) to `torch.bfloat16`. 
  Under FP16 training (`fp16=True`, `bf16=False`), Accelerate initializes `GradScaler`. During the first backward pass, autograd yields `torch.bfloat16` gradients. PyTorch's `GradScaler.unscale_()` checks gradient dtypes and throws a fatal `RuntimeError`: `"BF16 gradient detected... GradScaler will crash"`.
* **Workaround / Fix Options**:
  1. **Post-Initialization Re-casting**: Immediately after calling `SFTTrainer(...)`, iterate through `trainer.model.parameters()` and re-cast `requires_grad=True` parameters to `torch.float32` (or `torch.float16`).
  2. **Library Patch**: Apply `trl_qlora_fp16.patch` to update the condition to `if _is_quantized_model and getattr(args, "bf16", False):`.

---

## Issue 2: `torch_dtype` Deprecation in Transformers 5.x

* **Component**: `transformers.AutoModelForCausalLM.from_pretrained`
* **Affected Pipeline**: `03_qlora_training.ipynb`, `05_export_model.ipynb`, `training/scripts/train.py`, `training/scripts/merge_adapter.py`
* **Severity**: Low (Deprecation Warning)
* **Empirical Evidence**:
  - Console warning during model loading:
    `[transformers] torch_dtype is deprecated! Use dtype instead!`
* **Root Cause Analysis**:
  Transformers 5.x standardized parameter naming across model loaders, deprecating `torch_dtype` in favor of `dtype`.
* **Workaround / Recommended Action**:
  Replace `torch_dtype=compute_dtype` with `dtype=compute_dtype` in all `from_pretrained` calls.

---

## Issue 3: `evaluation_strategy` Deprecation in `TrainingArguments`

* **Component**: `transformers.TrainingArguments`
* **Affected Pipeline**: `03_qlora_training.ipynb`, `training/scripts/train.py`
* **Severity**: Low (Deprecation Warning)
* **Empirical Evidence**:
  - Transformers 4.41+ and 5.x deprecates `evaluation_strategy` keyword argument in favor of `eval_strategy`.
* **Root Cause Analysis**:
  Hugging Face renamed `evaluation_strategy` to `eval_strategy` for consistency with `save_strategy` and `logging_strategy`.
* **Workaround / Recommended Action**:
  Pass `eval_strategy="epoch"` instead of `evaluation_strategy="epoch"`.

---

## Issue 4: `torch.cuda.amp.GradScaler` Deprecation in PyTorch 2.3+

* **Component**: `torch.cuda.amp.GradScaler`
* **Affected Pipeline**: `accelerate`, PyTorch training loops
* **Severity**: Low (Deprecation Warning)
* **Empirical Evidence**:
  - Console warning:
    `FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated. Please use torch.amp.GradScaler('cuda', args...) instead.`
* **Root Cause Analysis**:
  PyTorch unified automatic mixed precision (AMP) across CUDA, CPU, and XPU backends under `torch.amp.GradScaler(device_type, ...)`.
* **Workaround / Recommended Action**:
  Managed automatically by `Accelerate` in updated releases (`accelerate >= 1.0.0`).
