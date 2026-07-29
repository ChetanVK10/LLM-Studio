"""TRL Precision Compatibility Module.

Provides precision alignment guards for Hugging Face TRL SFTTrainer.
Resolves the upstream TRL issue where SFTTrainer.__init__() unconditionally
mutates trainable parameters to torch.bfloat16 when training quantized models,
causing GradScaler crashes on pre-Ampere GPUs (e.g., Tesla T4).
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union
import torch

try:
    import trl
    TRL_VERSION = getattr(trl, "__version__", "unknown")
except ImportError:
    TRL_VERSION = "not_installed"

logger = logging.getLogger("trainer.compatibility")


def validate_environment(
    trainer: Any,
    target_dtype: Optional[torch.dtype] = None
) -> Dict[str, Any]:
    """Inspects system hardware, TRL version, and model precision state.

    Args:
        trainer: An instantiated SFTTrainer or Hugging Face Trainer.
        target_dtype: Target torch.dtype for trainable parameters. Defaults to torch.float32 for FP16 mixed precision.

    Returns:
        Dict containing environment profile metrics and a `fix_required` boolean flag.

    Raises:
        ValueError: If trainer or model reference is invalid.
        RuntimeError: If requesting BF16 precision on unsupported hardware (e.g. Tesla T4).
    """
    if trainer is None or not hasattr(trainer, "model"):
        raise ValueError("Invalid trainer reference: missing 'model' attribute.")

    model = trainer.model

    # Training arguments inspection
    args = getattr(trainer, "args", None)
    cfg_fp16 = getattr(args, "fp16", False) if args else False
    cfg_bf16 = getattr(args, "bf16", False) if args else False

    # Single Source of Truth for Target Dtype:
    # 1. If explicitly provided by caller, honor it.
    # 2. If bf16=True in training args, target torch.bfloat16.
    # 3. If fp16=True or default mixed precision, PyTorch GradScaler requires FP32 master weights (torch.float32).
    if target_dtype is None:
        if cfg_bf16:
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

    # Hardware & CUDA detection
    cuda_available = torch.cuda.is_available()
    gpu_name = "N/A"
    compute_capability = "N/A"
    bf16_supported = False

    if cuda_available and torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        compute_capability = f"{cap[0]}.{cap[1]}"
        bf16_supported = torch.cuda.is_bf16_supported()

    if cfg_bf16 and not bf16_supported:
        raise RuntimeError(
            f"Incompatible Precision Error: Training requested 'bf16=True' but GPU '{gpu_name}' "
            f"(Compute Capability {compute_capability}) does not support BFloat16 hardware instructions.\n"
            f"[HOW TO FIX]: Set 'bf16: false' and 'fp16: true' inside your training configuration."
        )

    # Model quantization & PEFT detection
    is_4bit = getattr(model, "is_loaded_in_4bit", False)
    is_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_quantized = is_4bit or is_8bit

    # Parameter inspection
    total_trainable = 0
    bf16_trainable = 0
    target_matched_trainable = 0
    deviating_trainable = 0

    for param in model.parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.dtype == torch.bfloat16:
                bf16_trainable += 1
            if param.dtype == target_dtype:
                target_matched_trainable += 1
            else:
                deviating_trainable += 1

    # Evaluate whether fix is required:
    # Fix is required if there are trainable parameters whose precision deviates from target_dtype.
    # If a future TRL release fixes line 1130, deviating_trainable will be 0, so fix_required automatically becomes False.
    fix_required = (not cfg_bf16) and (deviating_trainable > 0)

    return {
        "trl_version": TRL_VERSION,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "bf16_supported": bf16_supported,
        "is_quantized": is_quantized,
        "cfg_fp16": cfg_fp16,
        "cfg_bf16": cfg_bf16,
        "target_dtype": target_dtype,
        "total_trainable": total_trainable,
        "bf16_trainable": bf16_trainable,
        "target_matched_trainable": target_matched_trainable,
        "fix_required": fix_required
    }


def apply_trl_precision_fix(
    trainer: Any,
    target_dtype: Optional[torch.dtype] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Applies the precision alignment guard to trainer.model post-SFTTrainer initialization.

    Converts trainable parameters (requires_grad=True) that were mutated to bfloat16
    or set to float16 to full precision (torch.float32), ensuring PyTorch GradScaler stability.

    Args:
        trainer: Instantiated SFTTrainer object before trainer.train() is called.
        target_dtype: Desired target compute precision for trainable parameters. Defaults to torch.float32 for FP16.
        verbose: If True, prints a formatted diagnostic report to stdout and logger.

    Returns:
        Dict containing scan metrics: total_trainable, converted, already_correct, skipped, status.
    """
    profile = validate_environment(trainer, target_dtype)
    target_dtype = profile["target_dtype"]
    model = trainer.model

    metrics = {
        "total_trainable": profile["total_trainable"],
        "already_correct": 0,
        "converted": 0,
        "skipped": 0,
        "fix_applied": False,
        "status": "PASS"
    }

    if not profile["fix_required"]:
        # Count current state
        for param in model.parameters():
            if param.requires_grad:
                if param.dtype == target_dtype:
                    metrics["already_correct"] += 1
                else:
                    metrics["skipped"] += 1

        if verbose:
            _log_diagnostic_report(profile, metrics, fix_needed=False)
        return metrics

    # Perform targeted parameter conversion
    converted_count = 0
    already_correct_count = 0
    skipped_count = 0

    for param in model.parameters():
        if param.requires_grad:
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                converted_count += 1
            else:
                already_correct_count += 1

    # Post-condition verification assertion
    mismatched = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.dtype != target_dtype
    )
    if mismatched > 0:
        metrics["status"] = "FAIL"
        raise RuntimeError(
            f"Compatibility Layer Verification Error: Found {mismatched} trainable parameters "
            f"that could not be aligned to target precision {target_dtype}."
        )

    metrics["converted"] = converted_count
    metrics["already_correct"] = already_correct_count
    metrics["skipped"] = skipped_count
    metrics["fix_applied"] = True

    if verbose:
        _log_diagnostic_report(profile, metrics, fix_needed=True)

    return metrics


def _log_diagnostic_report(
    profile: Dict[str, Any],
    metrics: Dict[str, Any],
    fix_needed: bool
) -> None:
    """Prints and logs a formatted diagnostic summary table."""
    lines = [
        "============================================================",
        "          LLMOps Studio - TRL Precision Compatibility       ",
        "============================================================",
        f" TRL Version        : {profile['trl_version']}",
        f" GPU Device         : {profile['gpu_name']} (Compute Cap: {profile['compute_capability']})",
        f" Native BF16 Support: {'YES' if profile['bf16_supported'] else 'NO'}",
        f" Quantized Model    : {'YES' if profile['is_quantized'] else 'NO'}",
        f" Target Precision   : {profile['target_dtype']}",
        f" Fix Required       : {'YES' if fix_needed else 'NO (Auto-Bypassed)'}",
        "------------------------------------------------------------",
        " Parameter Scan Results:",
        f"   Trainable Params Found : {metrics['total_trainable']}",
        f"   Already Correct        : {metrics['already_correct']}",
        f"   Converted to Target    : {metrics['converted']}",
        f"   Skipped / Preserved    : {metrics['skipped']}",
        f" Verification Status      : {metrics['status']}",
        "============================================================"
    ]
    report = "\n".join(lines)
    logger.info(report)
    print(report)
