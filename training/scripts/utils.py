"""LLMOps Studio Training Shared Utilities.

Provides version diagnostics, system capability audits, defensive schema validations,
global random seed utilities, and state checker constraints for training pipelines.
"""

import sys
import json
import logging
import importlib
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger("trainer.utils")

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parses a version string into a clean numeric tuple, stripping suffixes."""
    try:
        clean_version = version_str.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        parts = []
        for p in clean_version.split("."):
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(0)
        return tuple(parts)
    except Exception:
        return (0,)

def validate_dependencies() -> list:
    """Validates that the installed library versions are within the project's tested compatibility range.

    Returns:
        List of warning strings. Emits recommendations if unvalidated versions are found.
    """
    ranges = {
        "torch": ((2, 0), (2, 15)),
        "transformers": ((4, 40), (5, 15)),
        "trl": ((0, 8), (1, 10)),
        "peft": ((0, 10), (0, 20)),
        "accelerate": ((0, 30), (1, 20)),
        "bitsandbytes": ((0, 40), (0, 55)),
        "datasets": ((2, 18), (6, 0))
    }
    
    warnings = []
    for lib, (min_v, max_v) in ranges.items():
        try:
            mod = importlib.import_module(lib)
            version_str = getattr(mod, "__version__", None)
            if not version_str:
                warnings.append(f"Compatibility Warning: Library '{lib}' does not expose a __version__ attribute.")
                continue
            v_tuple = parse_version(version_str)
            
            if v_tuple < min_v:
                warnings.append(
                    f"Compatibility Warning: Installed '{lib}' version ({version_str}) is below the tested minimum "
                    f"({'.'.join(map(str, min_v))}). [RECOMMENDATION]: Upgrade via: pip install {lib}>={'.'.join(map(str, min_v))}"
                )
            if v_tuple > max_v:
                warnings.append(
                    f"Compatibility Warning: Installed '{lib}' version ({version_str}) is above the tested maximum "
                    f"({'.'.join(map(str, max_v))}). You may encounter API deprecations or compatibility issues."
                )
        except ImportError:
            warnings.append(f"Compatibility Warning: Mandatory library '{lib}' is not installed in the current environment.")
            
    return warnings

def set_global_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, PyTorch, and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass

def get_gpu_info() -> Dict[str, Any]:
    """Gathers detailed hardware capabilities and CUDA specifications."""
    gpu_available = HAS_TORCH and torch.cuda.is_available()
    devices = []
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            # Native support for bf16 requires compute capability >= 8.0 and PyTorch support
            bf16_supported = (major >= 8) and torch.cuda.is_bf16_supported()
            devices.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "major_capability": major,
                "minor_capability": minor,
                "bf16_supported": bf16_supported
            })
    return {
        "cuda_available": gpu_available,
        "device_count": len(devices),
        "devices": devices,
        "cuda_version": torch.version.cuda if (HAS_TORCH and gpu_available) else "N/A"
    }

def resolve_precision_mode(train_cfg: Dict[str, Any], gpu_info: Dict[str, Any]) -> Tuple[bool, bool, Any, str]:
    """Resolves mixed precision training flags and compute dtypes.

    Raises:
        ValueError: For conflicting fp16/bf16 requests.
        RuntimeError: If requesting bf16 on unsupported hardware.
    """
    cfg_fp16 = bool(train_cfg.get("fp16", False))
    cfg_bf16 = bool(train_cfg.get("bf16", False))

    if cfg_fp16 and cfg_bf16:
        raise ValueError(
            "Configuration Error: 'fp16' and 'bf16' cannot both be True in training configurations.\n"
            "[HOW TO FIX]: Open training/configs/qlora_config.yaml and verify only one flag is set to true."
        )

    if not HAS_TORCH or not gpu_info["cuda_available"]:
        fp32_dtype = torch.float32 if HAS_TORCH else "float32"
        return False, False, fp32_dtype, "FP32 (CPU Mode)"

    primary_gpu = gpu_info["devices"][0]
    hw_bf16_supported = primary_gpu["bf16_supported"]

    if cfg_bf16:
        if not hw_bf16_supported:
            raise RuntimeError(
                f"Precision Error: Config requests 'bf16: true' but GPU '{primary_gpu['name']}' "
                f"(Compute Capability {primary_gpu['major_capability']}.{primary_gpu['minor_capability']}) "
                f"does not natively support BFloat16 precision.\n"
                f"[HOW TO FIX]: Set 'bf16: false' and 'fp16: true' inside training/configs/qlora_config.yaml."
            )
        return False, True, torch.bfloat16, "BF16 (Config-Requested)"

    if cfg_fp16:
        return True, False, torch.float16, "FP16 (Config-Requested)"

    return False, False, torch.float32, "FP32 (No Mixed Precision)"

def validate_config_schema(config: Dict[str, Any], project_root: Path, runtime_root: Path = None) -> None:
    """Performs defensive validation on top-level keys, range constraints, and directory paths."""
    target_runtime_root = runtime_root if runtime_root is not None else project_root

    # 1. Required Top-Level Sections
    required_sections = ["model", "dataset", "lora", "training"]
    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Configuration Error: Missing top-level section '{section}'.\n"
                f"[HOW TO FIX]: Ensure your qlora_config.yaml config contains the Top-Level '{section}:' key."
            )
            
    # 2. Validate model section
    model_cfg = config["model"]
    assert "base_model_name" in model_cfg and model_cfg["base_model_name"], (
        "Configuration Error: 'model.base_model_name' cannot be empty.\n"
        "[HOW TO FIX]: Specify a valid Hugging Face path (e.g. Qwen/Qwen2.5-3B-Instruct) in qlora_config.yaml."
    )
    
    # 3. Validate dataset section
    dataset_cfg = config["dataset"]
    required_dataset_keys = ["raw_path", "train_path", "val_path", "test_split_ratio", "columns"]
    for k in required_dataset_keys:
        assert k in dataset_cfg, f"Configuration Error: dataset section is missing required key '{k}'."
        
    assert isinstance(dataset_cfg["test_split_ratio"], (int, float)) and 0.0 < dataset_cfg["test_split_ratio"] < 1.0, (
        f"Configuration Error: 'dataset.test_split_ratio' must be a float between 0.0 and 1.0 (got {dataset_cfg['test_split_ratio']}).\n"
        f"[HOW TO FIX]: Edit test_split_ratio in qlora_config.yaml (e.g. 0.2)."
    )
    
    # 4. Validate lora section
    lora_cfg = config["lora"]
    required_lora_keys = ["r", "lora_alpha", "target_modules", "task_type"]
    for k in required_lora_keys:
        assert k in lora_cfg, f"Configuration Error: lora section is missing required key '{k}'."
        
    assert isinstance(lora_cfg["r"], int) and lora_cfg["r"] >= 1, (
        f"Configuration Error: LoRA rank 'r' must be a positive integer >= 1 (got {lora_cfg['r']})."
    )
    assert isinstance(lora_cfg["lora_alpha"], int) and lora_cfg["lora_alpha"] >= 1, (
        f"Configuration Error: LoRA alpha scaling factor must be a positive integer >= 1 (got {lora_cfg['lora_alpha']})."
    )
    assert isinstance(lora_cfg["target_modules"], list) and len(lora_cfg["target_modules"]) > 0, (
        "Configuration Error: 'lora.target_modules' cannot be empty."
    )
    
    # 5. Validate training section
    train_cfg = config["training"]
    required_training_keys = [
        "output_dir", "experiments_dir", "num_train_epochs", 
        "per_device_train_batch_size", "learning_rate", "fp16", "bf16"
    ]
    for k in required_training_keys:
        assert k in train_cfg, f"Configuration Error: training section is missing required key '{k}'."
        
    assert isinstance(train_cfg["num_train_epochs"], int) and train_cfg["num_train_epochs"] >= 1, (
        f"Configuration Error: 'training.num_train_epochs' must be an integer >= 1 (got {train_cfg['num_train_epochs']})."
    )
    assert isinstance(train_cfg["per_device_train_batch_size"], int) and train_cfg["per_device_train_batch_size"] >= 1, (
        f"Configuration Error: 'training.per_device_train_batch_size' must be an integer >= 1 (got {train_cfg['per_device_train_batch_size']})."
    )
    assert isinstance(train_cfg["learning_rate"], (int, float)) and train_cfg["learning_rate"] > 0.0, (
        f"Configuration Error: 'training.learning_rate' must be a positive float (got {train_cfg['learning_rate']})."
    )

    # Verify source directories exist under project_root
    for source_folder in ["training", "training/configs", "training/scripts"]:
        src_path = project_root / source_folder
        assert src_path.exists(), f"Configuration Error: Missing source folder under PROJECT_ROOT: '{src_path}'."
    
    # 6. Verify filesystem write accessibility and auto-create runtime directories under runtime_root
    for folder_key in ["output_dir", "experiments_dir"]:
        dir_path = target_runtime_root / train_cfg[folder_key]
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(
                f"System Error: Path training.{folder_key} '{dir_path}' is not writable.\n"
                f"Details: {e}\n"
                f"[HOW TO FIX]: Check storage folders permission settings inside workspace directory."
            )

def validate_dataset_schema(dataset_path: Path) -> None:
    """Verifies format turns schema compliance in target JSONL datasets."""
    assert dataset_path.exists(), (
        f"Dataset Error: File does not exist at '{dataset_path}'.\n"
        f"[HOW TO FIX]: Ensure Notebook 2 has been executed to generate processed splits."
    )
    assert dataset_path.stat().st_size > 0, f"Dataset Error: File at '{dataset_path}' is empty."
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Dataset Format Error: File '{dataset_path.name}' contains invalid JSON at line {idx+1}.\n"
                    f"Details: {e}"
                )
            
            assert "messages" in record, (
                f"Dataset Schema Error: Missing 'messages' field in '{dataset_path.name}' at line {idx+1}.\n"
                f"[HOW TO FIX]: Verify your conversation dictionary contains the root 'messages' list key."
            )
            messages = record["messages"]
            assert isinstance(messages, list), (
                f"Dataset Schema Error: 'messages' at line {idx+1} must be a list (got {type(messages).__name__})."
            )
            assert len(messages) >= 2, (
                f"Dataset Schema Error: 'messages' list at line {idx+1} has only {len(messages)} turns. "
                f"At least 'user' and 'assistant' are required."
            )
            
            for turn_idx, turn in enumerate(messages):
                assert isinstance(turn, dict), (
                    f"Dataset Schema Error: message turn {turn_idx} at line {idx+1} must be a dict (got {type(turn).__name__})."
                )
                assert "role" in turn and "content" in turn, (
                    f"Dataset Schema Error: turn {turn_idx} at line {idx+1} is missing 'role' or 'content'. Keys: {list(turn.keys())}."
                )
                assert turn["role"] in ["system", "user", "assistant"], (
                    f"Dataset Schema Error: Invalid role '{turn['role']}' in turn {turn_idx} at line {idx+1}."
                )
                assert isinstance(turn["content"], str) and turn["content"].strip(), (
                    f"Dataset Schema Error: content must be non-empty string in turn {turn_idx} at line {idx+1}."
                )

def validate_model_compatibility(model, tokenizer) -> None:
    """Verifies compatibility between causal models, quantizations, and tokenizers."""
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from peft import PeftModel
    
    # 1. Base model validation
    base_model = model.base_model.model if isinstance(model, PeftModel) else model
    assert isinstance(base_model, PreTrainedModel), (
        f"Model Error: Model object class ({type(base_model).__name__}) must inherit from PreTrainedModel.\n"
        f"[HOW TO FIX]: Ensure you instantiate using AutoModelForCausalLM.from_pretrained()."
    )
    
    # 2. Tokenizer validation
    assert isinstance(tokenizer, PreTrainedTokenizerBase), (
        f"Tokenizer Error: Tokenizer object class ({type(tokenizer).__name__}) must inherit from PreTrainedTokenizerBase."
    )
    
    # 3. Check vocabulary embedding bounds
    model_vocab_size = getattr(base_model.config, "vocab_size", None)
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size is not None:
        embeddings_count = base_model.get_input_embeddings().num_embeddings
        assert tokenizer_vocab_size <= embeddings_count, (
            f"Tokenizer Mismatch: Tokenizer vocab size ({tokenizer_vocab_size}) exceeds model embedding size ({embeddings_count}).\n"
            f"[HOW TO FIX]: Run: model.resize_token_embeddings(len(tokenizer)) before wrapping in PEFT."
        )
        
    # 4. Check padding alignment
    assert tokenizer.pad_token_id is not None, (
        "Tokenizer Error: Tokenizer 'pad_token' has not been configured.\n"
        "[HOW TO FIX]: Run: tokenizer.pad_token = tokenizer.eos_token"
    )
    assert base_model.config.pad_token_id == tokenizer.pad_token_id, (
        f"Model Mismatch: Model pad_token_id ({base_model.config.pad_token_id}) "
        f"is out of sync with tokenizer pad_token_id ({tokenizer.pad_token_id}).\n"
        f"[HOW TO FIX]: Run: model.config.pad_token_id = tokenizer.pad_token_id"
    )

def validate_checkpoint_integrity(checkpoint_dir: Path) -> bool:
    """Verifies that the checkpoints directory contains required files to resume training."""
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return False
        
    # Standard QLoRA checkpoints file requirements
    has_adapter_config = (checkpoint_dir / "adapter_config.json").exists()
    has_adapter_model = (checkpoint_dir / "adapter_model.safetensors").exists() or (checkpoint_dir / "adapter_model.bin").exists()
    
    has_trainer_state = (checkpoint_dir / "trainer_state.json").exists()
    has_optimizer = (checkpoint_dir / "optimizer.pt").exists()
    has_scheduler = (checkpoint_dir / "scheduler.pt").exists()
    
    # Check intermediate resume requirements
    is_intermediate = checkpoint_dir.name.startswith("checkpoint-") or has_trainer_state
    if is_intermediate:
        return has_adapter_config and has_adapter_model and has_trainer_state and has_optimizer and has_scheduler
    else:
        return has_adapter_config and has_adapter_model

def validate_notebook_state(
    model,
    tokenizer,
    peft_config,
    config: Dict[str, Any],
    gpu_info: Dict[str, Any],
    project_root: Path,
    runtime_root: Path = None
) -> None:
    """Defensively validates all state variables, path existences, and precision constraints before training starts."""
    from peft import PeftModel
    target_runtime_root = runtime_root if runtime_root is not None else project_root
    
    # 1. Validate files
    dataset_cfg = config.get("dataset", {})
    train_path = target_runtime_root / dataset_cfg.get("train_path", "")
    val_path = target_runtime_root / dataset_cfg.get("val_path", "")
    
    try:
        validate_dataset_schema(train_path)
        validate_dataset_schema(val_path)
    except AssertionError as e:
        raise AssertionError(
            f"{e}\n[HOW TO FIX]: Ensure you have executed Notebook 2 (Dataset Pipeline) successfully."
        ) from e
        
    # 2. Tokenizer and Model basics
    assert tokenizer is not None, (
        "State Error: Tokenizer is not initialized.\n"
        "[HOW TO FIX]: Ensure you executed Cell 6 (Load Model and Tokenizer) in the current session."
    )
    assert model is not None, (
        "State Error: Model is not initialized.\n"
        "[HOW TO FIX]: Ensure you executed Cell 6 (Load Model and Tokenizer) in the current session."
    )
    
    # 3. Model compatibility checks
    validate_model_compatibility(model, tokenizer)
    
    # 4. PEFT checks
    assert isinstance(model, PeftModel), (
        "State Error: Model is not wrapped with a PEFT adapter.\n"
        "[HOW TO FIX]: Ensure you executed Cell 7 (Setup PEFT LoRA Config) which calls get_peft_model()."
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, (
        "State Error: No trainable LoRA parameters found.\n"
        "[HOW TO FIX]: Verify that your target modules (e.g. q_proj, v_proj) are configured correctly in qlora_config.yaml."
    )
    
    # 5. Precision checks
    train_cfg = config.get("training", {})
    fp16_mode, bf16_mode, compute_dtype, precision_desc = resolve_precision_mode(train_cfg, gpu_info)
    
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if fp16_mode:
        assert torch.bfloat16 not in trainable_dtypes, (
            "Precision Conflict: Found bfloat16 trainable parameters while FP16 mode is requested.\n"
            "GradScaler will crash. [HOW TO FIX]: Verify model load compute_dtype matches float16."
        )
        
        # Verify unquantized parameters (embeddings, norms, lm_head) are not loaded in bfloat16
        for name, param in model.named_parameters():
            if not param.requires_grad and param.dtype == torch.bfloat16:
                if any(x in name for x in ["embed_tokens", "lm_head", "norm"]):
                    raise AssertionError(
                        f"Precision Mismatch: Unquantized layer '{name}' is loaded in bfloat16 "
                        f"while training in FP16 mode. This will crash the GradScaler.\n"
                        f"[HOW TO FIX]: Ensure torch_dtype=torch.float16 is passed to AutoModelForCausalLM.from_pretrained()."
                    )
                    
    # Check BNB quantized compute dtype matches trainer compute dtype
    import bitsandbytes as bnb
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            assert module.compute_dtype == compute_dtype, (
                f"Quantization Mismatch: BitsAndBytes layer '{name}' compute_dtype ({module.compute_dtype}) "
                f"does not match trainer compute_dtype ({compute_dtype}).\n"
                f"[HOW TO FIX]: Adjust compute_dtype parameters to match the training config precision."
            )

def log_system_info(config: Dict[str, Any], gpu_info: Dict[str, Any], logger_func=None) -> None:
    """Logs a detailed diagnostic system snapshot before starting training."""
    info_lines = []
    info_lines.append("=" * 50)
    info_lines.append("LLMOps Studio System Diagnostics")
    info_lines.append("=" * 50)
    info_lines.append(f"Python Version      : {sys.version.split()[0]}")
    info_lines.append(f"PyTorch Version     : {torch.__version__ if HAS_TORCH else 'NOT INSTALLED'}")
    
    libraries = ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]
    for lib in libraries:
        try:
            mod = importlib.import_module(lib)
            version = getattr(mod, "__version__", "unknown")
            info_lines.append(f"{lib.capitalize():<20}: {version}")
        except ImportError:
            info_lines.append(f"{lib.capitalize():<20}: NOT INSTALLED")
            
    info_lines.append(f"CUDA Version        : {gpu_info.get('cuda_version', 'N/A')}")
    info_lines.append("-" * 50)
    
    if gpu_info["cuda_available"]:
        for dev in gpu_info["devices"]:
            info_lines.append(f"GPU {dev['index']} Name          : {dev['name']}")
            info_lines.append(f"GPU {dev['index']} Capability    : {dev['major_capability']}.{dev['minor_capability']}")
            info_lines.append(f"GPU {dev['index']} Total Memory  : {dev['total_memory_gb']} GB")
            info_lines.append(f"GPU {dev['index']} BF16 Support  : {'YES' if dev['bf16_supported'] else 'NO'}")
    else:
        info_lines.append("GPU Device          : None (CPU Mode)")
    info_lines.append("-" * 50)
    
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    
    try:
        _, _, _, precision_desc = resolve_precision_mode(train_cfg, gpu_info)
        info_lines.append(f"Precision Mode      : {precision_desc}")
    except Exception as e:
        info_lines.append(f"Precision Mode      : Error resolving: {e}")
        
    info_lines.append(f"Base Model Name     : {model_cfg.get('base_model_name', 'N/A')}")
    info_lines.append(f"Train Dataset Path  : {dataset_cfg.get('train_path', 'N/A')}")
    info_lines.append(f"Val Dataset Path    : {dataset_cfg.get('val_path', 'N/A')}")
    info_lines.append("=" * 50)
    
    full_info = "\n".join(info_lines)
    if logger_func:
        logger_func(full_info)
    else:
        print(full_info)

def save_run_metadata(
    output_path: Path,
    config: Dict[str, Any],
    gpu_info: Dict[str, Any],
    precision_desc: str,
    compute_dtype: str,
    trainable_params: int,
    all_params: int
) -> None:
    """Saves a lightweight run_metadata.json metadata summary for auditing/reproducibility."""
    from datetime import datetime, timezone
    
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": gpu_info.get("cuda_version", "N/A")
    }
    
    libraries = ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]
    for lib in libraries:
        try:
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"
            
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "library_versions": versions,
        "hardware": {
            "cuda_available": gpu_info["cuda_available"],
            "device_count": gpu_info["device_count"],
            "devices": [
                {
                    "name": d["name"],
                    "capability": f"{d['major_capability']}.{d['minor_capability']}",
                    "total_memory_gb": d["total_memory_gb"]
                }
                for d in gpu_info.get("devices", [])
            ]
        },
        "precision": {
            "precision_mode": precision_desc,
            "compute_dtype": str(compute_dtype)
        },
        "model_parameters": {
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_percentage": round(100 * trainable_params / all_params, 4) if all_params > 0 else 0
        },
        "configuration": config
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def get_sft_trainer_kwargs(tokenizer) -> Dict[str, Any]:
    """Inspects TRL's SFTTrainer class signature to maintain version compatibility.

    Maps tokenizer argument dynamically (processing_class for TRL 0.11.0+, tokenizer for older versions).
    """
    import inspect
    from trl import SFTTrainer
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}
