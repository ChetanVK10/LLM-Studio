# Stable Environment v1.0 — LLMOps Studio

This document defines the official **Stable Environment v1.0** specification for the LLMOps Studio platform. All pipeline scripts, training routines, dataset processors, and notebook workflows are validated against this exact runtime baseline.

---

## 1. System & Hardware Baseline

| Specification | Environment Target |
| :--- | :--- |
| **Operating System** | Linux (Ubuntu 22.04 LTS / Colab Runtime) / Windows 11 |
| **Python Version** | `3.12.x` (Tested on 3.12.13) |
| **PyTorch Version** | `2.11.0+cu128` |
| **CUDA Toolkit** | `12.8` |
| **Primary Target GPU** | NVIDIA Tesla T4 (Turing Architecture, CUDA Compute Capability 7.5) |
| **VRAM Availability** | 16 GB GDDR6 |
| **Hardware BF16 Support** | **NO** (Requires Compute Capability $\ge 8.0$) |
| **Default Precision Mode** | FP16 Mixed Precision (`fp16=True`, `bf16=False`) |

---

## 2. Core Dependency Matrix (v1.0 Baseline)

| Package | Tested Version | Minimum Version | Primary Usage in Project |
| :--- | :--- | :--- | :--- |
| `transformers` | **5.13.1** | `4.45.0` | Model loading, Tokenization, Base `Trainer`, `TrainingArguments` |
| `trl` | **1.8.0** / **1.9.0** | `0.11.0` | Supervised Fine-Tuning (`SFTTrainer`, `SFTConfig`) |
| `peft` | **0.19.1** | `0.12.0` | LoRA/QLoRA adapter initialization (`LoraConfig`, `get_peft_model`) |
| `accelerate` | **1.14.0** | `0.34.0` | Mixed-precision gradient accumulation, `Accelerator` state |
| `bitsandbytes` | **0.49.2** | `0.43.0` | NF4 4-bit quantization (`BitsAndBytesConfig`, `Linear4bit`) |
| `datasets` | **5.0.0** | `3.0.0` | Arrow-backed JSONL dataset streaming and batch formatting |
| `pandas` | **2.2.x** | `2.0.0` | Raw dataset cleaning, null filtering, deduplication |
| `rouge-score` | **0.1.2** | `0.1.2` | Post-fine-tuning evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L) |
| `pydantic` | **2.13.4** | `2.7.0` | Schema validation for backend API configurations |
| `streamlit` | **1.35.0+** | `1.35.0` | Frontend dashboard and interactive evaluation UI |

---

## 3. Precision Solvers & Hardware Constraints

1. **Compute Dtype Resolution**:
   - For **NVIDIA Tesla T4** / Turing GPUs: `compute_dtype = torch.float16`.
   - For **NVIDIA Ampere / Ada / Hopper** GPUs: `compute_dtype = torch.bfloat16`.
2. **GradScaler Requirement**:
   - `fp16=True` training utilizes PyTorch's `GradScaler` to scale loss values and prevent underflow during backpropagation.
   - `GradScaler` requires all trainable gradients to be `torch.float32` or `torch.float16`. `torch.bfloat16` gradients trigger a runtime exception.
