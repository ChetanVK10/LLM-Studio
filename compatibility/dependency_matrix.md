# Dependency & API Compatibility Matrix — LLMOps Studio

This document maps all external library dependencies across the project and provides an empirical API compatibility audit based on official documentation (`compatibility_plan.md`) and release notes.

---

## 1. Project-Wide Library Dependency Mapping

| External Library | Import Module Name | Notebooks / Modules Consuming Dependency | Minimum Version | Tested Version |
| :--- | :--- | :--- | :--- | :--- |
| **Transformers** | `transformers` | `01_environment_setup.ipynb`, `02_dataset_pipeline.ipynb`, `03_qlora_training.ipynb`, `04_evaluation.ipynb`, `05_export_model.ipynb`, `train.py`, `evaluate.py`, `merge_adapter.py`, `utils.py` | `4.45.0` | `5.13.1` |
| **TRL** | `trl` | `01_environment_setup.ipynb`, `03_qlora_training.ipynb`, `train.py`, `utils.py` | `0.11.0` | `1.8.0` / `1.9.0` |
| **PEFT** | `peft` | `01_environment_setup.ipynb`, `03_qlora_training.ipynb`, `04_evaluation.ipynb`, `05_export_model.ipynb`, `train.py`, `evaluate.py`, `merge_adapter.py`, `export_adapter.py` | `0.12.0` | `0.19.1` |
| **Accelerate** | `accelerate` | `01_environment_setup.ipynb`, `03_qlora_training.ipynb`, `train.py`, `utils.py` | `0.34.0` | `1.14.0` |
| **BitsAndBytes** | `bitsandbytes` | `01_environment_setup.ipynb`, `03_qlora_training.ipynb`, `train.py`, `utils.py` | `0.43.0` | `0.49.2` |
| **Datasets** | `datasets` | `01_environment_setup.ipynb`, `02_dataset_pipeline.ipynb`, `03_qlora_training.ipynb`, `04_evaluation.ipynb`, `train.py`, `evaluate.py` | `3.0.0` | `5.0.0` |
| **PyTorch** | `torch` | All Notebooks (1–5), `train.py`, `evaluate.py`, `merge_adapter.py`, `export_adapter.py`, `utils.py` | `2.0.0` | `2.11.0+cu128` |
| **Pandas** | `pandas` | `02_dataset_pipeline.ipynb`, `train.py` | `2.0.0` | `2.2.x` |
| **Rouge-Score** | `rouge_score` | `04_evaluation.ipynb`, `evaluate.py` | `0.1.2` | `0.1.2` |

---

## 2. Comprehensive API Compatibility Audit

| Library | API Symbol / Function / Method | Usage Location | Compatibility Status | Technical Findings & Documentation Support |
| :--- | :--- | :--- | :--- | :--- |
| **Transformers** | `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)` | Notebook 3, 5; `train.py`, `merge_adapter.py` | **Deprecated** | `torch_dtype` is deprecated in Transformers 5.x. Release notes recommend passing `dtype=...` instead. Emits warning: `[transformers] torch_dtype is deprecated! Use dtype instead!`. |
| **Transformers** | `TrainingArguments(..., evaluation_strategy=...)` | Notebook 3; `train.py` | **Deprecated** | `evaluation_strategy` is deprecated in favor of `eval_strategy` in Transformers 4.41+ and 5.x. |
| **Transformers** | `AutoTokenizer.from_pretrained(...)` | Notebook 2, 3, 4, 5; `train.py`, `evaluate.py` | **Still Supported** | No changes required. Works across 4.x and 5.x. |
| **Transformers** | `tokenizer.apply_chat_template(...)` | Notebook 2, 3, 4; `train.py`, `evaluate.py` | **Still Supported** | Standard ChatML formatting interface. |
| **TRL** | `SFTTrainer(model=..., processing_class=...)` | Notebook 3; `train.py`, `utils.py` | **Changed Behavior** | In TRL 0.11.0+, `processing_class` is preferred over the deprecated `tokenizer` keyword argument. `utils.get_sft_trainer_kwargs()` handles this dynamically. |
| **TRL** | `SFTConfig(...)` | Notebook 3; `train.py` | **Still Supported** | Subclasses `TrainingArguments`. Default `loss_type` shifted to `"chunked_nll"` in v1.7.0 for 30%-50% VRAM reduction. |
| **TRL** | Internal `SFTTrainer.__init__` parameter cast | Notebook 3; `train.py` | **Breaking Change / Known Bug** | In TRL 1.8.0+, `SFTTrainer.__init__` forcibly executes `param.data = param.data.to(torch.bfloat16)` when `_is_quantized_model=True`. Under `fp16=True`, this causes autograd to output `bfloat16` gradients, crashing `GradScaler` on GPUs without native BF16 (e.g. Tesla T4). |
| **PEFT** | `LoraConfig(...)` | Notebook 3; `train.py` | **Still Supported** | PEFT 0.12.0+ introduced `autocast_adapter_dtype`. Setting `autocast_adapter_dtype=False` keeps adapter weights aligned. |
| **PEFT** | `get_peft_model(model, peft_config)` | Notebook 3; `train.py` | **Still Supported** | Standard PEFT model wrapping interface. |
| **PEFT** | `cast_mixed_precision_params(model, dtype=...)` | Notebook 3; `train.py` | **Still Supported** | Official PEFT 0.19.1 utility for casting trainable adapter parameters. |
| **PEFT** | `PeftModel.from_pretrained(base_model, adapter_dir)` | Notebook 4, 5; `evaluate.py`, `merge_adapter.py` | **Still Supported** | Standard adapter loading entry point. |
| **PEFT** | `model.merge_and_unload()` | Notebook 5; `merge_adapter.py` | **Still Supported** | Standard interface to fuse LoRA weights into base model. |
| **BitsAndBytes**| `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` | Notebook 3; `train.py` | **Still Supported** | Configures 4-bit NormalFloat quantization. `bnb_4bit_compute_dtype` must match target GPU capability. |
| **Accelerate** | `Accelerator()` / `trainer.accelerator` | Notebook 3; `train.py` | **Still Supported** | Manages mixed precision states (`fp16`/`bf16`). |
| **PyTorch** | `torch.cuda.amp.GradScaler()` | Notebook 3; `train.py` | **Deprecated** | PyTorch 2.3+ deprecates `torch.cuda.amp.GradScaler` in favor of `torch.amp.GradScaler('cuda')`. |
