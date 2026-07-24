# Future Upgrade Checklist & Dependency Roadmap — LLMOps Studio

This document outlines the upgrade evaluation roadmap for core ecosystem dependencies. No upgrades should be performed immediately; this document serves as an evaluation framework for future releases.

---

## 1. Upgrade Roadmap by Component

### Component 1: Hugging Face TRL (`trl`)
* **Current Version**: `1.8.0` / `1.9.0`
* **Latest Version**: `1.9.0`
* **Breaking Changes**: Refactored `DistillationTrainer`, introduced optional `train_dataset` in `environment_factory` setups, default loss shift to `"chunked_nll"`.
* **Migration Difficulty**: Low
* **Affected Notebooks**: `03_qlora_training.ipynb`
* **Required Code Changes**: Ensure `SFTTrainer` adapter parameter dtype bug is guarded via `trl_qlora_fp16.patch` or post-init parameter re-alignment.
* **Risk Level**: Medium
* **Testing Requirements**: Run full QLoRA training loop on Tesla T4 (`fp16=True`) and verify backward pass completion without `GradScaler` exception.
* **Upgrade Recommendation**: **RECOMMENDED WITH PATCH** (v1.9.0 includes 30-50% VRAM savings via `chunked_nll` loss).

---

### Component 2: Hugging Face PEFT (`peft`)
* **Current Version**: `0.19.1`
* **Latest Version**: `0.19.1`
* **Breaking Changes**: None in 0.19.x series. Introduced `autocast_adapter_dtype` kwarg in `LoraConfig` (v0.12.0+).
* **Migration Difficulty**: Low
* **Affected Notebooks**: `03_qlora_training.ipynb`, `04_evaluation.ipynb`, `05_export_model.ipynb`
* **Required Code Changes**: None required for 0.19.1.
* **Risk Level**: Low
* **Testing Requirements**: Run `merge_and_unload()` and test checkpoint serialization.
* **Upgrade Recommendation**: **RECOMMENDED** (Current version `0.19.1` is fully stable).

---

### Component 3: Hugging Face Transformers (`transformers`)
* **Current Version**: `5.13.1` (or 4.45.0+)
* **Latest Version**: `5.14.1`
* **Breaking Changes**: Deprecation of `torch_dtype` in favor of `dtype` in `from_pretrained`; deprecation of `evaluation_strategy` in favor of `eval_strategy`.
* **Migration Difficulty**: Low
* **Affected Notebooks**: `02_dataset_pipeline.ipynb`, `03_qlora_training.ipynb`, `04_evaluation.ipynb`, `05_export_model.ipynb`
* **Required Code Changes**: Update `torch_dtype` $\rightarrow$ `dtype` and `evaluation_strategy` $\rightarrow$ `eval_strategy`.
* **Risk Level**: Low
* **Testing Requirements**: Test tokenizer chat template formatting and causal LM generation pipelines.
* **Upgrade Recommendation**: **RECOMMENDED** (Provides latest model architecture support and FlashAttention optimizations).

---

### Component 4: Hugging Face Accelerate (`accelerate`)
* **Current Version**: `1.14.0`
* **Latest Version**: `1.14.0`
* **Breaking Changes**: None.
* **Migration Difficulty**: Very Low
* **Affected Notebooks**: `03_qlora_training.ipynb`
* **Required Code Changes**: None.
* **Risk Level**: Very Low
* **Testing Requirements**: Verify gradient accumulation and multi-GPU state resolution.
* **Upgrade Recommendation**: **RECOMMENDED** (Current version `1.14.0` is stable).

---

### Component 5: BitsAndBytes (`bitsandbytes`)
* **Current Version**: `0.49.2`
* **Latest Version**: `0.49.2`
* **Breaking Changes**: Deprecation of `_check_is_size` in PyTorch 2.9+ integration.
* **Migration Difficulty**: Low
* **Affected Notebooks**: `03_qlora_training.ipynb`
* **Required Code Changes**: None.
* **Risk Level**: Low
* **Testing Requirements**: Test 4-bit NF4 layer initialization and CUDA memory footprint.
* **Upgrade Recommendation**: **RECOMMENDED** (Current version `0.49.2` supports latest PyTorch 2.x CUDA kernels).

---

## 2. Future Upgrade Pre-Flight Checklist

Before performing any dependency upgrades:
- [ ] Backup current virtual environment or export `pip freeze > baseline_requirements.txt`.
- [ ] Run `pytest tests/` to confirm all unit tests pass on the baseline environment.
- [ ] Verify hardware GPU compute capability (`get_gpu_info()`).
- [ ] Verify `trl_qlora_fp16.patch` is applied if upgrading `trl` beyond `1.8.0`.
- [ ] Execute `03_qlora_training.ipynb` for 3 training steps to verify `GradScaler` stability.
