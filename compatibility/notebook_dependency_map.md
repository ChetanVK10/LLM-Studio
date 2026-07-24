# Notebook Dependency & Impact Analysis Map — LLMOps Studio

This document presents a notebook-by-notebook impact analysis covering dependencies, API usage, compatibility status, required changes, priority, risk level, and estimated effort across all training and evaluation notebooks.

---

## 1. Notebook 1: `01_environment_setup.ipynb`

* **Purpose**: System capability diagnostics, library version verification, and GPU compute capability discovery.
* **Current Dependencies**: `sys`, `torch`, `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`.
* **APIs Used**:
  - `torch.cuda.is_available()`, `torch.cuda.get_device_properties()`
  - `torch.__version__`, `transformers.__version__`, `peft.__version__`, `trl.__version__`, `datasets.__version__`, `accelerate.__version__`, `bitsandbytes.__version__`
* **Compatibility Status**: **Fully Compatible / Safe**
* **Required Changes**: None.
* **Priority**: Low
* **Risk Level**: Very Low
* **Estimated Effort**: 0 Hours

---

## 2. Notebook 2: `02_dataset_pipeline.ipynb`

* **Purpose**: Clean raw CSV datasets, remove null/duplicate records, format into Qwen ChatML JSONL splits (`messages` format).
* **Current Dependencies**: `pandas`, `json`, `pathlib`, `datasets`, `transformers`.
* **APIs Used**:
  - `pandas.read_csv`, `df.dropna`, `df.drop_duplicates`
  - `datasets.load_dataset("json", data_files=...)`
  - `AutoTokenizer.from_pretrained(...)`
  - `tokenizer.apply_chat_template(...)`
* **Compatibility Status**: **Fully Compatible / Safe**
* **Required Changes**: None.
* **Priority**: Low
* **Risk Level**: Very Low
* **Estimated Effort**: 0 Hours

---

## 3. Notebook 3: `03_qlora_training.ipynb`

* **Purpose**: Production-hardened 4-bit QLoRA fine-tuning loop for Qwen2.5-3B-Instruct using TRL `SFTTrainer`.
* **Current Dependencies**: `torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`, `yaml`, `json`.
* **APIs Used**:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=...)`
  - `AutoModelForCausalLM.from_pretrained(..., quantization_config=..., torch_dtype=...)`
  - `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)`
  - `LoraConfig(r=..., lora_alpha=..., target_modules=..., task_type=...)`
  - `get_peft_model(model, peft_config)`
  - `cast_mixed_precision_params(model, dtype=...)`
  - `TrainingArguments(..., eval_strategy=...)`
  - `SFTTrainer(model=..., args=..., train_dataset=..., eval_dataset=..., callbacks=..., **sft_kwargs)`
  - `trainer.train()`, `trainer.evaluate()`
* **Compatibility Status**: **Action Required (Known TRL Mixed-Precision Issue)**
* **Required Changes**:
  1. Update `torch_dtype=compute_dtype` in `from_pretrained` to `dtype=compute_dtype` to clear Transformers 5.x deprecation warning.
  2. Implement post-initialization parameter re-alignment (or apply `trl_qlora_fp16.patch`) immediately following `SFTTrainer(...)` instantiation to prevent TRL from forcing `torch.bfloat16` on trainable LoRA adapter weights during FP16 training on Tesla T4 GPUs.
* **Priority**: **High**
* **Risk Level**: Medium
* **Estimated Effort**: 0.5 Hours

---

## 4. Notebook 4: `04_evaluation.ipynb`

* **Purpose**: Quantitative & qualitative evaluation of fine-tuned QLoRA adapters against baseline models using ROUGE scores.
* **Current Dependencies**: `torch`, `transformers`, `peft`, `datasets`, `rouge_score`, `tqdm`.
* **APIs Used**:
  - `AutoModelForCausalLM.from_pretrained(...)`
  - `PeftModel.from_pretrained(base_model, adapter_dir)`
  - `AutoTokenizer.from_pretrained(...)`
  - `tokenizer.apply_chat_template(..., tokenize=False)`
  - `model.generate(...)`
  - `rouge_score.rouge_scorer.RougeScorer(...)`
* **Compatibility Status**: **Fully Compatible / Safe**
* **Required Changes**: None.
* **Priority**: Low
* **Risk Level**: Low
* **Estimated Effort**: 0 Hours

---

## 5. Notebook 5: `05_export_model.ipynb`

* **Purpose**: Merge QLoRA adapter weights back into base model parameters and export standalone sharded Hugging Face checkpoints.
* **Current Dependencies**: `torch`, `transformers`, `peft`, `pathlib`.
* **APIs Used**:
  - `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)`
  - `PeftModel.from_pretrained(base_model, adapter_dir)`
  - `model.merge_and_unload()`
  - `merged_model.save_pretrained(export_dir, max_shard_size="2GB", safe_serialization=True)`
  - `tokenizer.save_pretrained(export_dir)`
* **Compatibility Status**: **Minor Warning (Deprecation Notice)**
* **Required Changes**:
  - Update `torch_dtype` to `dtype` in `from_pretrained` for Transformers 5.x compatibility.
* **Priority**: Low
* **Risk Level**: Very Low
* **Estimated Effort**: 0.1 Hours

---

## Summary Table

| Notebook | Status | Priority | Risk Level | Est. Effort | Action Items |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `01_environment_setup.ipynb` | Safe | Low | Very Low | 0h | No changes required. |
| `02_dataset_pipeline.ipynb` | Safe | Low | Very Low | 0h | No changes required. |
| `03_qlora_training.ipynb` | **Action Required** | **High** | Medium | 0.5h | FP16/BF16 GradScaler fix + `dtype` kwarg update. |
| `04_evaluation.ipynb` | Safe | Low | Low | 0h | No changes required. |
| `05_export_model.ipynb` | Minor Warning | Low | Very Low | 0.1h | Update `torch_dtype` to `dtype`. |
