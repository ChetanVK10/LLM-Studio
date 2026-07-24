# API Migration & Compatibility Strategy — LLMOps Studio

This document outlines the engineering strategy for handling API transitions, deprecations, and mixed-precision execution across library version upgrades.

---

## 1. Migration Principles

1. **Non-Destructive Execution**: Project source files and notebooks remain unaltered until compatibility patches are fully validated in isolated test environments.
2. **Defensive Parameter Solver**: Precision settings, device allocations, and library version constraints are dynamically resolved at runtime using central utilities ([utils.py](file:///d:/Projects/LLM-Studio/training/scripts/utils.py)).
3. **Graceful Fallback Handling**: Backward-compatible keyword arguments (e.g. `processing_class` vs `tokenizer`, `eval_strategy` vs `evaluation_strategy`) are inspected and mapped dynamically.

---

## 2. API Transition Patterns

### Pattern 1: `SFTTrainer` Tokenizer Kwarg Compatibility
* **Old Pattern** (TRL < 0.11.0):
  ```python
  trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)
  ```
* **Modern Pattern** (TRL 0.11.0+ / 1.8.0+):
  ```python
  trainer = SFTTrainer(model=model, processing_class=tokenizer, ...)
  ```
* **Migration Implementation** (in `utils.py`):
  ```python
  def get_sft_trainer_kwargs(tokenizer):
      import inspect
      from trl import SFTTrainer
      sig = inspect.signature(SFTTrainer.__init__)
      if "processing_class" in sig.parameters:
          return {"processing_class": tokenizer}
      return {"tokenizer": tokenizer}
  ```

---

### Pattern 2: Model Loading `dtype` Kwarg Migration
* **Old Pattern** (Transformers 4.x):
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype=compute_dtype
  )
  ```
* **Modern Pattern** (Transformers 5.x):
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      base_model_name,
      dtype=compute_dtype
  )
  ```

---

### Pattern 3: Training Arguments Evaluation Strategy
* **Old Pattern** (Transformers < 4.41):
  ```python
  args = TrainingArguments(output_dir="./tmp", evaluation_strategy="epoch")
  ```
* **Modern Pattern** (Transformers 4.41+ / 5.x):
  ```python
  args = TrainingArguments(output_dir="./tmp", eval_strategy="epoch")
  ```

---

### Pattern 4: QLoRA Mixed-Precision Adapter Parameter Guard
* **Problem**: TRL 1.8.0/1.9.0 `SFTTrainer` converts trainable adapter parameters to `torch.bfloat16` during FP16 training on Turing GPUs (Tesla T4).
* **Migration Fix**:
  ```python
  trainer = SFTTrainer(model=model, args=training_args, ...)

  # Explicitly preserve FP32 adapter parameters for FP16 training
  if training_args.fp16 and not training_args.bf16:
      for param in trainer.model.parameters():
          if param.requires_grad:
              param.data = param.data.to(torch.float32)
  ```
