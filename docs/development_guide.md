# LLMOps Studio Development Guide

Welcome! This guide outlines how to build and expand **LLMOps Studio** components.

---

## 1. Prerequisites & Setup

Ensure you have Python 3.10+ and virtualenv tools installed:

1. **Initialize Virtualenv**:
   ```bash
   make install
   ```
2. **Launch Streamlit Client**:
   ```bash
   make dev
   ```

---

## 2. Coding Guidelines & Standards

- **Type Hints**: All functions must include complete typing annotations (parameters and return values).
- **Domain Constraints**: Avoid hardcoded strings for states or providers. Import and use the centralized Enums in `backend/core/constants.py`.
- **Validation**: Any frontend form submissions or REST API requests must pass validation checks through Pydantic classes inside `backend/schemas/`.

---

## 3. Implementing Pluggable Engines

When you start implementing the actual engines in subsequent development cycles:

### Fine-Tuning QLoRA
1. Open [trainer.py](file:///d:/Projects/LLM-Studio/backend/training/trainer.py).
2. Integrate Hugging Face `peft` and `bitsandbytes` library configurations.
3. Update `QLoRATrainer.execute_training_loop()` to run actual PyTorch trainer steps.
4. Bind this logic inside `TrainingService.trigger_training()` in `backend/services/training_service.py`.

### Scoring Evaluators
1. Open [evaluator.py](file:///d:/Projects/LLM-Studio/backend/evaluation/evaluator.py).
2. Replace mock outputs with evaluation metrics libraries (e.g. `rouge-score`, Hugging Face `evaluate` for BERTScore).
3. Query prompts templates from `/configs/prompts/` to run LLM-as-a-Judge API evaluators.
