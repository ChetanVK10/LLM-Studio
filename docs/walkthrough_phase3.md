# LLMOps Studio - Phase 3 Documentation Walkthrough

This document outlines the architecture, pipeline workflow, model registry persistence, monitoring, and verification results for **Phase 3 — Fine-Tuning Pipeline**.

---

## 1. System Architecture & Ingestion Pipeline

We have introduced a production-grade fine-tuning engine under `backend/training/` integrated with service layers and SQLite catalogs:

```
                            Streamlit UI Client
                       (frontend/pages/training.py)
                                    │
                                    ▼ (Service Injected API calls)
                             TrainingService
                  (backend/services/training_service.py)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
    TrainingRunner           SQLiteModelRegistry       TrainingMonitor
 (training/runner.py)      (registry/catalog.py)     (training/monitor.py)
          │
          ├─────────────────────────┬─────────────────────────┐
          ▼                         ▼                         ▼
   HF Trainer Hook            PEFT Adapter         Pre-Training Validator
 (training/pipeline.py)     (training/lora.py)     (training/validator.py)
```

---

## 2. Walkthrough of New Modules

### A. Pre-Training Validation Stage (`backend/training/validator.py`)
Executes synchronous checks prior to training thread dispatch:
- **Dataset Readiness**: Asserts dataset exists and is in `READY` status.
- **Model Availability**: Audits foundation model string structure (checks HF repository format or local path existence).
- **Paths Writability**: Confirms that checkpoint directories can be created using the storage manager.
- **Hyperparameter Boundaries**: Checks optimization limits (epochs $\ge 1$, batch size $\ge 1$, learning rate $> 0$, warmup ratio $\in [0, 1]$, and weight decay $\ge 0$).

### B. Supervised Fine-Tuning Tokenization (`backend/training/tokenizer.py`)
- Formats instruction templates: `<s>Instruction: {instruction}\nInput: {input}\nResponse: {output}</s>`.
- Sets label IDs corresponding to prompt tokens to `-100` to mask them. Optimizes strictly on output response generation loss.

### C. PEFT LoRA Configs (`backend/training/lora.py`)
- Bridges schema controls with Hugging Face `peft` `LoraConfig`, targeting custom matrices lists (e.g. `["q_proj", "v_proj"]`).

### D. Asynchronous Training Runner (`backend/training/runner.py` & `pipeline.py`)
- Employs daemon worker threads running training tasks asynchronously to ensure the Streamlit client remains responsive.
- Utilizes `threading.Lock` to guarantee only **one** local training job runs concurrently.
- Updates real-time step statistics to `run_stats.json` file records using standard Trainer callback hooks (`ProgressMonitorCallback`).
- Implements simulated loops when execution runs lack PyTorch/GPU, rendering loss steps and console logging feeds.

### E. Model Registry & Histories Database (`backend/registry/catalog.py`)
- Uses SQLite catalogs to persist checkpoint metadata registrations (`model_registry` table) and runs logs (`training_jobs` table).

---

## 3. UI Features (`frontend/pages/training.py`)

Redesigned the screen to include three tab components:
1. **⚙️ Configure Training**: Selects READY datasets, base models, LoRA rank details, optimization parameters, and dispatches jobs.
2. **📈 Active Job Monitor**: Auto-refreshing progress bars, metric cards (epoch, step, losses), live line charts plotting training curves, and scrolling console logs.
3. **📜 Training History**: Visual grids indexing all training histories and checkpoints from registry tables.

---

## 4. Verification Summary

We added 5 new test modules under `tests/` verifying all model registration, validation, SFT tokenizer padding, PEFT configurations, and thread runner lock features.

### Command Executed:
```powershell
.\.venv\Scripts\pytest
```

### Verification Logs:
```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Projects\LLM-Studio
configfile: pyproject.toml
collected 34 items

tests/test_cache.py .                                                    [  2%]
tests/test_config.py .....                                               [ 17%]
tests/test_fs.py ...                                                     [ 26%]
tests/test_loader.py ....                                                [ 38%]
tests/test_logger.py .                                                   [ 41%]
tests/test_lora.py .                                                     [ 44%]
tests/test_model_registry.py .                                           [ 47%]
tests/test_pre_training_validator.py ....                                [ 58%]
tests/test_processor.py ..                                               [ 64%]
tests/test_repository.py .                                               [ 67%]
tests/test_runner.py .                                                   [ 70%]
tests/test_schemas.py ..                                                 [ 76%]
tests/test_splitter.py ...                                               [ 85%]
tests/test_tokenizer.py .                                                [ 88%]
tests/test_validator.py ....                                             [100%]

============================ 34 passed in 1.46s =============================
```

All 34 tests compile, run, and pass successfully.
