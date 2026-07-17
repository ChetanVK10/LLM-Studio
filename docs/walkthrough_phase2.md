# LLMOps Studio - Phase 2 Documentation Walkthrough

This document presents a comprehensive overview of the design, architectural workflows, and testing outcomes for **Phase 2 — Dataset Management Pipeline**.

---

## 1. Architecture Overview

Phase 2 introduces a decoupled, multi-tiered dataset pipeline that isolates data storage, ingestion parsing, statistical profiling, quality validation, database persistence, and user interfaces.

```
                                  Streamlit UI
                           (frontend/pages/datasets.py)
                                        │
                                        ▼ (Service Injected API calls)
                                  DatasetService
                        (backend/services/dataset_service.py)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 DatasetProcessor            HuggingFaceDatasetIngestor
        (backend/datasets/processor.py)    (backend/integrations/huggingface.py)
                         │
        ┌────────────────┼────────────────┬────────────────┐
        ▼                ▼                ▼                ▼
  DatasetLoader   DatasetValidator  DatasetSplitter  SQLiteRepository
   (loader.py)     (validator.py)    (splitter.py)    (repository.py)
        │                                                  │
        └───────────────────────┬──────────────────────────┘
                                ▼
                          LocalStorage
                 (backend/storage/local_storage.py)
```

---

## 2. New Modules & Files

The dataset management engine is fully modular, adhering strictly to SOLID design principles:

### A. Storage Abstraction
- **[storage_manager.py](file:///d:/Projects/LLM-Studio/backend/storage/storage_manager.py)**: Abstract base class defining directory existence checks, byte reads, and write operations.
- **[local_storage.py](file:///d:/Projects/LLM-Studio/backend/storage/local_storage.py)**: Concrete disk filesystem implementation.
- **[__init__.py](file:///d:/Projects/LLM-Studio/backend/storage/__init__.py)**: Storage package initializer exports.

### B. Modular Dataset Engines (`backend/datasets/`)
- **[hashing.py](file:///d:/Projects/LLM-Studio/backend/datasets/hashing.py)**: Computes content checksums using SHA-256 algorithms.
- **[loader.py](file:///d:/Projects/LLM-Studio/backend/datasets/loader.py)**: Memory-efficient CSV, JSON, and JSONL format loaders.
- **[validator.py](file:///d:/Projects/LLM-Studio/backend/datasets/validator.py)**: Audits record columns and identifies empty entries or duplicate rows.
- **[preview.py](file:///d:/Projects/LLM-Studio/backend/datasets/preview.py)**: Compiles `DatasetProfile` properties (statistics, lengths, estimated tokens) and handles first N preview rows.
- **[token_estimator.py](file:///d:/Projects/LLM-Studio/backend/datasets/token_estimator.py)**: Pluggable base class token count estimator (heuristic base: 1 word ≈ 1.3 tokens).
- **[splitter.py](file:///d:/Projects/LLM-Studio/backend/datasets/splitter.py)**: Partitions datasets reproducibly into train/val/test splits using random seed controls.
- **[cache.py](file:///d:/Projects/LLM-Studio/backend/datasets/cache.py)**: Cache lookup manager to avoid repeated preprocessing.
- **[repository.py](file:///d:/Projects/LLM-Studio/backend/datasets/repository.py)**: SQLite database catalog provider managing dataset metadata registry.
- **[processor.py](file:///d:/Projects/LLM-Studio/backend/datasets/processor.py)**: Orchestrator coordinating all dataset submodules.

### C. Services & Integrations
- **[huggingface.py](file:///d:/Projects/LLM-Studio/backend/integrations/huggingface.py)**: Appended `HuggingFaceDatasetIngestor` mocking lightweight Hugging Face Hub pulls.
- **[dataset_service.py](file:///d:/Projects/LLM-Studio/backend/services/dataset_service.py)**: Core service routing frontend requests to the pipeline engine.
- **[dependencies.py](file:///d:/Projects/LLM-Studio/backend/core/dependencies.py)**: Injects lazy-loaded singletons for SQLite, LocalStorage, and processors.

### D. Presentation Layer
- **[datasets.py](file:///d:/Projects/LLM-Studio/frontend/pages/datasets.py)**: Interactive Streamlit interface showing uploads form, profile cards, validation warning lists, previews table, and split sliders.

---

## 3. Core Processing Pipeline Workflows

### A. Ingestion Workflow
When a file is ingested (either locally uploaded or via HF import):
1. **Hash Generation**: A SHA-256 hash is computed. If it matches a `READY` registry entry, processing is skipped (cache hit).
2. **Raw Storage**: Raw bytes write to `data/raw/{hash}.{ext}` via the storage layer.
3. **Pydantic Registry Insertion**: Registers base metadata (state: `UPLOADED`).

### B. Validation & Profiling Workflow
1. **Loading**: The loader parses files into row record dictionaries.
2. **Validation**: Checks for required columns (`instruction`, `output`), null cells, and duplicates. Generates a validation report with warnings or errors. If validation fails (any `ERROR` exists), the status remains `VALIDATED` and ingestion terminates.
3. **Profiling**: Extracts prompt/response character length means, column headers, and token estimations.

### C. Partitioning & Caching Workflow
1. **Splitting**: Randomizes dataset row lists based on random seeds and splits ratios (Train/Val/Test).
2. **Processed Storage**: Writes partition splits directly to `data/processed/{hash}/[train|validation|test].jsonl` using the storage manager.
3. **Finalize**: Updates registry database entry setting status to `READY`.

---

## 4. Dataset Lifecycle States

Ingestion is represented by consecutive transitions of the `DatasetLifecycleState` Enum:
1. `UPLOADED`: Raw files written to disk.
2. `VALIDATED`: Quality audits completed successfully.
3. `PROFILED`: Statistical metrics compiled.
4. `SPLIT`: Dataset segmented into splits partitions.
5. `REGISTERED`: Metadata serialized to the catalog database.
6. `READY`: Dataset is fully processed and ready for training.

---

## 5. Verification Summary

We created 6 new test modules targeting all dataset engine segments.

### Executed command:
```bash
.\.venv\Scripts\pytest
```

### Test Logs output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Projects\LLM-Studio
configfile: pyproject.toml
collected 26 items

tests/test_cache.py .                                                    [  3%]
tests/test_config.py .....                                               [ 23%]
tests/test_fs.py ...                                                     [ 34%]
tests/test_loader.py ....                                                [ 50%]
tests/test_logger.py .                                                   [ 53%]
tests/test_processor.py ..                                               [ 61%]
tests/test_repository.py .                                               [ 65%]
tests/test_schemas.py ..                                                 [ 73%]
tests/test_splitter.py ...                                               [ 84%]
tests/test_validator.py ....                                             [100%]

============================ 26 passed in 0.40s =============================
```

All 26 unit tests execute successfully, confirming modularity, validation logic correctness, database safety, and caching reproducibility.
