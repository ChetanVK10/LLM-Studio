# LLMOps Studio - Phase 4 Documentation Walkthrough

This document outlines the architecture, metrics engines, evaluation pipeline runs, database persistence registries, and verification summaries for **Phase 4 — Evaluation & Benchmarking Framework**.

---

## 1. System Architecture

We have introduced a modular evaluation system located under `backend/evaluation/` coupled to the DB layer:

```
                            Streamlit UI Client
                       (frontend/pages/evaluation.py)
                                    │
                                    ▼ (Service Injected API calls)
                             EvaluationService
                  (backend/services/evaluation_service.py)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   EvaluationPipeline        BenchmarkEngine         SQLiteModelRegistry
 (evaluation/pipeline.py)  (evaluation/benchmark.py) (registry/catalog.py)
          │
      ┌───┴───────────────┐
      ▼                   ▼
 ModelEvaluator     MetricsEvaluator
  (evaluator.py)     (metrics.py)
```

---

## 2. New Modules Overview

### A. Modular Metrics Engine (`backend/evaluation/metrics.py`)
- Employs object-oriented hierarchy extending `BaseMetric`:
  - **Generation metrics**: computes token-based unigram BLEU-1 precision and Longest Common Subsequence (LCS) ROUGE-L overlapping scores. Includes a semantic `BertScoreMetric` mock placeholder.
  - **Classification metrics**: computes macro-averaged Accuracy, Precision, Recall, and F1-Scores.

### B. Inferences & Pipeline (`backend/evaluation/evaluator.py` & `pipeline.py`)
- **`evaluator.py`**: Performs batched inferences under max new tokens and temperature controls, returning generated predictions.
- **`pipeline.py`**: Validates readiness config constraints, loads test splits, executes evaluator loops, runs metrics checks, writes `report.json` outputs, and updates leaderboard results.

### C. Benchmarking Leaderboards (`backend/evaluation/benchmark.py`)
- Groups metrics comparisons by model checkpoints, deduplicating records to display the best overall score per model on selected datasets.

### D. SQLite DB Extensions (`backend/registry/catalog.py`)
- Adds two new tables:
  - `evaluation_results`: Stores runtime configs, metrics summaries, report paths, and execution timestamps.
  - `benchmark_results`: Leaderboard registers cataloging primary metric scores.

### E. Service Layer (`backend/services/evaluation_service.py`)
- Integrates pipeline operations, catalog list lookups, and report markdown generators.

### F. Page UI (`frontend/pages/evaluation.py`)
- Implements 3 tab panels:
  - **Run Evaluation tab**: Form controls to select models, READY datasets, parameters, and trigger evaluation.
  - **Results Grid tab**: Visualizes historical evaluation lists, metric cards, and report down-loaders.
  - **Benchmark Leaderboard tab**: Leaderboard comparative table and `st.bar_chart` metrics comparisons.

---

## 3. Verification Summary

We added 5 new test modules under `tests/` verifying evaluators, custom metrics calculations, pipeline routines, DB catalog inserts, and rankings sorting.

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
collected 43 items

tests/test_benchmark.py .                                                [  2%]
tests/test_cache.py .                                                    [  4%]
tests/test_config.py .....                                               [ 16%]
tests/test_evaluation_pipeline.py ...                                    [ 23%]
tests/test_evaluation_registry.py .                                      [ 25%]
tests/test_evaluator.py .                                                [ 27%]
tests/test_fs.py ...                                                     [ 34%]
tests/test_loader.py ....                                                [ 44%]
tests/test_logger.py .                                                   [ 46%]
tests/test_lora.py .                                                     [ 48%]
tests/test_metrics.py ...                                                [ 55%]
tests/test_model_registry.py .                                           [ 58%]
tests/test_pre_training_validator.py ....                                [ 67%]
tests/test_processor.py ..                                               [ 72%]
tests/test_repository.py .                                               [ 74%]
tests/test_runner.py .                                                   [ 76%]
tests/test_schemas.py ...                                                [ 83%]
tests/test_splitter.py ...                                               [ 90%]
tests/test_tokenizer.py .                                                [ 93%]
tests/test_validator.py ....                                             [100%]

============================ 43 passed in 1.57s =============================
```

All 43 unit tests pass successfully.
