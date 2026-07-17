# LLMOps Studio Folder Directory Structure

This document outlines the folder layout of **LLMOps Studio** and describes the purpose of each directory.

---

## 1. Project Folders Blueprint

- `/backend`: Core Python logic. Contains core parameters, services, schemas validation layer, pipeline engines placeholders, and logging helpers.
- `/frontend`: Streamlit client dashboard pages, styling sheets, and reusable UI components.
- `/configs`: Local settings files, environment templates, and future LLM evaluation prompts.
- `/data`: Persistent operational file storage.
  - `/raw`: Untouched source files (e.g. initial JSON datasets uploads).
  - `/processed`: Tokenized samples and validation splits ready for fine-tuning.
  - `/cache`: HuggingFace tokenization alignments caches.
  - `/uploads`: Temporary buffer for user uploads.
- `/models`: Local cached foundation model weight parameters (e.g. model card configs, HuggingFace downloads).
- `/artifacts`: Generated output artifacts.
  - `/checkpoints`: Exported low-rank adapter weights (safetensors).
  - `/evaluations`: Evaluator scores reports (JSON).
  - `/reports`: Exported markdown/HTML summaries.
  - `/benchmarks`: Hardware VRAM profiles and latency charts logs.
  - `/exports`: Packaged zip checkpoints exports.
- `/logs`: Segregated rotation logs:
  - `application.log`: Central app operations details.
  - `training.log`: Fine-tuning optimization records.
  - `evaluation.log`: Generation score calculations.
  - `benchmark.log`: CUDA device profiling metrics logs.
- `/tests`: Pytest automation testing suite.
- `/docs`: Markdown design and architecture blueprints.
