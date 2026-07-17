# 🛠️ LLMOps Studio

<p align="center">
  <b>A Production-Ready Orchestration Platform for LLM Fine-Tuning, Benchmarking, and Evaluation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.59.2-FF4B4B.svg" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/Pydantic-2.13-E92063.svg" alt="Pydantic Version">
  <img src="https://img.shields.io/badge/Docker-Enabled-0db7ed.svg" alt="Docker Enabled">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 📌 Project Overview

**LLMOps Studio** is a platform for large language model (LLM) lifecycle management. It provides software engineers and researchers with a unified dashboard to ingest custom datasets, orchestrate parameter-efficient QLoRA fine-tuning, execute metrics-driven evaluations, compare hardware benchmarks, and catalog model adapter checkpoints.

> [!IMPORTANT]
> **Project Implementation Status (Phase 1: Architectural Foundation)**  
> This repository represents the **Phase 1 deliverable**, focusing strictly on software architecture and clean separation of concerns. All model training loops, evaluation logic (ROUGE/BERTScore), benchmarking routines, and third-party integrations (Hugging Face, Weights & Biases) are implemented as structured placeholder interfaces. There are no active machine learning weights loaded or executed in this phase.

---

## 🚀 Key Features (Phase 1 Mockups)

- **Datasets Panel**: Ingest raw JSON/Parquet datasets, track features schema, and review validation splits.
- **Fine-Tuning Portal**: Bind parameters for QLoRA rank ($r$), scaling ($\alpha$), learning rates, and target layers.
- **Evaluation Matrix**: Compare generation quality (ROUGE-L, BERTScore, Exact Match), average response latency, throughput speeds, and estimated dollar cost metrics.
- **Benchmarking Suite**: Profile throughput rates, peak memory footprint (VRAM), and relative GPU running costs.
- **Registry Catalog**: Promoting versioned adapters and checklists across environment pipelines.

---

## 📐 System Architecture

LLMOps Studio implements a highly-decoupled, multi-tiered architecture:

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit Frontend UI                  │
│       (frontend/pages/ & frontend/components/ui.py)      │
└────────────────────────────┬─────────────────────────────┘
                             │ (Dependency Injection via dependencies.py)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                Orchestration Service Layer               │
│                  (backend/services/*)                    │
└──────┬─────────────────────┬──────────────────────┬──────┘
       │                     │                      │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Validation  │      │  Domain Core │      │ Pluggable    │
│   Schemas    │      │ constants.py │      │ Physical     │
│  (Pydantic)  │      │  version.py  │      │ Execution    │
│ (schemas/*)  │      │  exceptions  │      │ Skeletons    │
└──────────────┘      └──────────────┘      └──────────────┘
```

### Rationale Behind the Structure
1. **Service Layer**: Decouples UI screens from business rules. If we migrate from Streamlit to a React/Next.js client in subsequent phases, the core backend logic remains untouched.
2. **Schemas Layer**: Restricts API calls to validated Pydantic models, converting incoming configurations and blocking malformed parameter definitions early.
3. **Registry Modules**: Separates catalog indices, physical adapter weight resolving, and metadata logs mapping to enable scalability.

---

## 📂 Folder Structure

```
LLM-Studio/
├── backend/
│   ├── api/                   # Future FastAPI routing blueprints
│   ├── core/                  # Configuration settings, dependencies, versions
│   ├── services/              # Orchestrators linking frontend to engines
│   ├── schemas/               # Validated Pydantic exchange data models
│   ├── experiment_tracking/   # Runs and metrics series logs managers
│   ├── integrations/          # Adapters (HuggingFace Hub, W&B, Colab)
│   ├── training/              # QLoRA fine-tuning execution skeletons
│   ├── evaluation/            # ROUGE, BERTScore, Judge validators
│   ├── benchmarking/          # Throughput and Peak VRAM profilers
│   ├── datasets/              # Data parsing and split preprocessors
│   └── utils/                 # Logging configs, startup path verifiers, health utilities
├── frontend/
│   ├── app.py                 # Streamlit client entrypoint
│   ├── pages/                 # Section panels (Dashboard, Training, etc.)
│   ├── components/            # Reusable UI widgets library
│   └── styles/                # Global style overrides CSS sheets
├── configs/
│   ├── prompts/               # Target prompt templates directory
│   └── config.yaml            # Default YAML parameters configurations
├── data/                      # Ingested datasets (raw, processed, cache, uploads)
├── artifacts/                 # Generated exports (checkpoints, evaluations, reports)
├── logs/                      # Rotating application, training, and hardware logs
└── tests/                     # Automated pytest testing suite
```

---

## 📦 Tech Stack

- **Backend**: Python 3.10+
- **Frontend Client**: Streamlit
- **Configuration & Validation**: Pydantic v2 & Pydantic Settings
- **Log Management**: Standard Logging (Rotating File Handlers)
- **Tooling**: Ruff (Linting), Black (Formatting), Pytest (Testing)
- **Infrastructure**: Docker & Docker Compose

---

## 🛠️ Installation Guide

### Local Environment Setup
Ensure you have Python 3.10 or higher installed:

1. **Clone the Repository & Install Virtualenv**:
   ```bash
   make install
   ```
2. **Launch Streamlit Dashboard**:
   ```bash
   make dev
   ```
3. **Execute Test Suite**:
   ```bash
   make test
   ```

### Running with Docker
Build and launch the container workspace:
```bash
make docker-up
```
Once initialized, access the dashboard at `http://localhost:8501`.

---

## 💡 Usage Guide

1. **Dashboard**: Inspect system startup checks success logs and directory mappings configurations.
2. **Datasets**: Select a registered data profile (such as `alpaca-cleaned-1k`) and check split columns metadata.
3. **Fine-Tuning**: Adjust QLoRA hyperparameter controls (learning rate, rank, alpha) to prepare request schemas payloads.
4. **Settings**: View cached settings variables loaded from `.env` files and merged with `configs/config.yaml`.

---

## 🗺️ Roadmap & Future Phases

- **Phase 2 (Data Pipelines)**: Implement real file parsing and cached tokenization splits.
- **Phase 3 (GPU Training)**: Integrate `peft` and `bitsandbytes` to execute 4-bit QLoRA fine-tuning loops.
- **Phase 4 (Evaluations Suite)**: Replace placeholders scoring with active libraries and LLM Judge evaluations.
- **Phase 5 (APIs & Scale)**: Implement API wrappers using FastAPI and scale jobs running using Celery.

---

## 🖼️ Screenshots (Placeholder)

*Screenshots displaying training losses logs and benchmarks comparisons chart lists will be updated here during the Phase 3 releases.*

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
