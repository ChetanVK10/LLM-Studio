# 🚀 LLM-Studio Training Module

This module implements a production-grade **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning, evaluation, and registry pipeline for the **Qwen2.5-3B-Instruct** model, using parameter-efficient fine-tuning (PEFT) and Hugging Face libraries.

## 📂 Module Structure

```
training/
├── notebooks/
│   ├── 01_environment_setup.ipynb   # Colab pip installs, VRAM/hardware tests
│   ├── 02_dataset_pipeline.ipynb     # Ingest CSV, validate schemas, format ChatML
│   ├── 03_qlora_training.ipynb       # 4-bit SFT fine-tuning and experiment logging
│   ├── 04_evaluation.ipynb           # ROUGE & EM metrics scoring, loss/metric plotting
│   └── 05_export_model.ipynb         # PEFT adapter registrations & optional merging
├── scripts/
│   ├── train.py                     # CLI fine-tuning script
│   ├── evaluate.py                  # CLI inference and scoring script
│   ├── export_adapter.py            # CLI adapter catalog registration script
│   └── merge_adapter.py             # CLI weight-merging utility script
├── configs/
│   └── qlora_config.yaml            # Parameter settings for training & evaluation
├── requirements.txt                 # Module python package dependencies
└── README.md                        # Documentation (this file)
```

---

## 🛠️ Config-Driven Architecture

The training pipelines (both scripts and notebooks) are driven by [configs/qlora_config.yaml](file:///d:/Projects/LLM-Studio/training/configs/qlora_config.yaml). This permits training alternate models, configuring column mapping, and editing hyperparameter structures without modifications to Python scripts.

Key YAML configurations:
- **`model`**: Base model name, quantization type (`4bit`, `8bit`, `none`), and device maps.
- **`dataset`**: Paths to input CSV files, train/validation targets, split ratios, and column mapping directories (`instruction`, `input`, `output`).
- **`lora`**: Low-rank configurations (rank `r`, scaling `lora_alpha`, dropout, and layer target lists).
- **`training`**: SFT parameters (epochs, learning rate, batch size, gradient accumulation, sequence length, and logs directory).
- **`evaluation`**: Max token generations, temperatures, and paths for metrics reports and charts.
- **`export`**: Target adapter registry names and registry database path pointers.

---

## ⚡ Quick Start: Python Scripts

Ensure dependencies are installed:
```bash
pip install -r training/requirements.txt
```

### 1. Execute Dataset Preprocessing and Fine-Tuning
The training script dynamically reads configs, compiles hardware spec metrics inside `gpu_info.json`, formats inputs into ChatML, and launches the training loop:
```bash
python training/scripts/train.py
```
*Outputs are saved under a timestamped run subdirectory: `artifacts/experiments/experiment_<timestamp>/`.*

### 2. Run Evaluation
Runs batch generation on the validation dataset split, computes ROUGE scores, writes predictions side-by-side to a CSV, and plots metrics (loss curves and ROUGE distributions):
```bash
python training/scripts/evaluate.py
```
*Creates `loss_curve.png`, `rouge_scores.png`, `metrics.json`, and `sample_predictions.csv` inside evaluation output folders, and syncs copies back to the experiment folder.*

### 3. Register PEFT Adapter to the Catalog
Copies the best weights checkpoints to the model registry catalog folder and tracks parameters inside the registry ledger:
```bash
python training/scripts/export_adapter.py --set_production
```
*Adapter folder is written to `models/adapters/<name>` and cataloged in `models/registry.json`.*

### 4. (Optional) Merge Adapter with Base Model
If standalone production weights are needed for serving (e.g. without PEFT wrappers), compile them:
```bash
python training/scripts/merge_adapter.py --set_production
```
*Merged model weights and configs are written to `models/merged/<name>-merged`.*

---

## 🌐 Jupyter Notebooks (Google Colab Compatible)

For cloud execution:
1. Open the [notebooks/](file:///d:/Projects/LLM-Studio/training/notebooks/) folder.
2. Upload the notebooks to Google Colab and set the runtime to **T4 GPU** (or any higher GPU compute instance).
3. Execute notebooks sequentially `01` through `05`. 
4. The notebooks handle pip dependency initialization and contain robust dry-run options to allow code cells to run locally without crashing.

---

## 📈 Experiment Tracking & Model Registry

### Experiment Structure
Every execution cycle produces standard run tracking assets:
```
artifacts/experiments/experiment_2026_07_18_110000/
├── config.yaml          # Config snapshot used
├── gpu_info.json        # GPU specs & VRAM info
├── metrics.json         # ROUGE & loss metrics
├── train.log            # Training execution output logs
├── sample_predictions.csv
├── plots/
│   ├── loss_curve.png
│   └── rouge_scores.png
└── adapter/             # Final trained adapter weights
```

### Registry Structure
The project-level `models/registry.json` maps registered models. E.g.:
```json
{
  "qwen-support-v1": {
    "type": "adapter",
    "path": "models/adapters/qwen-support-v1",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "hyperparameters": { "r": 16, "lora_alpha": 32, "epochs": 3 },
    "created_at": "2026-07-18T05:32:00Z",
    "is_production": true
  }
}
```

---

## 🛠️ Loading in Backend (PEFT Compatibility)

The backend can query `models/registry.json` to load the active production adapter dynamically:
```python
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Parse production configuration
with open("models/registry.json", "r") as f:
    registry = json.load(f)

production_cfg = next((v for k, v in registry.items() if v.get("is_production", False)), None)
if not production_cfg:
    raise RuntimeError("No active production model configured.")

model_path = Path(production_cfg["path"])

# 2. Dynamic loading based on model registry type
if production_cfg["type"] == "adapter":
    # Load base model + overlay adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        production_cfg["base_model"], 
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path)
else:
    # Load merged model directly
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
```
