#!/usr/bin/env python3
"""LLM-Studio Adapter Export and Registration Script.

Copies training checkpoints containing LoRA weight parameters to the central models directory,
and updates/initializes the models/registry.json tracking directory.
"""

import os
import sys
import yaml
import json
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("exporter")

# PROJECT_ROOT and RUNTIME_ROOT configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RUNTIME_ROOT = PROJECT_ROOT

# Fallback/override to Colab paths if running in Colab
if 'google.colab' in sys.modules:
    PROJECT_ROOT = Path("/content/LLM-Studio")
    RUNTIME_ROOT = Path("/content/drive/MyDrive/LLM-Studio")

# Derive subdirectories
CONFIG_DIR = PROJECT_ROOT / "training" / "configs"
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = RUNTIME_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = RUNTIME_ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "adapters"
MERGED_MODEL_DIR = MODELS_DIR / "merged"
ARTIFACTS_DIR = RUNTIME_ROOT / "artifacts"

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_latest_checkpoint(checkpoints_dir: Path) -> Path:
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoints directory does not exist: {checkpoints_dir}")
        sys.exit(1)
    runs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        logger.error(f"No fine-tuned runs found in checkpoints directory: {checkpoints_dir}")
        sys.exit(1)
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Export LoRA adapter to the model catalog.")
    parser.add_argument("--config", type=str, default="", help="Path to config file.")
    parser.add_argument("--name", type=str, default="", help="Registry adapter folder name. Overrides config.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path to export. Overrides scanning latest.")
    parser.add_argument("--set_production", action="store_true", help="Marks this exported adapter as the active production model.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Resolve configuration path
    config_path = Path(args.config) if args.config else CONFIG_DIR / "qlora_config.yaml"
    config = load_config(config_path)
    export_cfg = config["export"]
    
    # Resolve roots from RUNTIME_ROOT
    checkpoints_root = RUNTIME_ROOT / config["training"]["output_dir"]
    adapters_root = RUNTIME_ROOT / export_cfg["adapters_dir"]
    registry_path = RUNTIME_ROOT / export_cfg["registry_path"]
    
    # 2. Resolve source checkpoint
    src_checkpoint = args.checkpoint
    if not src_checkpoint:
        src_checkpoint = str(get_latest_checkpoint(checkpoints_root))
    src_path = Path(src_checkpoint)
    
    if not src_path.exists():
        logger.error(f"Source checkpoint path does not exist: {src_path}")
        sys.exit(1)
        
    # Check if this contains adapter config
    cfg_file = src_path / "adapter_config.json"
    if not cfg_file.exists():
        logger.warning(f"Warning: {cfg_file} not found. Running mock registration checks.")
        
    # 3. Resolve destination path
    adapter_name = args.name if args.name else export_cfg.get("adapter_name", "qwen-support-v1")
    dest_dir = adapters_root / adapter_name
    
    logger.info(f"Exporting adapter from '{src_path}' to '{dest_dir}'...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copied_files = []
    for f in src_path.iterdir():
        if f.is_file():
            shutil.copy2(f, dest_dir / f.name)
            copied_files.append(f.name)
            
    logger.info(f"Copied {len(copied_files)} files: {copied_files}")
    
    # 4. Update registry database
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    registry = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r") as fh:
                registry = json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Existing registry file was malformed. Re-initializing.")
            
    # Set all other adapters to not production if we mark this one as production
    if args.set_production:
        for k, v in registry.items():
            if v.get("type") == "adapter":
                v["is_production"] = False
                
    # Compile hyperparameters description
    hyperparams = {
        "r": config["lora"]["r"],
        "lora_alpha": config["lora"]["lora_alpha"],
        "epochs": config["training"]["num_train_epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["per_device_train_batch_size"]
    }
    
    # Update entry
    registry[adapter_name] = {
        "type": "adapter",
        "path": f"models/adapters/{adapter_name}",
        "base_model": config["model"]["base_model_name"],
        "hyperparameters": hyperparams,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "is_production": args.set_production or (len(registry) == 0) # True by default if it's the first model
    }
    
    with open(registry_path, "w") as fh:
        json.dump(registry, fh, indent=2)
        
    logger.info(f"Registry updated at '{registry_path}'. Active production: {registry[adapter_name]['is_production']}")
    logger.info("Export adapter process completed successfully.")

if __name__ == "__main__":
    main()
