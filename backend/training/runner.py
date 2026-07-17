"""Training Job Worker Runner.

Manages fine-tuning execution cycles using worker threads, ensuring single
concurrencies using locks, validating inputs, and writing checkpoint outputs.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.exceptions import TrainingError
from backend.schemas.registry import RegistryEntry
from backend.schemas.training import TrainingJobConfig, TrainingJobState, TrainingJobStatus
from backend.storage.storage_manager import StorageManager
from backend.training.validator import PreTrainingValidator

logger = logging.getLogger("app")

# Thread lock to guarantee a single concurrent training job runs locally
_lock = threading.Lock()
_active_jobs: Dict[str, TrainingJobStatus] = {}


class TrainingRunner:
    """Spawns training runs in thread blocks and manages concurrent locks."""

    def __init__(
        self,
        storage_manager: StorageManager,
        dataset_service: Any,
        model_registry: Any  # SQLiteModelRegistry
    ) -> None:
        """Initializes the runner.

        Args:
            storage_manager: Read/write disk operations manager.
            dataset_service: Registry client to check dataset states.
            model_registry: Registry catalog provider.
        """
        self.storage_manager = storage_manager
        self.dataset_service = dataset_service
        self.model_registry = model_registry

    def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        """Queries training job status maps in memory or database catalog registries.

        Args:
            job_id: Unique training job run identifier.

        Returns:
            TrainingJobStatus object if tracked, None otherwise.
        """
        # 1. Resolve from active memory states
        if job_id in _active_jobs:
            status = _active_jobs[job_id]
            
            # Read progress metrics updates from local run_stats file
            stats_path = f"artifacts/checkpoints/{job_id}/run_stats.json"
            if self.storage_manager.exists(stats_path):
                try:
                    raw_data = self.storage_manager.read_file(stats_path)
                    stats = json.loads(raw_data.decode("utf-8"))
                    status.current_step = stats.get("current_step", status.current_step)
                    status.total_steps = stats.get("total_steps", status.total_steps)
                    status.current_epoch = stats.get("current_epoch", status.current_epoch)
                    status.train_loss = stats.get("train_loss", status.train_loss)
                    status.eval_loss = stats.get("eval_loss", status.eval_loss)
                except Exception:
                    pass
            return status

        # 2. Revert to persistent model DB records
        return self.model_registry.get_job_status(job_id)

    def start_training_job(self, config: TrainingJobConfig) -> str:
        """Runs the validation audits and triggers thread training processes.

        Args:
            config: Job parameters configuration settings.

        Returns:
            Unique job ID string.

        Raises:
            TrainingError: If locks are occupied or validation checks fail.
        """
        # 1. Run Pre-Training Validation Stage
        PreTrainingValidator.validate_readiness(
            config=config,
            dataset_service=self.dataset_service,
            storage_manager=self.storage_manager
        )

        # 2. Acquire lock to restrict training to one thread
        acquired = _lock.acquire(blocking=False)
        if not acquired:
            raise TrainingError(
                "Training runner is occupied: Another training job is currently running."
            )

        # 3. Initialize memory mappings
        status = TrainingJobStatus(
            job_id=config.job_id,
            state=TrainingJobState.RUNNING,
            checkpoint_path=f"artifacts/checkpoints/{config.job_id}"
        )
        _active_jobs[config.job_id] = status
        self.model_registry.save_job_status(status)

        # 4. Spawns daemon worker thread
        thread = threading.Thread(
            target=self._execute_training_loop,
            args=(config, status),
            daemon=True
        )
        thread.start()

        return config.job_id

    def _execute_training_loop(self, config: TrainingJobConfig, status: TrainingJobStatus) -> None:
        """Thread worker routine simulating/running HF training steps."""
        checkpoint_dir = f"artifacts/checkpoints/{config.job_id}"
        
        try:
            logger.info(f"TrainingRunner: Beginning job execution thread for '{config.job_id}'")
            self.storage_manager.ensure_dir(checkpoint_dir)

            # Simulated steps iteration representing training passes
            total_steps = config.hyperparameters.epochs * 5
            status.total_steps = total_steps
            start_time = time.time()

            for step in range(1, total_steps + 1):
                time.sleep(0.2)  # Simulate GPU execution time
                
                # Mock declining loss metrics curves
                train_loss = max(0.08, 1.85 - (step * 0.12) + (step % 2) * 0.04)
                current_epoch = float(step) / 5.0
                elapsed = time.time() - start_time
                status.elapsed_seconds = elapsed

                stats_update = {
                    "current_step": step,
                    "total_steps": total_steps,
                    "current_epoch": round(current_epoch, 2),
                    "train_loss": round(train_loss, 4),
                    "eval_loss": round(train_loss * 1.05, 4) if step % 5 == 0 else None,
                    "learning_rate": config.hyperparameters.learning_rate * (1.0 - (step / total_steps))
                }

                # Save steps metrics JSON to file
                self.storage_manager.write_file(
                    f"{checkpoint_dir}/run_stats.json",
                    json.dumps(stats_update, indent=2)
                )

                # Write console outputs to rotating training log
                log_line = (
                    f"Epoch {stats_update['current_epoch']:.2f} | "
                    f"Step {step}/{total_steps} | "
                    f"Loss: {stats_update['train_loss']:.4f}\n"
                )
                self._append_training_log(log_line)

            # Write mock weights checkpoints files to represent output exports
            self.storage_manager.write_file(
                f"{checkpoint_dir}/adapter_config.json",
                '{"peft_type": "LORA", "r": 8, "lora_alpha": 16}'
            )
            self.storage_manager.write_file(
                f"{checkpoint_dir}/adapter_model.safetensors",
                b"adapter_model_weights_safetensors_bytes"
            )

            # Finalize status updates
            status.state = TrainingJobState.COMPLETED
            status.train_loss = 0.08
            status.elapsed_seconds = time.time() - start_time
            self.model_registry.save_job_status(status)

            # Register trained model checkpoint in catalog
            entry = RegistryEntry(
                checkpoint_id=config.job_id,
                model_name=f"adapter-checkpoint-{config.job_id[:8]}",
                base_model=config.base_model_name,
                path=checkpoint_dir,
                tags=["fine-tuned", "lora"],
                hyperparameters={
                    "epochs": config.hyperparameters.epochs,
                    "batch_size": config.hyperparameters.batch_size,
                    "learning_rate": config.hyperparameters.learning_rate,
                    "lora_r": config.lora_config.r
                }
            )
            self.model_registry.save_checkpoint(entry)

        except Exception as err:
            logger.error(f"TrainingRunner: Job execution thread failed: {err}")
            status.state = TrainingJobState.FAILED
            status.error_message = str(err)
            self.model_registry.save_job_status(status)
            
        finally:
            # Re-pop memory indexes and release concurrent locks
            _active_jobs.pop(config.job_id, None)
            _lock.release()
            logger.info("TrainingRunner: Concurrency lock released.")

    def _append_training_log(self, new_log: str) -> None:
        log_file = "logs/training.log"
        existing_log = ""
        
        if self.storage_manager.exists(log_file):
            try:
                existing_log = self.storage_manager.read_file(log_file).decode("utf-8")
            except Exception:
                pass
                
        self.storage_manager.write_file(log_file, existing_log + new_log)
