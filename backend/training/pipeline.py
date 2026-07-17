"""Training Pipeline & Callbacks.

Defines custom training arguments hooks and progress callbacks matching Hugging Face
TrainerCallback designs, logging optimization metrics to local files for real-time frontend charts.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("app")

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
    class TrainerCallback:
        """Mock TrainerCallback."""
        pass
        
    class TrainingArguments:
        """Mock TrainingArguments."""
        pass
        
    class TrainerState:
        """Mock TrainerState."""
        pass
        
    class TrainerControl:
        """Mock TrainerControl."""
        pass


class ProgressMonitorCallback(TrainerCallback):
    """Callback mapping Trainer logs to local status files."""

    def __init__(self, stats_path: str) -> None:
        """Initializes callback.

        Args:
            stats_path: Target filepath destination to write progress.
        """
        self.stats_path = stats_path

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Invoked by the trainer on logging steps. Writes statistics to stats_path.

        Args:
            args: Active training arguments.
            state: Current state parameters of trainer.
            control: Flow controller object.
            logs: Logged metrics dictionary.
            kwargs: Variadic kwargs.
        """
        if not logs:
            return

        try:
            # Extract step progress metrics
            stats = {
                "current_step": state.global_step,
                "total_steps": getattr(state, "max_steps", 0) or args.max_steps,
                "current_epoch": round(state.epoch or 0.0, 2),
                "train_loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "learning_rate": logs.get("learning_rate"),
            }
            
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.error(f"Callback Monitor Error: Failed to write step stats: {e}")
