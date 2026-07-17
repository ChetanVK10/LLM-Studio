"""Registry Orchestration Service.

Binds training outputs to checkpoint listings, metadata records, and path resolving catalogs.
"""

from datetime import datetime
import logging
from typing import List

from backend.core.exceptions import CheckpointNotFoundError
from backend.schemas.registry import RegistryEntry

logger = logging.getLogger("app")


class RegistryService:
    """Coordinates checkpoint catalog operations and tags updates."""

    def __init__(self) -> None:
        """Initializes the registry service instance."""
        pass

    def register_checkpoint(self, entry: RegistryEntry) -> RegistryEntry:
        """Registers a checkpoint entry in the catalog index.

        Args:
            entry: Checkpoint metadata configuration.

        Returns:
            The registered RegistryEntry.
        """
        logger.info(
            f"Cataloging model checkpoint '{entry.checkpoint_id}' "
            f"for model '{entry.model_name}'"
        )
        return entry

    def get_checkpoint(self, checkpoint_id: str) -> RegistryEntry:
        """Looks up a model checkpoint profile by identifier.

        Args:
            checkpoint_id: Query ID.

        Returns:
            RegistryEntry for the requested checkpoint.

        Raises:
            CheckpointNotFoundError: If no entry matches the query ID.
        """
        logger.info(f"Retrieving catalog entry for checkpoint: '{checkpoint_id}'")
        if checkpoint_id == "qlora_llama_001":
            return RegistryEntry(
                checkpoint_id="qlora_llama_001",
                model_name="LLaMA-3-8B-QLoRA-Custom",
                base_model="meta-llama/Meta-Llama-3-8B-Instruct",
                path="artifacts/checkpoints/qlora_llama_001",
                tags=["v1", "active"],
                hyperparameters={"learning_rate": 0.0002, "lora_r": 16},
                registered_at=datetime.utcnow()
            )
        raise CheckpointNotFoundError(
            f"Checkpoint ID '{checkpoint_id}' is not cataloged in LLMOps Studio."
        )

    def list_checkpoints(self) -> List[RegistryEntry]:
        """Queries stored checkpoints lists.

        Returns:
            List of RegistryEntry items.
        """
        logger.info("Fetching all cataloged model checkpoint entries")
        return [
            RegistryEntry(
                checkpoint_id="qlora_llama_001",
                model_name="LLaMA-3-8B-QLoRA-Custom",
                base_model="meta-llama/Meta-Llama-3-8B-Instruct",
                path="artifacts/checkpoints/qlora_llama_001",
                tags=["v1", "active"],
                hyperparameters={"learning_rate": 0.0002, "lora_r": 16},
                registered_at=datetime.utcnow()
            )
        ]
