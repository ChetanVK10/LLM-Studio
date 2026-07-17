"""Model Checkpoint Catalog index.

Placeholder module outlining future responsibilities for tracking and indexing model adapter
registries and tag management.
"""

import logging
from typing import List

from backend.schemas.registry import RegistryEntry

logger = logging.getLogger("app")


class ModelCatalog:
    """Manages indices matching checkpoint IDs, names, base foundation models, and active tags."""

    def __init__(self) -> None:
        """Initializes model catalog tracker instance."""
        logger.info("Initialized Registry Model Catalog index.")

    def add_entry(self, entry: RegistryEntry) -> RegistryEntry:
        """Adds a verified model checkpoint entry to indices catalogs.

        Args:
            entry: Checkpoint entry to insert.

        Returns:
            The inserted RegistryEntry.
        """
        logger.info(f"Catalog: Adding entry '{entry.checkpoint_id}' to indices.")
        return entry

    def search_by_tag(self, tag: str) -> List[RegistryEntry]:
        """Queries model catalogs containing tag constraints.

        Args:
            tag: Search label.

        Returns:
            List of matching RegistryEntry models.
        """
        logger.info(f"Catalog: Querying entries matching tag: '{tag}'")
        return []
