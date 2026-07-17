"""Dataset Preview and Profile Generator.

Extracts statistical information (row counts, null densities, duplication indices)
and estimates dataset token counts utilizing the TokenEstimator abstraction.
"""

from typing import Any, Dict, List

from backend.datasets.token_estimator import TokenEstimator
from backend.schemas.datasets import DatasetProfile


class DatasetPreviewGenerator:
    """Generates preview lists and compiles statistical summary profiles."""

    def __init__(self, token_estimator: TokenEstimator) -> None:
        """Initializes the preview manager.

        Args:
            token_estimator: Target estimator to convert strings to token counts.
        """
        self.token_estimator = token_estimator

    def generate_profile(self, records: List[Dict[str, Any]]) -> DatasetProfile:
        """Runs aggregation metrics across all rows to construct a DatasetProfile.

        Args:
            records: Loaded list of dictionaries representing record rows.

        Returns:
            Completed DatasetProfile.
        """
        if not records:
            return DatasetProfile()

        # 1. Gather all header columns
        columns_set = set()
        for record in records:
            columns_set.update(record.keys())
        columns = sorted(list(columns_set))

        # 2. Track missing fields and compute lengths
        missing_values: Dict[str, int] = {col: 0 for col in columns}
        total_prompt_chars = 0
        total_response_chars = 0
        total_tokens_estimated = 0

        seen_records = set()
        duplicates_count = 0

        for record in records:
            # Check duplicates
            fingerprint = tuple(sorted((k, str(v)) for k, v in record.items()))
            if fingerprint in seen_records:
                duplicates_count += 1
            else:
                seen_records.add(fingerprint)

            # Check null cells
            for col in columns:
                val = record.get(col)
                if val is None or str(val).strip() == "":
                    missing_values[col] += 1

            # Compile text strings for tokens and lengths estimation
            instruction = str(record.get("instruction") or "")
            user_input = str(record.get("input") or "")
            response = str(record.get("output") or "")

            prompt_combined = f"{instruction} {user_input}".strip()
            
            total_prompt_chars += len(prompt_combined)
            total_response_chars += len(response)

            # Estimate token bounds
            total_tokens_estimated += self.token_estimator.estimate_tokens(prompt_combined)
            total_tokens_estimated += self.token_estimator.estimate_tokens(response)

        total_rows = len(records)
        avg_prompt_len = float(total_prompt_chars) / total_rows if total_rows > 0 else 0.0
        avg_response_len = float(total_response_chars) / total_rows if total_rows > 0 else 0.0

        return DatasetProfile(
            rows_count=total_rows,
            columns=columns,
            duplicate_count=duplicates_count,
            missing_values=missing_values,
            avg_prompt_length=round(avg_prompt_len, 2),
            avg_response_length=round(avg_response_len, 2),
            estimated_tokens=total_tokens_estimated
        )

    def get_preview_samples(self, records: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        """Extracts the first N sample records for table rendering.

        Args:
            records: Loaded list of dictionaries.
            n: Limit of samples to fetch.

        Returns:
            Truncated list of sample rows.
        """
        return records[:n]
