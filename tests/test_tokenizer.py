"""Unit tests for the dataset token formatter.

Validates that prompt inputs are masked with -100 labels for SFT optimization.
"""

from backend.training.tokenizer import DatasetTokenFormatter


class MockTokenizer:
    """Mock tokenizer returning list of integer character codes."""

    def __init__(self) -> None:
        self.bos_token = "<s>"
        self.eos_token = "</s>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes characters to their integer codes.

        Args:
            text: Target string content.
            add_special_tokens: Mock parameter.

        Returns:
            List of integers.
        """
        return [ord(c) for c in text]


def test_dataset_token_formatter_masking() -> None:
    """Asserts prompt tokens labels map to -100 and responses match token codes."""
    tokenizer = MockTokenizer()
    formatter = DatasetTokenFormatter(tokenizer, max_length=512)

    record = {
        "instruction": "Explain gravity",
        "input": "context",
        "output": "Gravity attracts."
    }

    result = formatter.format_and_tokenize(record)

    # 1. Assert required inputs are returned
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result

    input_ids = result["input_ids"]
    labels = result["labels"]

    # 2. Count prompt mask size
    prompt_mask_count = sum(1 for val in labels if val == -100)
    assert prompt_mask_count > 0

    # 3. Assert prompt labels are strictly -100
    assert labels[:prompt_mask_count] == [-100] * prompt_mask_count

    # 4. Assert responses labels match input character codes
    assert labels[prompt_mask_count:] == input_ids[prompt_mask_count:]
    assert len(result["attention_mask"]) == len(input_ids)
