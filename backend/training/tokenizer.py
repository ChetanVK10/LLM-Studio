"""Dataset Token Formatting Module.

Handles tokenization formatting rules for Supervised Fine-Tuning (SFT),
masking prompt inputs (setting labels to -100) to isolate loss calculations.
"""

from typing import Any, Dict, List


class DatasetTokenFormatter:
    """Formats and tokenizes instruction records for Supervised Fine-Tuning."""

    def __init__(self, tokenizer: Any, max_length: int = 512) -> None:
        """Initializes the formatter.

        Args:
            tokenizer: Hugging Face PreTrainedTokenizer or mock wrapper.
            max_length: Maximum target sequence length parameter.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_and_tokenize(self, record: Dict[str, Any]) -> Dict[str, List[int]]:
        """Converts instruction-response pairs into token list dictionaries with masked labels.

        Args:
            record: Raw input data containing instruction, input, and output.

        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        instruction = record.get("instruction") or ""
        user_input = record.get("input") or ""
        response = record.get("output") or ""

        # Format prompt template
        prompt = f"Instruction: {instruction}\n"
        if user_input:
            prompt += f"Input: {user_input}\n"
        prompt += "Response: "

        # Retrieve BOS/EOS configurations
        bos_token = getattr(self.tokenizer, "bos_token", "") or ""
        eos_token = getattr(self.tokenizer, "eos_token", "") or ""

        # Encode prompt (including BOS token if present)
        prompt_with_bos = f"{bos_token}{prompt}"
        
        # We handle token encoding differently depending on whether it's a mock or real tokenizer
        if hasattr(self.tokenizer, "encode") and callable(self.tokenizer.encode):
            prompt_ids = self.tokenizer.encode(prompt_with_bos, add_special_tokens=False)
            response_ids = self.tokenizer.encode(f"{response}{eos_token}", add_special_tokens=False)
        else:
            # Fallback mock encoding for testing purposes
            prompt_ids = [ord(c) for c in prompt_with_bos]
            response_ids = [ord(c) for c in f"{response}{eos_token}"]

        input_ids = prompt_ids + response_ids
        
        # Mask prompt token labels with -100 value to ignore them in training loss
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncate sequence if it exceeds maximum limit
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
