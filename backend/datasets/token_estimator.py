"""Token Estimator Framework.

Defines the abstract interface TokenEstimator and the default HeuristicTokenEstimator,
facilitating pluggable tokenizer injection (e.g. Hugging Face LLaMA tokenizers) in future phases.
"""

from abc import ABC, abstractmethod


class TokenEstimator(ABC):
    """Abstract base class representing a token counter."""

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimates the number of tokens required to represent a string.

        Args:
            text: Target string content.

        Returns:
            Estimated count of tokens.
        """
        pass


class HeuristicTokenEstimator(TokenEstimator):
    """Heuristic estimator mapping average word lengths to token estimates."""

    def estimate_tokens(self, text: str) -> int:
        """Heuristically counts tokens using word length scaling (1 word ≈ 1.3 tokens).

        Args:
            text: Target string content.

        Returns:
            Integer representing estimated tokens count.
        """
        if not text or not isinstance(text, str):
            return 0
        
        words = text.split()
        if not words:
            return 0
            
        return max(1, int(len(words) * 1.3))
