"""Dataset Content Hashing Module.

Calculates SHA-256 checksum hashes of raw datasets files to support caching
mechanisms and uniquely identify datasets records.
"""

import hashlib


def calculate_content_hash(content: bytes) -> str:
    """Computes a hexadecimal SHA-256 checksum representation for the binary content.

    Args:
        content: Raw binary data content of a dataset file.

    Returns:
        Cryptographic hex string checksum.
    """
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()
