"""Unit tests for dataset validators.

Validates that schema validation successfully flags empty datasets, missing headers,
and duplicate rows with correct severity indicators.
"""

from backend.core.constants import ValidationSeverity
from backend.datasets.validator import DatasetValidator


def test_validator_valid_records() -> None:
    """Asserts that clean conforming records lists produce valid reports with no issues."""
    validator = DatasetValidator()
    records = [
        {"instruction": "Explain gravity", "input": "context", "output": "Gravity is..."},
        {"instruction": "Calculate sum", "input": "2+2", "output": "4"}
    ]
    report = validator.validate_records(records)
    assert report.is_valid is True
    assert len(report.issues) == 0


def test_validator_missing_headers() -> None:
    """Asserts that missing mandatory columns generate blocking ERROR flags."""
    validator = DatasetValidator()
    # Missing mandatory output header
    records = [
        {"instruction": "Explain gravity", "input": ""}
    ]
    report = validator.validate_records(records)
    assert report.is_valid is False
    assert any(
        issue.severity == ValidationSeverity.ERROR and issue.affected_field == "output" 
        for issue in report.issues
    )


def test_validator_duplicates() -> None:
    """Asserts duplicate rows trigger warning reports but do not block ingestion."""
    validator = DatasetValidator()
    records = [
        {"instruction": "Explain gravity", "input": "", "output": "Gravity"},
        {"instruction": "Explain gravity", "input": "", "output": "Gravity"}
    ]
    report = validator.validate_records(records)
    assert report.is_valid is True  # WARNING severity issues are non-blocking
    assert any(
        issue.severity == ValidationSeverity.WARNING and "duplicate" in issue.message.lower() 
        for issue in report.issues
    )


def test_validator_empty_dataset() -> None:
    """Asserts completely empty dataset inputs generate blocking errors."""
    validator = DatasetValidator()
    report = validator.validate_records([])
    assert report.is_valid is False
    assert any(issue.severity == ValidationSeverity.ERROR for issue in report.issues)
