"""Dataset Validator Module.

Audits dataset record schemas, missing values, empty entries, and duplicate records
returning detailed severity-categorized ValidationIssue reports.
"""

from typing import Any, Dict, List

from backend.core.constants import ValidationSeverity
from backend.schemas.datasets import DatasetValidationReport, ValidationIssue


class DatasetValidator:
    """Audits record schemas and issues severity flags (INFO, WARNING, ERROR)."""

    def validate_records(self, records: List[Dict[str, Any]]) -> DatasetValidationReport:
        """Evaluates loaded records against target training expectations.

        Args:
            records: Parsed dataset rows as a list of dictionaries.

        Returns:
            DatasetValidationReport indicating validation issues and is_valid status.
        """
        issues: List[ValidationIssue] = []

        # 1. Validate complete absence of records
        if not records:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Dataset file contains no parsed records.",
                    affected_field="records"
                )
            )
            return DatasetValidationReport(is_valid=False, issues=issues)

        # 2. Extract column keys
        header_keys = set()
        for record in records:
            header_keys.update(record.keys())

        # Assert mandatory instruction fields exist in headers
        required_fields = {"instruction", "output"}
        missing_headers = required_fields - header_keys

        for header in missing_headers:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Dataset lacks required fine-tuning column header: '{header}'",
                    affected_field=header
                )
            )

        # 3. Assess row-level missing values and duplicate rows
        missing_instruction_rows = 0
        missing_output_rows = 0
        missing_input_rows = 0
        duplicate_rows_count = 0
        
        seen_records = set()

        for idx, row in enumerate(records):
            # Check row duplication by serializing items to a hashable sorted tuple
            row_fingerprint = tuple(sorted((k, str(v)) for k, v in row.items()))
            if row_fingerprint in seen_records:
                duplicate_rows_count += 1
            else:
                seen_records.add(row_fingerprint)

            # Evaluate instruction values
            inst_val = row.get("instruction")
            if inst_val is None or str(inst_val).strip() == "":
                missing_instruction_rows += 1

            # Evaluate output values
            out_val = row.get("output")
            if out_val is None or str(out_val).strip() == "":
                missing_output_rows += 1

            # Evaluate optional input field
            in_val = row.get("input")
            if in_val is None or str(in_val).strip() == "":
                missing_input_rows += 1

        # 4. Generate issues based on row metrics
        total_rows = len(records)

        if missing_instruction_rows > 0:
            severity = (
                ValidationSeverity.ERROR 
                if missing_instruction_rows == total_rows 
                else ValidationSeverity.WARNING
            )
            issues.append(
                ValidationIssue(
                    severity=severity,
                    message=f"Found {missing_instruction_rows} records with null or empty 'instruction' fields.",
                    affected_field="instruction"
                )
            )

        if missing_output_rows > 0:
            severity = (
                ValidationSeverity.ERROR 
                if missing_output_rows == total_rows 
                else ValidationSeverity.WARNING
            )
            issues.append(
                ValidationIssue(
                    severity=severity,
                    message=f"Found {missing_output_rows} records with null or empty 'output' fields.",
                    affected_field="output"
                )
            )

        if duplicate_rows_count > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {duplicate_rows_count} duplicate row entries.",
                    affected_field="records"
                )
            )

        if missing_input_rows > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Found {missing_input_rows} records with empty or omitted optional 'input' values.",
                    affected_field="input"
                )
            )

        # 5. Evaluate overall dataset validity
        # An ERROR severity issue blocks the validation path
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return DatasetValidationReport(is_valid=is_valid, issues=issues)
