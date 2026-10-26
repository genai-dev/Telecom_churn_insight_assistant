
"""PII protection for customer records and model inputs.

The LLM must receive only approved business evidence.
This module applies field allowlisting and masks common PII patterns.
"""

import re
from typing import Any

ALLOWED_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "customer_id",
        "type",
        "date",
        "source_system",
        "verified",
        "location_key",
        "status",
        "change_pct",
        "port_out",
        "requested_discount_pct",
        "requested_months",
    }
)

# Apply longer/specific patterns before shorter ones.
PII_PATTERNS = (
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "[EMAIL_REDACTED]",
    ),
    (
        re.compile(
            r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)"
        ),
        "[CARD_NUMBER_REDACTED]",
    ),
    (
        re.compile(
            r"(?<!\w)(?:\+91[-\s]?)?[6-9]\d{9}(?!\w)"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        re.compile(
            r"\b\d{1,5}\s+"
            r"(?:[A-Za-z0-9]+\s+){0,5}"
            r"(?:Street|Road|Lane|Avenue|Boulevard)\b"
            r"(?:[^\n,;]*)",
            flags=re.IGNORECASE,
        ),
        "[ADDRESS_REDACTED]",
    ),
)


def mask_pii(value: str) -> str:
    """Mask common PII patterns in untrusted text."""
    if not isinstance(value, str):
        raise TypeError("mask_pii expects a string")

    masked = value

    for pattern, replacement in PII_PATTERNS:
        masked = pattern.sub(replacement, masked)

    return masked


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize nested dictionaries, lists and strings."""
    if isinstance(value, str):
        return mask_pii(value)

    if isinstance(value, dict):
        return {
            key: sanitize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [sanitize_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_value(item) for item in value)

    return value


def redact_customer_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allow only approved record fields and mask their text values."""
    sanitized_records = []

    for record in records:
        if record.get("verified") is not True:
            continue

        approved_record = {
            key: value
            for key, value in record.items()
            if key in ALLOWED_RECORD_FIELDS
        }

        sanitized_records.append(sanitize_value(approved_record))

    return sanitized_records
