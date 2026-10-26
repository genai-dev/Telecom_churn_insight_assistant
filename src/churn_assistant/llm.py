
"""Grounded Ollama summarization with deterministic output guardrails.

The LLM only explains verified evidence. It does not calculate risk,
select retention actions, approve offers, or control escalation.
"""

import json
import logging
import os
import re

import httpx

from .config import ROOT, USE_LLM
from .schemas import InsightResponse
from .security import mask_pii

logger = logging.getLogger(__name__)

PROMPT_PATH = ROOT / "prompts" / "summarize.txt"

SOURCE_ID_PATTERN = re.compile(
    r"\b(?:TKT|CALL|USG|NOTE)-[A-Za-z0-9@-]+"
)

NUMBER_PATTERN = re.compile(
    r"\d+(?:\.\d+)?"
)

RISK_PATTERN = re.compile(
    r"\b(low|medium|high)\s+"
    r"(?:churn\s+)?risk\b",
    flags=re.IGNORECASE,
)

# Claims the summarizer must never make.
PROHIBITED_PROMISE_PATTERNS = (
    re.compile(
        r"\b(?:we|company|agent)\s+"
        r"(?:will|shall|guarantee)\s+"
        r"(?:refund|credit|discount|repair|fix|restore)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:approved|authorized|guaranteed)\s+"
        r"(?:refund|credit|discount|compensation)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:repair|service restoration)\s+"
        r"(?:will happen|is scheduled|is guaranteed)\b",
        flags=re.IGNORECASE,
    ),
)


def _allowed_citations(
    result: InsightResponse,
) -> set[str]:
    """Collect citations from verified dissatisfaction drivers."""

    return {
        citation
        for driver in result.drivers
        for citation in driver.citations
    }


def _allowed_numeric_values(
    result: InsightResponse,
) -> set[str]:
    """Collect numerical values already present in approved facts."""

    approved_text = " ".join(
        [
            driver.evidence
            + " "
            + " ".join(driver.citations)
            for driver in result.drivers
        ]
    )

    if result.risk is not None:
        approved_text += f" {result.risk.score}"

        for signal in result.risk.signals_used:
            approved_text += f" {signal}"

    return set(
        NUMBER_PATTERN.findall(approved_text)
    )


def _contains_unsupported_topic(
    summary: str,
    result: InsightResponse,
) -> bool:
    """Reject obvious dissatisfaction topics absent from retrieved drivers.

    This is an additional heuristic check, not a semantic proof
    that every generated statement is entailed by the evidence.
    """

    driver_names = {
        driver.driver
        for driver in result.drivers
    }

    text = summary.lower()

    topic_checks = {
        "network_quality": (
            "dropped call",
            "coverage complaint",
            "network quality",
            "network issue",
        ),
        "billing_dispute": (
            "billing dispute",
            "incorrect charge",
            "double charge",
            "billing issue",
        ),
        "competitor_intent": (
            "competitor offer",
            "competitor mention",
            "port-out",
            "port out",
        ),
        "usage_decline": (
            "usage decline",
            "usage drop",
            "usage decreased",
        ),
    }

    for driver_name, phrases in topic_checks.items():
        if driver_name in driver_names:
            continue

        if any(phrase in text for phrase in phrases):
            return True

    return False


def validate_generated_summary(
    summary: str,
    result: InsightResponse,
) -> bool:
    """Validate an LLM summary before accepting it."""

    if not isinstance(summary, str):
        return False

    summary = summary.strip()

    if not summary or len(summary) > 600:
        return False

    # The generated answer must not contain recognizable PII.
    if mask_pii(summary) != summary:
        return False

    # Every driver citation must be represented.
    allowed_citations = _allowed_citations(result)

    if not allowed_citations:
        return False

    if not all(
        citation in summary
        for citation in allowed_citations
    ):
        return False

    # No invented source-record identifiers.
    generated_citations = set(
        SOURCE_ID_PATTERN.findall(summary)
    )

    if not generated_citations.issubset(
        allowed_citations
    ):
        return False

    # No new numerical claims.
    generated_numbers = set(
        NUMBER_PATTERN.findall(summary)
    )

    approved_numbers = _allowed_numeric_values(result)

    if not generated_numbers.issubset(
        approved_numbers
    ):
        return False

    # The model may explain the risk but cannot change it.
    detected_risk_levels = RISK_PATTERN.findall(summary)

    if detected_risk_levels:
        if result.risk is None:
            return False

        if any(
            level.lower() != result.risk.level
            for level in detected_risk_levels
        ):
            return False

    if result.risk is None and re.search(
        r"\b(?:risk score|score of)\s+\d+",
        summary,
        flags=re.IGNORECASE,
    ):
        return False

    # No unauthorized promises.
    if any(
        pattern.search(summary)
        for pattern in PROHIBITED_PROMISE_PATTERNS
    ):
        return False

    # No obviously unrelated dissatisfaction topics.
    if _contains_unsupported_topic(
        summary,
        result,
    ):
        return not _contains_unsupported_topic(
             summary,
             result,
        )

    return True


def _build_verified_facts(
    result: InsightResponse,
) -> dict:
    """Construct the restricted JSON payload sent to the LLM."""

    return {
        "verified_drivers": [
            driver.model_dump(mode="json")
            for driver in result.drivers
        ],
        "fixed_risk": (
            result.risk.model_dump(mode="json")
            if result.risk is not None
            else None
        ),
        "missing_information": (
            result.missing_information
        ),
        "unsupported": result.unsupported,
        "data_conflicts": result.data_conflicts,
    }


def summarize_verified_insight(
    result: InsightResponse,
) -> str:
    """Generate a grounded summary using Ollama Cloud.

    Invalid or unavailable model responses fail closed to the
    deterministic Python-generated summary.
    """

    fallback_summary = result.summary

    if not USE_LLM:
        return fallback_summary

    if not result.drivers:
        return fallback_summary

    api_key = os.environ.get(
        "OLLAMA_API_KEY", ""
    ).strip()

    if not api_key:
        logger.warning(
            "Ollama key not configured; "
            "using deterministic summary."
        )
        return fallback_summary

    try:
        system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        verified_facts = _build_verified_facts(
            result
        )

        # Defense in depth: sanitize before external transmission.
        sanitized_facts = json.loads(
            mask_pii(
                json.dumps(
                    verified_facts,
                    ensure_ascii=False,
                )
            )
        )

        request_payload = {
            "model": os.getenv(
                "OLLAMA_MODEL",
                "gpt-oss:120b-cloud",
            ),
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Verified evidence JSON:\n"
                        + json.dumps(
                            sanitized_facts,
                            ensure_ascii=False,
                        )
                    ),
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        response = httpx.post(
            os.getenv(
                "OLLAMA_CLOUD_URL",
                "https://ollama.com/api/chat",
            ),
            headers={
                "Authorization": (
                    f"Bearer {api_key}"
                )
            },
            json=request_payload,
            timeout=45,
        )

        response.raise_for_status()

        response_body = response.json()

        raw_content = response_body[
            "message"
        ]["content"]

        # The prompt requires exactly one JSON object.
        parsed = json.loads(raw_content)

        if not isinstance(parsed, dict):
            return fallback_summary

        if set(parsed.keys()) != {"summary"}:
            return fallback_summary

        generated_summary = parsed["summary"]

        if not isinstance(
            generated_summary,
            str,
        ):
            return fallback_summary

        if not validate_generated_summary(
            generated_summary,
            result,
        ):
            logger.warning(
                "LLM summary rejected by grounding guard."
            )

            return fallback_summary

        return generated_summary.strip()

    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ):
        logger.warning(
            "Ollama summary unavailable; "
            "using deterministic summary."
        )

        return fallback_summary
