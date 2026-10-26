
"""Deterministic churn analysis, retention rules and guardrails.

The LLM cannot:
- Calculate or change the churn-risk score.
- Select an unauthorized retention action.
- Approve discounts, credits or policy exceptions.
- Override citation or supervisor-escalation checks.
"""

import re
from collections import Counter
from datetime import date, timedelta
from typing import Any

from .schemas import Driver, InsightResponse, Recommendation, Risk

DRIVER_ORDER = [
    "network_quality",
    "billing_dispute",
    "plan_fit",
    "service_experience",
    "competitor_intent",
    "usage_decline",
]

WEIGHTS = {
    "network_complaints_3plus": 3,
    "open_billing_dispute": 3,
    "competitor_or_portout": 3,
    "usage_drop_30pct": 2,
    "repeat_contact": 1,
}

ALLOWED_ACTIONS = frozenset(
    {
        "plan_review",
        "network_escalation",
        "bill_correction",
        "callback",
    }
)


def risk_band(score: int) -> str:
    """Return the fixed risk band defined by the assessment."""

    if score >= 6:
        return "high"

    if score >= 3:
        return "medium"

    return "low"


def is_recent(
    record: dict[str, Any],
    as_of: date,
    days: int = 30,
) -> bool:
    """Check whether a record falls in the requested time window."""

    try:
        record_date = date.fromisoformat(record["date"])
    except (KeyError, TypeError, ValueError):
        return False

    return (
        as_of - timedelta(days=days)
        <= record_date
        <= as_of
    )


def same_issue_repeat_contacts(
    records: list[dict[str, Any]],
) -> list[str]:
    """Identify repeat contacts supported by a shared issue key.

    For network complaints, the same location identifies the
    recurring network issue in the supplied assessment data.

    For other contact types, an explicit issue_key is required.
    """

    issue_groups: dict[tuple[str, str], list[str]] = {}

    for record in records:
        record_type = record.get("type", "")

        if record_type == "network_complaint":
            issue_key = record.get("location_key")
            category = "network"

        elif record_type in {
            "billing_dispute",
            "billing_followup",
        }:
            issue_key = record.get("issue_key")
            category = "billing"

        else:
            continue

        if not issue_key:
            continue

        group_key = (category, str(issue_key))

        issue_groups.setdefault(
            group_key, []
        ).append(record["record_id"])

    for citations in issue_groups.values():
        if len(set(citations)) >= 2:
            return list(dict.fromkeys(citations))

    return []



def extract_requested_discount(
    text: str,
) -> float | None:
    """Find an explicitly requested percentage discount."""

    patterns = [
        (
            r"(\d+(?:\.\d+)?)\s*%\s*"
            r"(?:off|discount)"
        ),
        (
            r"(?:discount|off)\s*(?:of\s*)?"
            r"(\d+(?:\.\d+)?)\s*%"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None



def extract_requested_months(
    text: str,
) -> int | None:
    """Extract an explicitly requested discount duration."""

    match = re.search(
        r"(\d+)\s*months?",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twelve": 12,
    }

    for word, number in word_numbers.items():
        if re.search(
            rf"\b{word}\s+months?\b",
            text,
            flags=re.IGNORECASE,
        ):
            return number

    return None



def extract_requested_credit(
    text: str,
) -> float | None:
    """Find a requested bill credit without confusing it with a bill."""

    patterns = [
        (
            r"(?:bill\s+)?credit\s+"
            r"(?:of\s*)?(?:rs\.?|inr|₹)?\s*"
            r"(\d+(?:\.\d+)?)"
        ),
        (
            r"(?:rs\.?|inr|₹)\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"(?:bill\s+)?credit"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None



def policy_exception_reasons(
    records: list[dict[str, Any]],
    question: str,
    stated_reason: str,
    policies: dict[str, Any],
) -> list[str]:
    """Detect requests exceeding the active retention policy."""

    reasons = []

    retention_policy = policies.get("RET-POL")

    if retention_policy is None:
        return [
            "RET-POL is unavailable; supervisor review required."
        ]

    max_discount = retention_policy["max_discount_pct"]
    max_months = retention_policy["max_discount_months"]
    max_credit = retention_policy["max_bill_credit_inr"]

    text = f"{question} {stated_reason}"

    discount_values = [
        float(record.get("requested_discount_pct", 0) or 0)
        for record in records
    ]

    duration_values = [
        int(record.get("requested_months", 0) or 0)
        for record in records
    ]

    credit_values = [
        float(record.get("requested_credit_inr", 0) or 0)
        for record in records
    ]

    text_discount = extract_requested_discount(text)
    text_duration = extract_requested_months(text)
    text_credit = extract_requested_credit(text)

    if text_discount is not None:
        discount_values.append(text_discount)

    if text_duration is not None:
        duration_values.append(text_duration)

    if text_credit is not None:
        credit_values.append(text_credit)

    # A duration mentioned without a discount or credit is not
    # automatically a retention-offer request.
    discount_requested = any(
        value > 0 for value in discount_values
    )

    if discount_values and max(discount_values) > max_discount:
        reasons.append(
            "Requested discount exceeds RET-POL "
            f"limit of {max_discount}%."
        )

    if (
        discount_requested
        and duration_values
        and max(duration_values) > max_months
    ):
        reasons.append(
            "Requested discount duration exceeds RET-POL "
            f"limit of {max_months} months."
        )

    if credit_values and max(credit_values) > max_credit:
        reasons.append(
            "Requested bill credit exceeds RET-POL "
            f"limit of Rs {max_credit}."
        )

    return reasons


def analyze(
    customer_id: str,
    customer: dict,
    records: list[dict],
    policies: dict,
    question: str,
    stated_reason: str,
    as_of: date,
) -> InsightResponse:
    """Calculate a grounded insight without delegating decisions to the LLM."""

    # Defense in depth: never score unverified records.
    verified_records = [
        record
        for record in records
        if record.get("verified") is True
    ]

    recent = [
        record
        for record in verified_records
        if is_recent(record, as_of)
    ]

    signals: dict[str, list[str]] = {}

    drivers: list[Driver] = []
    conflicts: list[str] = []
    missing: list[str] = []
    unsupported: list[str] = []

    # ---------------------------------------------------------
    # 1. Network complaints
    # ---------------------------------------------------------

    network = [
        record
        for record in recent
        if record.get("type") == "network_complaint"
    ]

    location_counts = Counter(
        record.get("location_key")
        for record in network
        if record.get("location_key")
    )

    eligible_locations = sorted(
        location
        for location, count in location_counts.items()
        if count >= 3
    )

    if eligible_locations:
        selected_location = eligible_locations[0]

        network_citations = [
            record["record_id"]
            for record in network
            if record.get("location_key") == selected_location
        ]

        signals["network_complaints_3plus"] = (
            network_citations
        )

        drivers.append(
            Driver(
                driver="network_quality",
                evidence=(
                    "3 or more verified coverage complaints "
                    "at the same location in 30 days"
                ),
                citations=network_citations,
            )
        )

    elif network:
        drivers.append(
            Driver(
                driver="network_quality",
                evidence="Verified network complaint",
                citations=[
                    record["record_id"]
                    for record in network
                ],
            )
        )

    # ---------------------------------------------------------
    # 2. Billing disputes and conflicts
    # ---------------------------------------------------------

    billing = [
        record
        for record in recent
        if record.get("type") in {
            "billing_dispute",
            "billing_followup",
        }
    ]

    billing.sort(
        key=lambda record: (
            record["date"],
            record["record_id"],
        )
    )

    if billing:
        latest = billing[-1]

        latest_status = latest.get(
            "status", "unknown"
        )

        if latest_status in {"open", "disputed"}:
            signals["open_billing_dispute"] = [
                latest["record_id"]
            ]

        drivers.append(
            Driver(
                driver="billing_dispute",
                evidence=(
                    "Latest verified billing status: "
                    f"{latest_status}"
                ),
                citations=[latest["record_id"]],
            )
        )

        statuses = {
            record.get("status")
            for record in billing
        }

        if (
            "resolved" in statuses
            and statuses.intersection(
                {"open", "disputed"}
            )
        ):
            conflict_ids = [
                record["record_id"]
                for record in billing
            ]

            conflicts.append(
                "Conflicting verified billing records: "
                + ", ".join(conflict_ids)
                + ". Latest verified contact determines "
                "the current reported status."
            )

    # ---------------------------------------------------------
    # 3. Competitor or port-out intent
    # ---------------------------------------------------------

    competitor = [
        record
        for record in recent
        if record.get("type") == "competitor_intent"
    ]

    if competitor:
        competitor_ids = [
            record["record_id"]
            for record in competitor
        ]

        signals["competitor_or_portout"] = (
            competitor_ids
        )

        drivers.append(
            Driver(
                driver="competitor_intent",
                evidence=(
                    "Verified competitor mention "
                    "or port-out intent"
                ),
                citations=competitor_ids,
            )
        )

    # ---------------------------------------------------------
    # 4. Usage decline
    # ---------------------------------------------------------

    usage = [
        record
        for record in recent
        if (
            record.get("type") == "usage"
            and isinstance(
                record.get("change_pct"),
                (int, float),
            )
        )
    ]

    usage.sort(
        key=lambda record: (
            record["date"],
            record["record_id"],
        )
    )

    if usage:
        latest_usage = usage[-1]

        change = latest_usage["change_pct"]

        if change < -30:
            usage_ids = [
                latest_usage["record_id"]
            ]

            signals["usage_drop_30pct"] = (
                usage_ids
            )

            drivers.append(
                Driver(
                    driver="usage_decline",
                    evidence=(
                        f"Usage decreased {abs(change):g}% "
                        "month on month"
                    ),
                    citations=usage_ids,
                )
            )

    # ---------------------------------------------------------
    # 5. Repeat contact for the same issue
    # ---------------------------------------------------------

    repeat_ids = same_issue_repeat_contacts(recent)

    if repeat_ids:
        signals["repeat_contact"] = repeat_ids

    # ---------------------------------------------------------
    # 6. Deterministic risk classification
    # ---------------------------------------------------------

    drivers.sort(
        key=lambda driver: DRIVER_ORDER.index(
            driver.driver
        )
    )

    score = sum(
        WEIGHTS[signal]
        for signal in signals
    )

    if recent:
        risk = Risk(
            level=risk_band(score),
            score=score,
            signals_used=list(signals),
        )
    else:
        risk = None

        missing.extend(
            [
                "recent customer interactions",
                "recent usage trend",
            ]
        )

    if not stated_reason.strip():
        missing.append(
            "customer's stated reason for calling"
        )

    # ---------------------------------------------------------
    # 7. Unsupported information
    # ---------------------------------------------------------

    query_text = (
        f"{question} {stated_reason}"
    ).lower()

    if (
        "competitor" in query_text
        and any(
            word in query_text
            for word in (
                "price",
                "cheaper",
                "cost",
                "pricing",
            )
        )
    ):
        unsupported.append(
            "Competitor pricing is not present "
            "in verified records."
        )

    if any(
        phrase in query_text
        for phrase in (
            "repair date",
            "when fixed",
            "restored",
            "repair time",
        )
    ):
        unsupported.append(
            "Network repair dates are not "
            "present in approved records."
        )

    if "network_complaints_3plus" in signals:
        missing.append(
            "customer's preferred callback time"
        )

        repair_notice = (
            "Network repair dates are not "
            "present in approved records."
        )

        if repair_notice not in unsupported:
            unsupported.append(repair_notice)

    festive_requested = any(
        phrase in query_text
        for phrase in (
            "festive",
            "offer-fest",
        )
    )

    if (
        festive_requested
        and "OFFER-FEST" not in policies
    ):
        unsupported.append(
            "OFFER-FEST has expired "
            "and cannot be recommended."
        )

    # ---------------------------------------------------------
    # 8. Select exactly one permitted action
    # ---------------------------------------------------------

    action = None

    if "open_billing_dispute" in signals:
        action = Recommendation(
            action="bill_correction",
            reasoning=(
                "Latest verified contact disputes "
                "the charge. Investigate and verify "
                "before correcting the bill."
            ),
            based_on_signals=[
                "open_billing_dispute"
            ],
            citations=(
                signals["open_billing_dispute"]
                + ["BILL-POL@v1"]
            ),
            follow_up="callback",
        )

    elif "network_complaints_3plus" in signals:
        action = Recommendation(
            action="network_escalation",
            reasoning=(
                "Verified same-location complaints "
                "meet the NET-ESC threshold."
            ),
            based_on_signals=[
                "network_complaints_3plus"
            ],
            citations=(
                signals["network_complaints_3plus"]
                + ["NET-ESC@v1"]
            ),
            follow_up="callback",
        )

    elif "competitor_or_portout" in signals:
        action = Recommendation(
            action="plan_review",
            reasoning=(
                "Review eligible plans. "
                "No discount is authorized "
                "by this recommendation."
            ),
            based_on_signals=[
                "competitor_or_portout"
            ],
            citations=(
                signals["competitor_or_portout"]
                + ["RET-POL@v1"]
            ),
        )

    elif signals:
        first_signal = next(iter(signals))

        action = Recommendation(
            action="callback",
            reasoning=(
                "Follow up on the verified "
                "dissatisfaction signal."
            ),
            based_on_signals=[
                first_signal
            ],
            citations=signals[first_signal],
        )

    # ---------------------------------------------------------
    # 9. Policy exceptions and escalation
    # ---------------------------------------------------------

    escalation_reasons = policy_exception_reasons(
        recent,
        question,
        stated_reason,
        policies,
    )

    if conflicts:
        escalation_reasons.append(
            "Unresolved disagreement between "
            "verified records requires supervisor review."
        )

    # ---------------------------------------------------------
    # 10. Grounded fallback summary
    # ---------------------------------------------------------

    if drivers:
        summary = (
            "; ".join(
                driver.evidence
                for driver in drivers
            )
            + "."
        )
    else:
        summary = (
            "No recent verified dissatisfaction "
            "signals available."
        )

    return InsightResponse(
        customer_id=customer_id,
        summary=summary,
        drivers=drivers,
        risk=risk,
        recommended_action=action,
        missing_information=missing,
        unsupported=unsupported,
        data_conflicts=conflicts,
        escalate=bool(escalation_reasons),
        escalation_reason=(
            " ".join(escalation_reasons)
            if escalation_reasons
            else None
        ),
    )


def validate(
    result: InsightResponse,
    records: list[dict],
    policies: dict,
) -> list[str]:
    """Validate the structured insight before returning it."""

    errors: list[str] = []

    verified_records = {
        record["record_id"]: record
        for record in records
        if record.get("verified") is True
    }

    allowed_citations = set(verified_records)

    allowed_citations.update(
        policy["citation"]
        for policy in policies.values()
        if "citation" in policy
    )

    # ---------------------------------------------------------
    # 1. Verify driver citations
    # ---------------------------------------------------------

    for driver in result.drivers:
        if not driver.citations:
            errors.append(
                f"Missing citations for {driver.driver}"
            )

        elif any(
            citation not in verified_records
            for citation in driver.citations
        ):
            errors.append(
                f"Invalid citations in {driver.driver}"
            )

    # ---------------------------------------------------------
    # 2. Validate driver order
    # ---------------------------------------------------------

    names = [
        driver.driver
        for driver in result.drivers
    ]

    expected_order = sorted(
        names,
        key=DRIVER_ORDER.index,
    )

    if names != expected_order:
        errors.append("Driver order incorrect")

    # ---------------------------------------------------------
    # 3. Verify deterministic risk
    # ---------------------------------------------------------

    if result.risk is not None:
        signals = result.risk.signals_used

        if len(signals) != len(set(signals)):
            errors.append("Duplicate scored signals")

        if any(
            signal not in WEIGHTS
            for signal in signals
        ):
            errors.append("Unknown scored signal")

        else:
            expected_score = sum(
                WEIGHTS[signal]
                for signal in signals
            )

            expected_level = risk_band(
                expected_score
            )

            if result.risk.score != expected_score:
                errors.append("Risk score mismatch")

            if result.risk.level != expected_level:
                errors.append("Risk level mismatch")

    elif not result.missing_information:
        errors.append(
            "Missing information required "
            "when risk is null"
        )

    # ---------------------------------------------------------
    # 4. Verify recommended action
    # ---------------------------------------------------------

    action = result.recommended_action

    if action is not None:
        if action.action not in ALLOWED_ACTIONS:
            errors.append(
                "Action is outside the approved catalog"
            )

        retention_policy = policies.get("RET-POL")

        if (
            retention_policy is not None
            and action.action not in retention_policy.get(
                "actions", []
            )
        ):
            errors.append(
                "Action is not allowed by RET-POL"
            )

        if not action.citations:
            errors.append(
                "Recommended action has no citations"
            )

        elif any(
            citation not in allowed_citations
            for citation in action.citations
        ):
            errors.append(
                "Recommended action has invalid citations"
            )

        if (
            result.risk is None
            or not action.based_on_signals
            or not set(
                action.based_on_signals
            ).issubset(
                set(result.risk.signals_used)
            )
        ):
            errors.append(
                "Recommended action is not grounded "
                "in scored signals"
            )

    # ---------------------------------------------------------
    # 5. Required policy checks
    # ---------------------------------------------------------

    if (
        action is not None
        and action.action == "network_escalation"
        and "NET-ESC" not in policies
    ):
        errors.append(
            "NET-ESC policy unavailable"
        )

    if (
        action is not None
        and action.action == "bill_correction"
        and "BILL-POL" not in policies
    ):
        errors.append(
            "BILL-POL policy unavailable"
        )

    if (
        action is not None
        and action.action == "plan_review"
        and "RET-POL" not in policies
    ):
        errors.append(
            "RET-POL policy unavailable"
        )

    # ---------------------------------------------------------
    # 6. Escalation consistency
    # ---------------------------------------------------------

    if (
        result.escalate
        and not result.escalation_reason
    ):
        errors.append(
            "Escalation reason is required"
        )

    if (
        result.data_conflicts
        and not result.escalate
    ):
        errors.append(
            "Unresolved data conflict "
            "must be escalated"
        )

    return errors
