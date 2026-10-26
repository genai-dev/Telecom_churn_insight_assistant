
"""Golden and security tests for the Telecom Churn Insight Assistant."""

import json
from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from churn_assistant import llm
from churn_assistant.api import health, insight
from churn_assistant.engine import (
    policy_exception_reasons,
    same_issue_repeat_contacts,
    validate,
)
from churn_assistant.graph import (
    resume_insight,
    run_insight,
)
from churn_assistant.repository import Repository
from churn_assistant.schemas import (
    Context,
    InsightRequest,
)
from churn_assistant.security import (
    mask_pii,
    redact_customer_records,
)

SESSION = date(2026, 10, 15)


@pytest.fixture(autouse=True)
def disable_live_llm(monkeypatch):
    """Keep golden tests deterministic and independent of API access."""
    monkeypatch.setattr(llm, "USE_LLM", False)


def ask(
    customer_id: str,
    question: str = "Why might the customer leave?",
    reason: str = "",
    thread_id: str | None = None,
):
    """Run a test request using the fixed assessment session date."""

    return run_insight(
        InsightRequest(
            customer_id=customer_id,
            question=question,
            context=Context(
                channel="inbound_call",
                stated_reason=reason,
            ),
        ),
        as_of=SESSION,
        thread_id=thread_id,
    )


# ============================================================
# Eight mandatory assessment scenarios
# ============================================================


def test_t1_happy_path():
    """T1: High risk, network escalation, verified citations."""

    result = ask(
        "CUST-1001",
        question=(
            "Why might this customer leave, "
            "and what should I do?"
        ),
        reason="Calls keep dropping at home",
    )

    assert result.risk is not None
    assert result.risk.level == "high"
    assert result.risk.score == 6

    assert {
        driver.driver
        for driver in result.drivers
    } == {
        "network_quality",
        "usage_decline",
    }

    assert result.recommended_action is not None
    assert (
        result.recommended_action.action
        == "network_escalation"
    )

    assert result.escalate is False

    repository = Repository()

    errors = validate(
        result,
        repository.verified_records(
            "CUST-1001",
            SESSION,
        ),
        repository.active_policies(SESSION),
    )

    assert errors == []


def test_t2_unknown_customer():
    """T2: Unknown customer must not create fictional history."""

    result = ask("CUST-9999")

    assert result.status == "unknown_customer"

    assert result.risk is None
    assert result.drivers == []
    assert result.recommended_action is None

    assert "CUST-9999" in str(
        result.unsupported
    )


def test_t3_unverified_note_excluded():
    """T3: NOTE-77 must never affect scoring or citations."""

    result = ask(
        "CUST-1001",
        reason="Calls keep dropping at home",
    )

    assert result.risk is not None
    assert result.risk.score == 6

    serialized = result.model_dump_json()

    assert "NOTE-77" not in serialized


def test_t4_conflicting_billing():
    """T4: Conflicting billing records require supervisor review."""

    result = ask(
        "CUST-1002",
        question=(
            "Why is the roaming charge "
            "still on the bill?"
        ),
        reason="I was charged twice for roaming.",
    )

    assert "TKT-6610" in str(
        result.data_conflicts
    )

    assert "CALL-8821" in str(
        result.data_conflicts
    )

    assert result.recommended_action is not None

    assert (
        result.recommended_action.action
        == "bill_correction"
    )

    assert result.escalate is True
    assert result.status == "supervisor_pending"


def test_t5_competitor_price_unsupported():
    """T5: No invented competitor pricing."""

    result = ask(
        "CUST-1004",
        question="Is the competitor's plan cheaper?",
        reason="I want to switch to a competitor.",
    )

    assert "Competitor pricing" in str(
        result.unsupported
    )

    assert not any(
        "cheaper" in driver.evidence.lower()
        for driver in result.drivers
    )


def test_t6_expired_offer():
    """T6: OFFER-FEST must be expired on 2026-10-15."""

    result = ask(
        "CUST-1003",
        question=(
            "Can I give this customer "
            "the festive offer?"
        ),
        reason="My data connection was slow once.",
    )

    assert "expired" in str(
        result.unsupported
    ).lower()

    assert "OFFER-FEST@v1" not in str(
        result.recommended_action
    )


def test_t7_policy_exception():
    """T7: A 50% discount for six months requires escalation."""

    result = ask(
        "CUST-1004",
        question=(
            "Can I offer 50% off for 6 months?"
        ),
        reason=(
            "I will port out unless I receive "
            "50% off for six months."
        ),
    )

    assert result.escalate is True
    assert result.status == "supervisor_pending"

    assert "RET-POL" in str(
        result.escalation_reason
    )


def test_t8_sparse_history():
    """T8: No recent records means unknown risk, not low risk."""

    result = ask(
        "CUST-1005",
        question="",
        reason="",
    )

    assert result.risk is None
    assert result.drivers == []
    assert result.recommended_action is None

    assert "recent customer interactions" in (
        result.missing_information
    )

    assert "recent usage trend" in (
        result.missing_information
    )


# ============================================================
# Additional customer and scoring tests
# ============================================================


def test_additional_customer():
    """The learner-added customer must be supported."""

    result = ask("CUST-1006")

    assert result.risk is not None
    assert result.risk.level == "low"
    assert result.risk.score == 1


def test_repeat_contact_requires_same_issue():
    """Different issues must not count as repeat contact."""

    records = [
        {
            "record_id": "TKT-A",
            "type": "network_complaint",
            "location_key": "CELL-A",
        },
        {
            "record_id": "TKT-B",
            "type": "network_complaint",
            "location_key": "CELL-B",
        },
    ]

    assert same_issue_repeat_contacts(records) == []


def test_repeat_contact_same_location():
    """Two contacts about the same network location count."""

    records = [
        {
            "record_id": "TKT-A",
            "type": "network_complaint",
            "location_key": "CELL-A",
        },
        {
            "record_id": "TKT-B",
            "type": "network_complaint",
            "location_key": "CELL-A",
        },
    ]

    assert same_issue_repeat_contacts(records) == [
        "TKT-A",
        "TKT-B",
    ]


# ============================================================
# Privacy and data-verification tests
# ============================================================


def test_mask_phone_number():
    text = "Please call me on 9876543210."

    output = mask_pii(text)

    assert "9876543210" not in output
    assert "[PHONE_REDACTED]" in output


def test_mask_email_address():
    text = "My email is customer@example.com"

    output = mask_pii(text)

    assert "customer@example.com" not in output
    assert "[EMAIL_REDACTED]" in output


def test_mask_card_number():
    text = "My card number is 4111111111111111."

    output = mask_pii(text)

    assert "4111111111111111" not in output
    assert "[CARD_NUMBER_REDACTED]" in output


def test_mask_street_address():
    text = "I live at 123 Main Street."

    output = mask_pii(text)

    assert "123 Main Street" not in output
    assert "[ADDRESS_REDACTED]" in output


def test_unverified_record_removed():
    """Unverified records cannot proceed through evidence retrieval."""

    records = [
        {
            "record_id": "TKT-100",
            "type": "network_complaint",
            "verified": True,
            "customer_id": "CUST-1001",
            "date": "2026-10-10",
            "source_system": "tickets",
            "location_key": "CELL-A",
        },
        {
            "record_id": "NOTE-77",
            "type": "agent_note",
            "verified": False,
            "customer_id": "CUST-1001",
            "date": "2026-10-10",
            "source_system": "crm",
            "text": "Unverified speculation",
        },
    ]

    output = redact_customer_records(records)

    assert len(output) == 1
    assert output[0]["record_id"] == "TKT-100"
    assert "NOTE-77" not in str(output)


def test_unapproved_record_fields_removed():
    records = [
        {
            "record_id": "TKT-100",
            "verified": True,
            "type": "network_complaint",
            "customer_id": "CUST-1001",
            "date": "2026-10-10",
            "source_system": "tickets",
            "location_key": "CELL-A",
            "customer_phone": "9876543210",
            "customer_name": "Example Customer",
        }
    ]

    output = redact_customer_records(records)

    assert "customer_phone" not in output[0]
    assert "customer_name" not in output[0]


# ============================================================
# Retention-policy tests
# ============================================================


def test_discount_above_policy_limit():
    policies = Repository().active_policies(SESSION)

    reasons = policy_exception_reasons(
        [],
        "Can I offer 25% discount for 4 months?",
        "",
        policies,
    )

    assert any(
        "discount exceeds" in reason.lower()
        for reason in reasons
    )

    assert any(
        "duration exceeds" in reason.lower()
        for reason in reasons
    )


def test_bill_credit_above_policy_limit():
    policies = Repository().active_policies(SESSION)

    reasons = policy_exception_reasons(
        [],
        "Can I give a bill credit of Rs 800?",
        "",
        policies,
    )

    assert any(
        "bill credit exceeds" in reason.lower()
        for reason in reasons
    )


def test_discount_within_policy_limit():
    policies = Repository().active_policies(SESSION)

    reasons = policy_exception_reasons(
        [],
        "Can I offer a 20% discount for 3 months?",
        "",
        policies,
    )

    assert reasons == []


# ============================================================
# Supervisor interrupt/resume tests
# ============================================================


def test_supervisor_rejects_pending_case():
    """A rejected case resumes without retaining its proposed action."""

    thread_id = str(uuid4())

    pending = ask(
        "CUST-1004",
        question="Can I offer 50% off for 6 months?",
        thread_id=thread_id,
    )

    assert pending.status == "supervisor_pending"

    reviewed = resume_insight(
        thread_id=thread_id,
        decision="reject",
        reviewer_id="TEST-SUPERVISOR",
    )

    assert reviewed.status == "complete"
    assert reviewed.recommended_action is None

    assert "rejected" in str(
        reviewed.escalation_reason
    ).lower()


def test_supervisor_approves_review_without_offer():
    """Approval resumes the workflow but does not grant a discount."""

    thread_id = str(uuid4())

    pending = ask(
        "CUST-1004",
        question="Can I offer 50% off for 6 months?",
        thread_id=thread_id,
    )

    assert pending.status == "supervisor_pending"

    reviewed = resume_insight(
        thread_id=thread_id,
        decision="approve",
        reviewer_id="TEST-SUPERVISOR",
    )

    assert reviewed.status == "complete"

    assert reviewed.recommended_action is not None

    assert (
        reviewed.recommended_action.action
        == "plan_review"
    )

    assert "No exceptional offer" in str(
        reviewed.escalation_reason
    )


def test_cannot_resume_nonexistent_case():
    with pytest.raises(ValueError):
        resume_insight(
            thread_id=str(uuid4()),
            decision="approve",
            reviewer_id="TEST-SUPERVISOR",
        )


# ============================================================
# Citation and model-output guardrail tests
# ============================================================


def test_invalid_citation_rejected():
    result = ask("CUST-1001")

    result.drivers[0].citations = [
        "NOTE-77"
    ]

    repository = Repository()

    errors = validate(
        result,
        repository.verified_records(
            "CUST-1001",
            SESSION,
        ),
        repository.active_policies(SESSION),
    )

    assert errors


def test_llm_rejects_unsupported_citation():
    result = ask("CUST-1003")

    assert not llm.validate_generated_summary(
        "Network complaint [TKT-9999]",
        result,
    )


def test_llm_rejects_invented_numbers():
    result = ask("CUST-1003")

    assert not llm.validate_generated_summary(
        "Network complaint [TKT-7100], "
        "compensation Rs 5000.",
        result,
    )


def test_llm_rejects_wrong_risk():
    result = ask("CUST-1003")

    assert not llm.validate_generated_summary(
        "Verified network complaint [TKT-7100]; "
        "high risk.",
        result,
    )


def test_llm_requires_all_driver_citations():
    result = ask("CUST-1001")

    assert not llm.validate_generated_summary(
        "Network complaint [TKT-5521]",
        result,
    )


def test_llm_rejects_personal_information():
    result = ask("CUST-1003")

    assert not llm.validate_generated_summary(
        "Network complaint [TKT-7100]. "
        "Call 9876543210 for support.",
        result,
    )


def test_llm_rejects_unauthorized_promise():
    result = ask("CUST-1003")

    assert not llm.validate_generated_summary(
        "Network complaint [TKT-7100]. "
        "We will refund the customer.",
        result,
    )


# ============================================================
# LLM integration test (mocked, no real API usage)
# ============================================================


def test_ollama_uses_prompt_file_and_secret_header(
    monkeypatch,
    tmp_path,
):
    result = ask("CUST-1003")

    monkeypatch.setattr(
        llm,
        "USE_LLM",
        True,
    )

    monkeypatch.setenv(
        "OLLAMA_API_KEY",
        "synthetic-test-key",
    )

    prompt_file = tmp_path / "summarize.txt"

    prompt_file.write_text(
        "Summarize verified evidence only.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        llm,
        "PROMPT_PATH",
        prompt_file,
    )

    captured = {}

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": (
                                "Verified network complaint "
                                "[TKT-7100]; low risk score 0."
                            )
                        }
                    )
                }
            }

    def stub_post(
        url,
        headers,
        json,
        timeout,
    ):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json

        return StubResponse()

    monkeypatch.setattr(
        llm.httpx,
        "post",
        stub_post,
    )

    output = llm.summarize_verified_insight(
        result
    )

    assert "TKT-7100" in output

    assert captured["headers"][
        "Authorization"
    ] == "Bearer synthetic-test-key"

    assert "synthetic-test-key" not in str(
        captured["payload"]
    )

    assert (
        captured["payload"]["messages"][0]["content"]
        == prompt_file.read_text()
    )


# ============================================================
# API and schema tests
# ============================================================


def test_api_health():
    assert health() == {
        "status": "ok"
    }


def test_api_insight():
    result = insight(
        InsightRequest(
            customer_id="CUST-1001",
            question="Why might this customer leave?",
        )
    )

    assert result.risk is not None
    assert result.risk.level == "high"


def test_invalid_customer_id():
    with pytest.raises(ValidationError):
        InsightRequest(
            customer_id="bad id"
        )


def test_customer_id_normalization():
    request = InsightRequest(
        customer_id="cust-1001"
    )

    assert request.customer_id == "CUST-1001"


def test_extra_request_fields_rejected():
    with pytest.raises(ValidationError):
        InsightRequest.model_validate(
            {
                "customer_id": "CUST-1001",
                "unexpected_field": "test",
            }
        )
