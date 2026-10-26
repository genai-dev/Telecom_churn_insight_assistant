
"""Pydantic schemas for the Telecom Churn Insight Assistant.

These models define the assessment's request/response contracts.
Business scoring and citation verification remain in engine.py.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

DriverName = Literal[
    "network_quality",
    "billing_dispute",
    "plan_fit",
    "service_experience",
    "competitor_intent",
    "usage_decline",
]

ActionName = Literal[
    "plan_review",
    "network_escalation",
    "bill_correction",
    "callback",
]

RiskLevel = Literal[
    "low",
    "medium",
    "high",
]

RequestChannel = Literal[
    "inbound_call",
    "outbound_call",
    "chat",
    "email",
]


class Context(BaseModel):
    """Context provided by the customer-service agent."""

    model_config = ConfigDict(extra="forbid")

    channel: RequestChannel = "inbound_call"

    stated_reason: str = Field(
        default="",
        max_length=1000,
        description=(
            "Customer's stated reason for contacting support."
        ),
    )


class InsightRequest(BaseModel):
    """Input request for customer churn analysis."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9-]+$",
        description="Customer identifier, e.g. CUST-1001.",
    )

    question: str = Field(
        default="",
        max_length=1000,
        description="Question asked by the service agent.",
    )

    context: Context = Field(
        default_factory=Context,
    )

    @field_validator("customer_id")
    @classmethod
    def normalize_customer_id(cls, value: str) -> str:
        """Normalize customer ID without accepting invalid formats."""

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "Customer ID cannot be empty."
            )

        return normalized


class Driver(BaseModel):
    """A dissatisfaction driver supported by source records."""

    model_config = ConfigDict(extra="forbid")

    driver: DriverName

    evidence: str = Field(
        min_length=1,
        description=(
            "Evidence describing the dissatisfaction driver."
        ),
    )

    citations: list[str] = Field(
        default_factory=list,
        description=(
            "Verified source-record identifiers."
        ),
    )


class Risk(BaseModel):
    """Deterministic churn-risk calculation."""

    model_config = ConfigDict(extra="forbid")

    level: RiskLevel

    score: int = Field(
        ge=0,
        description=(
            "Churn score calculated by deterministic Python rules."
        ),
    )

    signals_used: list[str] = Field(
        default_factory=list,
        description=(
            "Scored signals contributing to churn risk."
        ),
    )


class Recommendation(BaseModel):
    """One retention action from the approved catalog."""

    model_config = ConfigDict(extra="forbid")

    action: ActionName

    reasoning: str = Field(
        min_length=1,
        description=(
            "Explanation grounded in customer evidence."
        ),
    )

    based_on_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Scored signals supporting the selected action."
        ),
    )

    citations: list[str] = Field(
        default_factory=list,
        description=(
            "Verified customer records and policy citations."
        ),
    )

    follow_up: ActionName | None = Field(
        default=None,
        description="Optional permitted follow-up action.",
    )


class InsightResponse(BaseModel):
    """Structured response required by the assessment."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str

    summary: str = ""

    drivers: list[Driver] = Field(
        default_factory=list,
    )

    risk: Risk | None = None

    recommended_action: Recommendation | None = None

    missing_information: list[str] = Field(
        default_factory=list,
    )

    unsupported: list[str] = Field(
        default_factory=list,
    )

    data_conflicts: list[str] = Field(
        default_factory=list,
    )

    escalate: bool = False

    escalation_reason: str | None = None

    # Internal workflow status is useful for the application UI.
    status: Literal[
        "complete",
        "supervisor_pending",
        "unknown_customer",
    ] = "complete"


class SupervisorReviewRequest(BaseModel):
    """Input for resuming an interrupted supervisor-review workflow.

    Authentication and authorization must be enforced by the API
    or application layer before accepting this request.
    """

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(
        min_length=1,
        max_length=128,
    )

    decision: Literal[
        "approve",
        "reject",
    ]

    reviewer_id: str = Field(
        min_length=1,
        max_length=128,
    )


class HealthResponse(BaseModel):
    """Simple API health-check response."""

    status: Literal["ok"] = "ok"
