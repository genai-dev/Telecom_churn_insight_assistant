
"""LangGraph workflow for the Telecom Churn Insight Assistant.

Workflow:
1. Check customer input
2. Retrieve verified records
3. Retrieve active policies
4. Sanitize records
5. Calculate risk and recommendation
6. Generate grounded explanation
7. Validate guardrails
8. Pause for supervisor review when required
"""

from datetime import date
from functools import lru_cache
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .engine import analyze, validate
from .llm import summarize_verified_insight
from .repository import Repository
from .schemas import InsightRequest, InsightResponse
from .security import redact_customer_records

SupervisorDecision = Literal["approve", "reject"]


class State(TypedDict, total=False):
    request: InsightRequest
    as_of: date
    customer: dict[str, Any] | None
    records: list[dict[str, Any]]
    policies: dict[str, Any]
    response: InsightResponse
    errors: list[str]
    supervisor_decision: str
    reviewed_by: str


def build_graph(
    repo: Repository | None = None,
    *,
    enable_interrupt: bool = True,
    checkpointer: Any = None,
):
    """Build the eight-node LangGraph workflow."""

    repository = repo or Repository()

    def check_input(state: State) -> dict:
        request = state["request"]

        customer = repository.customer(request.customer_id)

        if customer is None:
            response = InsightResponse(
                customer_id=request.customer_id,
                summary=(
                    "Customer ID not found. "
                    "No customer history was retrieved."
                ),
                missing_information=["valid customer ID"],
                unsupported=[
                    f"Unknown customer ID: {request.customer_id}"
                ],
                status="unknown_customer",
            )

            return {
                "customer": None,
                "response": response,
            }

        return {"customer": customer}

    def fetch_records(state: State) -> dict:
        records = repository.verified_records(
            state["request"].customer_id,
            state["as_of"],
        )

        return {"records": records}

    def fetch_policies(state: State) -> dict:
        policies = repository.active_policies(state["as_of"])

        return {"policies": policies}

    def redact(state: State) -> dict:
        """Remove unapproved fields and mask common PII."""

        sanitized_records = redact_customer_records(
            state["records"]
        )

        return {"records": sanitized_records}

    def derive(state: State) -> dict:
        """Calculate risk and action with deterministic Python."""

        request = state["request"]

        response = analyze(
            request.customer_id,
            state["customer"],
            state["records"],
            state["policies"],
            request.question,
            request.context.stated_reason,
            state["as_of"],
        )

        return {"response": response}

    def explain(state: State) -> dict:
        """Use the LLM only to explain verified evidence."""

        response = state["response"].model_copy(deep=True)

        response.summary = summarize_verified_insight(
            response
        )

        return {"response": response}

    def guard(state: State) -> dict:
        """Validate citations, risk and business constraints."""

        response = state["response"].model_copy(deep=True)

        errors = validate(
            response,
            state["records"],
            state["policies"],
        )

        if errors:
            response.escalate = True

            validation_reason = (
                "Guard validation failed: "
                + "; ".join(errors)
            )

            if response.escalation_reason:
                response.escalation_reason += (
                    " " + validation_reason
                )
            else:
                response.escalation_reason = validation_reason

        if response.escalate:
            response.status = "supervisor_pending"

        return {
            "response": response,
            "errors": errors,
        }

    def supervisor(state: State) -> dict:
        """Pause until an authorized supervisor reviews the case."""

        response = state["response"].model_copy(deep=True)

        if not enable_interrupt:
            return {"response": response}

        review = interrupt(
            {
                "customer_id": state["request"].customer_id,
                "reason": response.escalation_reason,
                "required_decision": [
                    "approve",
                    "reject",
                ],
                "instruction": (
                    "Supervisor review is required. "
                    "No discount, credit or exceptional "
                    "offer is automatically authorized."
                ),
            }
        )

        if not isinstance(review, dict):
            raise TypeError(
                "Supervisor response must be an object."
            )

        decision = review.get("decision")
        reviewer_id = str(
            review.get("reviewer_id", "")
        ).strip()

        if decision not in {"approve", "reject"}:
            raise ValueError(
                "Supervisor decision must be approve or reject."
            )

        if not reviewer_id:
            raise ValueError(
                "Supervisor reviewer ID is required."
            )

        if decision == "reject":
            response.recommended_action = None
            response.escalation_reason = (
                (response.escalation_reason or "")
                + " Supervisor rejected the proposed action."
            )

        else:
            response.escalation_reason = (
                (response.escalation_reason or "")
                + " Supervisor reviewed the case. "
                "No exceptional offer is automatically authorized."
            )

        response.status = "complete"

        return {
            "response": response,
            "supervisor_decision": decision,
            "reviewed_by": reviewer_id,
        }

    graph = StateGraph(State)

    graph.add_node("check_input", check_input)
    graph.add_node("fetch_records", fetch_records)
    graph.add_node("fetch_policies", fetch_policies)
    graph.add_node("redact", redact)
    graph.add_node("derive", derive)
    graph.add_node("explain", explain)
    graph.add_node("guard", guard)
    graph.add_node("supervisor", supervisor)

    graph.add_edge(START, "check_input")

    graph.add_conditional_edges(
        "check_input",
        lambda state: (
            "known" if state["customer"] else "unknown"
        ),
        {
            "known": "fetch_records",
            "unknown": END,
        },
    )

    graph.add_edge("fetch_records", "fetch_policies")
    graph.add_edge("fetch_policies", "redact")
    graph.add_edge("redact", "derive")
    graph.add_edge("derive", "explain")
    graph.add_edge("explain", "guard")

    graph.add_conditional_edges(
        "guard",
        lambda state: (
            "supervisor"
            if state["response"].escalate
            else "done"
        ),
        {
            "supervisor": "supervisor",
            "done": END,
        },
    )

    graph.add_edge("supervisor", END)

    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    """Reuse the default graph and checkpointer within the process."""

    return build_graph(
        checkpointer=MemorySaver()
    )


def run_insight(
    request: InsightRequest,
    *,
    repo: Repository | None = None,
    as_of: date | None = None,
    thread_id: str | None = None,
) -> InsightResponse:
    """Run a new request and return its structured insight."""

    session_date = as_of or date.today()

    # A unique thread prevents unrelated cases from sharing state.
    case_thread_id = thread_id or str(uuid4())

    if repo is None:
        graph = get_graph()
    else:
        # Custom repositories are primarily used for isolated runs.
        graph = build_graph(
            repo,
            checkpointer=MemorySaver(),
        )

    config = {
        "configurable": {
            "thread_id": case_thread_id
        }
    }

    result = graph.invoke(
        {
            "request": request,
            "as_of": session_date,
        },
        config=config,
    )

    return InsightResponse.model_validate(
        result["response"]
    )


def resume_insight(
    *,
    thread_id: str,
    decision: SupervisorDecision,
    reviewer_id: str,
) -> InsightResponse:
    """Resume a paused case after an authenticated supervisor review.

    The caller must enforce supervisor authentication and authorization.
    """

    if not thread_id.strip():
        raise ValueError("Thread ID is required.")

    if decision not in {"approve", "reject"}:
        raise ValueError("Invalid supervisor decision.")

    if not reviewer_id.strip():
        raise ValueError("Reviewer ID is required.")

    graph = get_graph()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    snapshot = graph.get_state(config)

    if not snapshot.next or "supervisor" not in snapshot.next:
        raise ValueError(
            "No supervisor-pending workflow exists "
            "for this thread."
        )

    result = graph.invoke(
        Command(
            resume={
                "decision": decision,
                "reviewer_id": reviewer_id,
            }
        ),
        config=config,
    )

    return InsightResponse.model_validate(
        result["response"]
    )
