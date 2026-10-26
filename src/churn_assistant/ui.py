
"""One-screen assessment UI matching the reference layout."""

import streamlit as st

from churn_assistant.engine import WEIGHTS
from churn_assistant.graph import run_insight
from churn_assistant.schemas import Context, InsightRequest

st.set_page_config(
    page_title="Telecom Churn Insight Assistant",
    layout="wide",
)

st.title("Telecom Churn Insight Assistant")

st.caption(
    "Agents enter the request on the left "
    "and read a cited answer on the right"
)


with st.container(border=True):

    left, right = st.columns([1, 1.25], gap="large")

    # ---------------------------
    # CUSTOMER INPUT SECTION
    # ---------------------------
    with left:

        st.subheader("Customer request")

        customer_id = st.text_input(
            "Customer ID",
            value="CUST-1001",
        )

        question = st.text_input(
            "Agent question",
            value="Why might this customer leave?",
        )

        channel = st.selectbox(
            "Channel",
            [
                "inbound_call",
                "outbound_call",
                "chat",
                "email",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        stated_reason = st.text_area(
            "Customer's stated reason",
            value="Calls keep dropping at home",
        )

        if st.button(
            "Analyse",
            type="primary",
            width="stretch",
        ):

            # Empty Customer ID guardrail
            if not customer_id.strip():

                st.session_state.pop("insight", None)

                st.error(
                    "Customer ID is required. "
                    "Enter a valid ID such as CUST-1001."
                )

            else:

                try:
                    request = InsightRequest(
                        customer_id=customer_id,
                        question=question,
                        context=Context(
                            channel=channel,
                            stated_reason=stated_reason,
                        ),
                    )

                    st.session_state["insight"] = run_insight(
                        request
                    )

                except Exception:  # noqa: BLE001 - Prevent internal errors from appearing in the UI
                    st.session_state.pop("insight", None)

                    st.error(
                        "Unable to process this request. "
                        "Check input and application setup."
                    )

    # ---------------------------
    # CHURN INSIGHT OUTPUT
    # ---------------------------
    with right:

        st.subheader("Churn insight")

        result = st.session_state.get("insight")

        if result is None:

            st.info(
                "Enter a customer request and "
                "click Analyse to see verified insights."
            )

        else:

            # ---------------------------
            # RISK CLASSIFICATION
            # ---------------------------
            if result.risk is None:

                st.warning(
                    "Churn risk: UNAVAILABLE — "
                    "insufficient recent evidence"
                )

            elif result.risk.level == "high":

                st.error(
                    f"Churn risk: HIGH · "
                    f"score {result.risk.score}"
                )

            elif result.risk.level == "medium":

                st.warning(
                    f"Churn risk: MEDIUM · "
                    f"score {result.risk.score}"
                )

            else:

                st.success(
                    f"Churn risk: LOW · "
                    f"score {result.risk.score}"
                )

            # ---------------------------
            # RETENTION RECOMMENDATION
            # ---------------------------
            if result.recommended_action:

                action = result.recommended_action

                action_name = (
                    action.action.replace("_", " ").title()
                )

                st.success(
                    f"Recommended action: {action_name}"
                )

                st.caption(action.reasoning)

                st.caption(
                    "Action sources: "
                    + ", ".join(action.citations)
                )

            else:

                st.info(
                    "Recommended action: none — "
                    "gather more verified information"
                )

            # ---------------------------
            # CUSTOMER SUMMARY
            # ---------------------------
            st.write(result.summary)

            # ---------------------------
            # VERIFIED EVIDENCE
            # ---------------------------
            st.markdown("**Drivers**")

            if result.drivers:

                driver_rows = [
                    {
                        "Driver": driver.driver.replace(
                            "_", " "
                        ).title(),
                        "Evidence": driver.evidence,
                        "Source": ", ".join(
                            driver.citations
                        ),
                    }
                    for driver in result.drivers
                ]

                st.dataframe(
                    driver_rows,
                    hide_index=True,
                    width="stretch",
                )

            else:

                st.caption(
                    "No verified dissatisfaction "
                    "drivers available"
                )

            # ---------------------------
            # RISK SCORE BREAKDOWN
            # ---------------------------
            if result.risk:

                breakdown = ", ".join(
                    f"{signal.replace('_', ' ')} "
                    f"+{WEIGHTS[signal]}"
                    for signal in result.risk.signals_used
                )

                st.caption(
                    "Score: "
                    + (breakdown or "No scored signals")
                )

            # ---------------------------
            # MISSING INFORMATION
            # ---------------------------
            if result.missing_information:

                st.warning(
                    "Missing: "
                    + "; ".join(
                        result.missing_information
                    )
                )

            # ---------------------------
            # UNSUPPORTED INFORMATION
            # ---------------------------
            if result.unsupported:

                st.info(
                    "Not supported: "
                    + "; ".join(result.unsupported)
                )

            # ---------------------------
            # DATA CONFLICTS
            # ---------------------------
            if result.data_conflicts:

                st.error(
                    "Data conflict: "
                    + "; ".join(result.data_conflicts)
                )

            # ---------------------------
            # SUPERVISOR ESCALATION
            # ---------------------------
            if result.escalate:

                st.error(
                    "Supervisor review: REQUIRED · "
                    + str(result.escalation_reason)
                )

            else:

                st.success(
                    "Supervisor review: not needed"
                )

            # ---------------------------
            # STRUCTURED JSON OUTPUT
            # ---------------------------
            with st.expander(
                "View structured JSON response"
            ):

                st.json(
                    result.model_dump(mode="json")
                )
