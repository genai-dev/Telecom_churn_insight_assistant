# Telecom Churn Insight Assistant

Evidence-grounded customer retention analysis | AI L2 — Agent Engineering

A LangGraph-orchestrated assistant for telecom service agents. It consolidates verified customer interactions, applies deterministic churn-risk rules, and recommends a policy-permitted next action. Ollama Cloud explains the evidence; Python owns scoring, eligibility, citations and escalation.

> Scope: assessment prototype using fictional records. No production CRM connection or real customer data.

## Use case

When a customer contacts support, relevant signals may be scattered across service tickets, call records, billing disputes and usage trends. Agents need a concise, trustworthy explanation of why the customer may leave and what they can do within retention policy.

The application accepts a customer ID, an agent question and the customer's stated reason. It retrieves verified records and currently valid policies, produces cited dissatisfaction drivers, calculates a LOW/MEDIUM/HIGH risk score, and recommends one of four actions: plan review, network escalation, bill correction or callback. Conflicts and policy exceptions are escalated for supervisor review.

For example, CUST-1001 has three verified same-location coverage complaints, a 35% usage decline and repeated contact. The predefined signals produce a score of 6 (HIGH); the recommended action is network escalation, with ticket and policy citations.

## Architecture

The application uses an eight-node LangGraph workflow:

1. Validate customer ID and reject unknown customers.
2. Retrieve verified customer records.
3. Retrieve active retention policies.
4. Sanitize evidence using field allowlisting and PII masking.
5. Derive dissatisfaction drivers, risk and action in Python.
6. Generate a brief, grounded Ollama explanation.
7. Validate citations, scoring and business guardrails.
8. Interrupt for supervisor review when escalation is required.

The LLM cannot change the score, choose an unauthorized action, approve a discount or override the escalation flag. A failed or unsafe model response falls back to the deterministic explanation.

### Decision rules

| Verified signal | Score |
| --- | ---: |
| At least three network complaints at one location within 30 days | +3 |
| Open or disputed billing issue | +3 |
| Competitor mention or port-out intent | +3 |
| Usage decline greater than 30% month over month | +2 |
| Repeat contact for the same issue | +1 |

Risk bands: **0–2 LOW**, **3–5 MEDIUM**, **6+ HIGH**. With no recent verified interactions, risk is `null` rather than LOW.

| Policy | Enforced behavior |
| --- | --- |
| RET-POL | Bill credit up to Rs 500 or discount up to 20% for three months; higher requests require review |
| NET-ESC | Escalate network cases with three or more same-location complaints in 30 days |
| BILL-POL | Investigate and verify disputed charges before correction; notify within 48 hours |
| OFFER-FEST | Exclude the offer after its validity end date (2026-10-08) |

The latest verified billing contact determines the current reported status, but contradictory earlier records remain visible and trigger escalation. A record is eligible only when its `verified` flag is `true`. Unverified NOTE-77 is excluded from both scoring and citations.

## Technology

| Component | Library / service |
| --- | --- |
| Runtime and dependency management | Python 3.12+, uv |
| Agent workflow | LangGraph |
| Generative explanation | Ollama Cloud, `gpt-oss:120b-cloud` |
| User interface | Streamlit |
| Optional REST interface | FastAPI, Uvicorn |
| Model HTTP calls | HTTPX |
| Schemas and validation | Pydantic |
| Local configuration | python-dotenv |
| Tests and linting | pytest, Ruff |

Dependencies are specified in `pyproject.toml` and locked in `uv.lock`.

## Run locally

Requirements: Python 3.12 or later, internet connectivity, and `uv`. VS Code is optional. Run commands from the project root—the folder containing `pyproject.toml`.

### macOS (zsh)

```bash
python3 --version
brew install uv
uv sync --extra dev --extra llm
cp .env.example .env
```

Open the newly created private `.env`, set `CHURN_USE_LLM=true`, and supply your own `OLLAMA_API_KEY`. Then start the UI:

```bash
uv run streamlit run src/churn_assistant/ui.py
```

### Windows (PowerShell)

```powershell
python --version
winget install --id=astral-sh.uv -e
uv sync --extra dev --extra llm
Copy-Item .env.example .env
```

Open `.env`, enable the LLM and configure your API key. Start the UI:

```powershell
uv run streamlit run src/churn_assistant/ui.py
```

Open the **Local URL** printed by Streamlit (typically `http://localhost:8501`; another port may be selected). Stop the process with **Ctrl+C**.

### Environment configuration

Create `.env` from the committed `.env.example`. The following values are examples, never credentials:

```dotenv
CHURN_DATA_PATH=data/customers.json
CHURN_POLICY_PATH=data/policies.json
CHURN_USE_LLM=true
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_CLOUD_URL=https://ollama.com/api/chat
OLLAMA_API_KEY=YOUR_PRIVATE_KEY_HERE
```

The application runs with a deterministic summary when the LLM is disabled or unavailable, but a live Ollama key is required to demonstrate real GenAI interaction. Never commit or submit `.env`.

## Verify the solution

### Live Streamlit check

Enter the following values and click **Analyse**:

| Field | Example |
| --- | --- |
| Customer ID | `CUST-1001` |
| Agent question | `Why might this customer leave, and what should I do?` |
| Channel | `Inbound Call` |
| Stated reason | `Calls keep dropping at home` |

Expected: HIGH risk, score 6; Network Escalation; Network Quality and Usage Decline drivers; verified ticket and usage citations; no supervisor review. The generated summary should cite `TKT-5521`, `TKT-5540`, `TKT-5562` and `USG-1001@2026-10`.

### Required golden cases

| Case | Scenario | Expected |
| --- | --- | --- |
| T1 | CUST-1001, dropped calls | HIGH risk, network escalation, verified citations |
| T2 | Unknown CUST-9999 | Unknown ID; no invented history |
| T3 | Unverified NOTE-77 | Never used for risk or cited |
| T4 | CUST-1002 billing conflict | TKT-6610 / CALL-8821 conflict shown; supervisor review |
| T5 | Competitor price question | Unsupported pricing; no fabricated comparison |
| T6 | Expired festive offer | Offer rejected as expired |
| T7 | 50% discount for six months | Supervisor escalation; no automatic approval |
| T8 | CUST-1005 with no recent history | `risk: null`; missing evidence reported |

Automated tests use a fixed session date of **2026-10-15** for repeatable policy-expiry results.

```bash
uv run pytest tests/test_golden.py -v
uv run ruff check src tests
```

Expected with the updated source and test suite: **36 passed** and **All checks passed!**. LLM unit tests use mocks; use Streamlit separately to verify a real Ollama response.

### Optional REST API

```bash
uv run uvicorn churn_assistant.api:app --reload
```

Open `http://127.0.0.1:8000/docs`, test `POST /insight`, or call `GET /health` (expected: `{"status":"ok"}`). Streamlit runs independently of this API process.

## Prompt design, safeguards and limitations

The system prompt lives in `prompts/summarize.txt`. It defines a strict trust boundary: retrieved content is evidence, not instructions; the response must use verified driver facts and exact citation IDs; the LLM must return a JSON object containing only a summary. `llm.py` checks citation coverage, unexpected source IDs and numerical claims, risk-band wording, common PII patterns and prohibited promises.

`security.py` allowlists record fields and masks common phone, email, payment-card and street-address patterns. PII pattern matching is not a complete privacy guarantee.

Escalation uses LangGraph `interrupt()` and supports programmatic approve/reject resume. Supervisor identity must be authenticated and authorized by any service that exposes the resume operation. Checkpoints currently live in memory, so pending reviews do not survive a process restart; the Streamlit UI displays the pending review but does not provide an authenticated approval console.

The fictional data pack contains the five specified assessment customers plus one learner-created example. Some record dates and other metadata were constructed because the scenario description does not provide them. Records must not be mistaken for actual customer history. Billing repeat-contact scoring requires evidence of a common issue; the assistant does not infer an unsupported connection. This prototype has no live CRM, payment, billing-adjustment or network-repair integrations.
