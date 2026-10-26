from fastapi import FastAPI

from .graph import run_insight
from .schemas import InsightRequest, InsightResponse

app = FastAPI(title="Telecom Customer Churn Insight Assistant", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/insight", response_model=InsightResponse)
def insight(request: InsightRequest):
    return run_insight(request)
