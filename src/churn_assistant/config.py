import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)


def path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


DATA_PATH = path_env("CHURN_DATA_PATH", "data/customers.json")
POLICY_PATH = path_env("CHURN_POLICY_PATH", "data/policies.json")
USE_LLM = os.getenv("CHURN_USE_LLM", "false").lower() == "true"
