import json
from datetime import date

from .config import DATA_PATH, POLICY_PATH


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


class Repository:
    def __init__(self, customers=None, policies=None):
        self.customers = customers if customers is not None else load_json(DATA_PATH)
        self.policies = policies if policies is not None else load_json(POLICY_PATH)

    def customer(self, customer_id: str):
        return self.customers.get(customer_id)

    def verified_records(self, customer_id: str, as_of: date):
        customer = self.customer(customer_id)
        if customer is None:
            return []
        return [r for r in customer["records"] if r.get("verified") is True
                and date.fromisoformat(r["date"]) <= as_of]

    def active_policies(self, as_of: date):
        return {k: p for k, p in self.policies.items()
                if (not p.get("valid_from") or date.fromisoformat(p["valid_from"]) <= as_of)
                and (not p.get("valid_until") or date.fromisoformat(p["valid_until"]) >= as_of)}
