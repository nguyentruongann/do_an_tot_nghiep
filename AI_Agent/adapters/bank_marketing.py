from typing import Dict, Any
from feature_store.transforms import age_band

class BankMarketingAdapter:
    def build_context(self, p: Dict[str, Any]) -> Dict[str, Any]:
        # payload: keep as-is
        payload = dict(p)
        # derived: simple useful fields for rule fallback or remote tabular
        derived = {
            "age_band": age_band(p.get("age")),
            "age": p.get("age"),
            "duration": p.get("duration"),
            "campaign": p.get("campaign"),
            "pdays": p.get("pdays"),
            "previous": p.get("previous"),
            "balance": p.get("balance"),
            "default": p.get("default"),
            "loan": p.get("loan"),
            "job": (p.get("job") or "").lower(),
            "education": (p.get("education") or "").lower(),
            "month": (p.get("month") or "").lower(),
        }
        schema = {
            "primary_key": "ID",
            "description": "UCI bank marketing style record used to predict term deposit propensity.",
            "fields": list(payload.keys())
        }
        industry = "banking"
        market = (p.get("country") or "unknown")
        return {"payload": payload, "derived": derived, "schema": schema, "industry": industry, "market": market}
