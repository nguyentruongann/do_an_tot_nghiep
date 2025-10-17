import httpx, json
from typing import Dict, Any

class LLMScorer:
    def __init__(self, provider: Dict[str, Any]):
        self.endpoint = provider["endpoint"]
        self.headers  = provider.get("headers") or {}
        self.body_key = provider.get("request_key", "prompt")
        self.model    = provider.get("model", "")

    def predict(self, prompt: str) -> Dict[str, Any]:
        payload = {self.body_key: prompt}
        if self.model:
            payload["model"] = self.model

        with httpx.Client(timeout=60.0) as client:
            r = client.post(self.endpoint, json=payload, headers=self.headers)
        r.raise_for_status()

        # Try parse JSON directly or through common wrappers
        try:
            data = r.json()
            if isinstance(data, dict) and "choices" in data:
                text = data["choices"][0]["message"]["content"]
                out = json.loads(text)
            elif isinstance(data, dict) and "text" in data:
                out = json.loads(data["text"])
            else:
                out = data
        except ValueError:
            out = json.loads(r.text)

        # normalize output fields
        score = float(out["score"])
        score = max(0.0, min(100.0, score))
        rationale = out.get("rationale", "")
        contrib = out.get("contrib") or out.get("contribs") or {}
        if isinstance(contrib, dict):
            contrib = [{"name": k, "impact": v} for k, v in contrib.items()]
        return {"score": score, "rationale": rationale, "contrib": contrib}
