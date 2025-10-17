from typing import Dict, Any, List
import httpx

class RuleScorer:
    def predict(self, f: Dict[str, Any]):
        score = 50.0; reasons: List[str] = []
        dur = (f.get("duration") or 0)
        if dur >= 300: score += 20; reasons.append("call_long")
        elif dur >= 180: score += 12; reasons.append("call_med")
        elif dur >= 60: score += 5; reasons.append("call_short")
        camp = (f.get("campaign") or 0)
        if camp <= 2: score += 8; reasons.append("few_contacts")
        elif camp <= 4: score += 4; reasons.append("moderate_contacts")
        pdays = f.get("pdays")
        if pdays is not None:
            if pdays == -1: pass
            elif pdays < 10: score += 8; reasons.append("recent_contact")
            elif pdays < 20: score += 5; reasons.append("contact_20d")
            else: score += 2; reasons.append("contact_old")
        prev = (f.get("previous") or 0)
        if prev >= 2: score += 6; reasons.append("history2+")
        elif prev == 1: score += 3; reasons.append("history1")
        if (f.get("balance") or 0) >= 1500: score += 5; reasons.append("good_balance")
        if (f.get("age_band") or "") in ("45-54","55-64","65+"): score += 5; reasons.append("age45+")
        if f.get("default") == 1: score -= 12; reasons.append("has_default")
        if f.get("loan") == 1: score -= 5; reasons.append("has_loan")

        score = max(0.0, min(100.0, score))
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
        contribs = [{"name": r, "impact": 1.0} for r in reasons]
        return score, grade, ",".join(reasons), contribs

class RemoteScorer:
    def __init__(self, provider_cfg: Dict[str, Any]):
        self.cfg = provider_cfg
    def predict(self, feats: Dict[str, Any]):
        method = self.cfg.get("method", "POST").upper()
        endpoint = self.cfg["endpoint"]
        headers = self.cfg.get("headers") or {}
        req_key = self.cfg.get("request_key", "features")
        body = {req_key: feats}
        with httpx.Client(timeout=10.0) as client:
            if method == "POST":
                r = client.post(endpoint, json=body, headers=headers)
            else:
                r = client.get(endpoint, params=feats, headers=headers)
        r.raise_for_status()
        data = r.json()
        score_key = self.cfg.get("response_score_key", "score")
        score = float(data[score_key])
        scale = self.cfg.get("score_scale", "0-100")
        if scale == "0-1":
            score = max(0.0, min(1.0, score))*100.0
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
        contribs = []
        contrib_key = self.cfg.get("response_contrib_key")
        if contrib_key and contrib_key in data:
            raw = data[contrib_key]
            if isinstance(raw, dict):
                contribs = sorted([{"name":k,"impact":v} for k,v in raw.items()], key=lambda x: -abs(x["impact"]))[:10]
        rationale = ",".join([c["name"] for c in contribs[:3]]) if contribs else "remote_model_score"
        return score, grade, rationale, contribs
