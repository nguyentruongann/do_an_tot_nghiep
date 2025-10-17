import json
from typing import Dict, Any
from models.registry import get_model_provider, get_app_config
from models.scorer_llm import LLMScorer
from models.scorer import RuleScorer, RemoteScorer
from database.repositories import save_score

def llm_score_json(prompt_ctx: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_app_config()
    version = cfg["app"]["model_version"]
    provider = get_model_provider(version)

    # prefer LLM
    if provider["kind"] == "llm":
        scorer = LLMScorer(provider)
        try:
            out = scorer.predict(prompt_ctx["prompt"])
            return out
        except Exception as e:
            if cfg["app"].get("fallback_rule_if_llm_fails", True):
                # fallback: try to compute a coarse rule-based score from derived features if available
                derived = prompt_ctx.get("context", {}).get("derived") or {}
                rule_score = RuleScorer().predict(derived)
                return {"score": rule_score[0], "rationale": rule_score[2], "contrib": rule_score[3]}
            raise

    # remote tabular (alternative mode)
    elif provider["kind"] == "remote":
        derived = prompt_ctx.get("context", {}).get("derived") or {}
        score, grade, rationale, contribs = RemoteScorer(provider).predict(derived)
        return {"score": score, "rationale": rationale, "contrib": contribs}

    # rule-only mode (not typical for LLM-first, but available)
    else:
        derived = prompt_ctx.get("context", {}).get("derived") or {}
        score, grade, rationale, contribs = RuleScorer().predict(derived)
        return {"score": score, "rationale": rationale, "contrib": contribs}

def compute_grade(score: float) -> str:
    return "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"

def save_score_record(db, customer_id: int, scored: Dict[str, Any]):
    save_score(db, customer_id, scored["score"], compute_grade(scored["score"]), scored.get("rationale",""), model_version=get_app_config()["app"]["model_version"])
