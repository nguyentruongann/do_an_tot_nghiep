from typing import Dict, Any, List, Optional
from services.validator import validate_minimal
from services.prompt_manager import build_prompt_for
from services.score_service import llm_score_json, compute_grade, save_score_record
from database.repositories import fetch_bank_rows_all, fetch_bank_rows_by_ids

class Orchestrator:
    def handle_score(self, db, data_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1) minimal checks
        ok, errors = validate_minimal(payload)
        if not ok:
            return {"status": "need_more", "errors": errors}

        # 2) build prompt from adapter + prompt pack
        ctx = build_prompt_for(data_type=data_type, payload=payload)
        # 3) call LLM to get structured JSON
        scored = llm_score_json(ctx)
        scored["grade"] = compute_grade(scored["score"])
        # 4) persist
        customer_id = payload.get("ID", payload.get("id", -1))
        save_score_record(db, customer_id=customer_id, scored=scored)
        return {"customer_id": customer_id, **scored}

    def handle_score_batch(self, db, data_type: str, limit: Optional[int] = 100):
        if data_type == "bank_marketing":
            rows = fetch_bank_rows_all(db, limit=limit or 100)
        else:
            raise ValueError(f"Batch not supported for data_type: {data_type}")
        results = []
        for row in rows:
            res = self.handle_score(db, data_type=data_type, payload=row)
            results.append(res)
        return results
