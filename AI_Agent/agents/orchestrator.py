# agents/orchestrator.py (Updated)
from typing import Dict, Any, List, Optional
from .coordinator import MultiAgentCoordinator
from database.repositories import fetch_bank_rows_all, fetch_bank_rows_by_ids
from database.session import Session

class Orchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.coordinator = MultiAgentCoordinator(config)
        self.config = config or {}
    
    async def handle_score(self, db: Session, data_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single lead scoring with multi-agent pipeline"""
        try:
            # Prepare input data
            input_data = {
                "data_type": data_type,
                "payload": payload
            }
            
            # Process through multi-agent pipeline
            result = await self.coordinator.process(input_data)
            
            if result.status.value == "error":
                return {
                    "status": "error",
                    "error": result.error,
                    "details": result.data
                }
            
            # Save to database
            customer_id = payload.get("ID", payload.get("id", -1))
            await self._save_score_record(db, customer_id, result.data)
            
            return {
                "status": "success",
                "customer_id": customer_id,
                "data": result.data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": {"pipeline_stage": "orchestrator"}
            }
    
    async def handle_score_batch(self, db: Session, data_type: str, limit: Optional[int] = 100) -> List[Dict[str, Any]]:
        """Handle batch scoring with multi-agent pipeline"""
        if data_type == "bank_marketing":
            rows = fetch_bank_rows_all(db, limit=limit or 100)
        else:
            raise ValueError(f"Batch not supported for data_type: {data_type}")
        
        results = []
        for i, row in enumerate(rows):
            try:
                result = await self.handle_score(db, data_type=data_type, payload=row)
                results.append(result)
                
                # Add progress logging
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(rows)} records")
                    
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "customer_id": row.get("ID", row.get("id", -1))
                })
        
        return results
    
    async def _save_score_record(self, db: Session, customer_id: int, scored_data: Dict[str, Any]):
        """Save scoring result to database"""
        from database.repositories import save_score
        from models.registry import get_app_config
        
        score = scored_data.get("score", 0)
        grade = scored_data.get("grade", "D")
        rationale = scored_data.get("technical_explanation", "")
        model_version = get_app_config()["app"]["model_version"]
        
        save_score(db, customer_id, score, grade, rationale, model_version)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return self.coordinator.get_agent_health_status()