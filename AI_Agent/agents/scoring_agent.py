# agents/scoring_agent.py
import time
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentResult, AgentStatus
from services.prompt_manager import build_prompt_for
from services.score_service import llm_score_json, compute_grade

class ScoringAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("scoring_agent", config)
        self.scoring_strategies = self.config.get("scoring_strategies", ["llm", "rule", "hybrid"])
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # Extract context from previous agents
            data_type = context.get("data_type") if context else None
            validation_result = context.get("validation_result") if context else None
            
            # 1. Build enhanced prompt context
            prompt_context = await self._build_enhanced_context(input_data, data_type, validation_result)
            
            # 2. Perform scoring with multiple strategies
            scoring_results = await self._multi_strategy_scoring(prompt_context)
            
            # 3. Ensemble scoring if multiple strategies available
            final_score = await self._ensemble_scoring(scoring_results)
            
            # 4. Calculate confidence and grade
            confidence = self._calculate_confidence(final_score, scoring_results)
            grade = compute_grade(final_score["score"])
            
            result = AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "score": final_score["score"],
                    "grade": grade,
                    "rationale": final_score["rationale"],
                    "contrib": final_score["contrib"],
                    "confidence": confidence,
                    "scoring_strategies_used": list(scoring_results.keys()),
                    "ensemble_details": scoring_results
                },
                confidence=confidence
            )
            
            self.status = AgentStatus.SUCCESS
            self.update_metrics(result, time.time() - start_time)
            return result
            
        except Exception as e:
            result = AgentResult(
                status=AgentStatus.ERROR,
                data={"error": str(e)},
                error=str(e)
            )
            self.status = AgentStatus.ERROR
            self.update_metrics(result, time.time() - start_time)
            return result
    
    async def _build_enhanced_context(self, input_data: Dict[str, Any], data_type: Optional[str], validation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build enhanced context for scoring"""
        # Use existing prompt manager
        base_context = build_prompt_for(data_type=data_type, payload=input_data)
        
        # Enhance with validation insights
        if validation_result:
            base_context["validation_insights"] = {
                "quality_score": validation_result.get("quality_score", 0),
                "missing_fields": validation_result.get("missing_fields", []),
                "data_completeness": 1.0 - (len(validation_result.get("missing_fields", [])) / 10.0)
            }
        
        # Add scoring metadata
        base_context["scoring_metadata"] = {
            "timestamp": time.time(),
            "agent_version": "scoring_agent_v1",
            "strategies_enabled": self.scoring_strategies
        }
        
        return base_context
    
    async def _multi_strategy_scoring(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Perform scoring using multiple strategies"""
        results = {}
        
        # LLM Strategy
        if "llm" in self.scoring_strategies:
            try:
                llm_result = llm_score_json(context)
                results["llm"] = {
                    "score": llm_result["score"],
                    "rationale": llm_result["rationale"],
                    "contrib": llm_result["contrib"],
                    "strategy": "llm",
                    "confidence": 0.8  # LLM confidence
                }
            except Exception as e:
                results["llm"] = {"error": str(e), "strategy": "llm"}
        
        # Rule-based Strategy
        if "rule" in self.scoring_strategies:
            try:
                from models.scorer import RuleScorer
                derived = context.get("context", {}).get("derived", {})
                rule_score, grade, rationale, contribs = RuleScorer().predict(derived)
                results["rule"] = {
                    "score": rule_score,
                    "rationale": rationale,
                    "contrib": contribs,
                    "strategy": "rule",
                    "confidence": 0.6  # Rule-based confidence
                }
            except Exception as e:
                results["rule"] = {"error": str(e), "strategy": "rule"}
        
        return results
    
    async def _ensemble_scoring(self, scoring_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple scoring strategies"""
        valid_results = {k: v for k, v in scoring_results.items() if "error" not in v}
        
        if not valid_results:
            raise Exception("All scoring strategies failed")
        
        if len(valid_results) == 1:
            return list(valid_results.values())[0]
        
        # Weighted ensemble
        weights = {"llm": 0.7, "rule": 0.3}  # LLM gets higher weight
        
        ensemble_score = 0.0
        ensemble_rationale = []
        ensemble_contrib = {}
        
        total_weight = 0.0
        for strategy, result in valid_results.items():
            weight = weights.get(strategy, 0.5)
            ensemble_score += result["score"] * weight
            ensemble_rationale.append(f"{strategy}: {result['rationale']}")
            
            # Combine contributions
            if "contrib" in result:
                for contrib in result["contrib"]:
                    name = contrib.get("name", "unknown")
                    impact = contrib.get("impact", 0)
                    if name in ensemble_contrib:
                        ensemble_contrib[name] += impact * weight
                    else:
                        ensemble_contrib[name] = impact * weight
            
            total_weight += weight
        
        ensemble_score = ensemble_score / total_weight if total_weight > 0 else ensemble_score
        
        return {
            "score": ensemble_score,
            "rationale": " | ".join(ensemble_rationale),
            "contrib": [{"name": k, "impact": v} for k, v in ensemble_contrib.items()]
        }
    
    def _calculate_confidence(self, final_score: Dict[str, Any], all_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate confidence based on agreement between strategies"""
        valid_results = {k: v for k, v in all_results.items() if "error" not in v}
        
        if len(valid_results) <= 1:
            return 0.5
        
        # Calculate variance in scores
        scores = [r["score"] for r in valid_results.values()]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher confidence
        confidence = max(0.1, 1.0 - (variance / 1000.0))
        return min(1.0, confidence)