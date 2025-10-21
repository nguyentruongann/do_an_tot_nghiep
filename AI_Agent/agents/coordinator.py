# agents/coordinator.py
import asyncio
import time
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResult, AgentStatus
from .validate_agent import ValidationAgent
from .scoring_agent import ScoringAgent
from .explanation_agent import ExplanationAgent
from .communication_agent import CommunicationAgent

class MultiAgentCoordinator(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("coordinator", config)
        
        config = config or {}  # Thêm dòng này
        self.agents = {
            "validator": ValidationAgent(config.get("validation", {})),
            "scorer": ScoringAgent(config.get("scoring", {})),
            "explainer": ExplanationAgent(config.get("explanation", {})),
            "communicator": CommunicationAgent(config.get("communication", {}))
        }
        
        self.processing_pipeline = [
            "validator",
            "scorer", 
            "explainer",
            "communicator"
        ]
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # Initialize processing context
            processing_context = context or {}
            processing_context["data_type"] = input_data.get("data_type")
            
            # Execute agent pipeline
            agent_results = {}
            
            for agent_name in self.processing_pipeline:
                agent = self.agents[agent_name]
                
                # Process with current agent
                result = await agent.process(input_data, processing_context)
                
                if result.status == AgentStatus.ERROR:
                    # Handle agent failure
                    return await self._handle_agent_failure(agent_name, result, processing_context)
                
                # Store result for next agents
                agent_results[f"{agent_name}_result"] = result.data
                processing_context[f"{agent_name}_result"] = result.data
                
                # Add agent metadata
                processing_context[f"{agent_name}_metadata"] = {
                    "status": result.status.value,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time
                }
            
            # Compile final result
            final_result = await self._compile_final_result(agent_results, processing_context)
            
            result = AgentResult(
                status=AgentStatus.SUCCESS,
                data=final_result,
                confidence=self._calculate_overall_confidence(agent_results)
            )
            
            self.status = AgentStatus.SUCCESS
            self.update_metrics(result, time.time() - start_time)
            return result
            
        except Exception as e:
            result = AgentResult(
                status=AgentStatus.ERROR,
                data={"error": str(e), "pipeline_stage": "coordinator"},
                error=str(e)
            )
            self.status = AgentStatus.ERROR
            self.update_metrics(result, time.time() - start_time)
            return result
    
    async def _handle_agent_failure(self, failed_agent: str, error_result: AgentResult, 
                                  context: Dict[str, Any]) -> AgentResult:
        """Handle agent failure with fallback strategies"""
        
        if failed_agent == "validator":
            # Validation failure - return error immediately
            return AgentResult(
                status=AgentStatus.ERROR,
                data={
                    "error": "Data validation failed",
                    "details": error_result.data,
                    "pipeline_stage": "validation"
                },
                error=error_result.error
            )
        
        elif failed_agent == "scorer":
            # Scoring failure - try fallback scoring
            try:
                from services.score_service import llm_score_json, compute_grade
                fallback_result = llm_score_json({"prompt": "Fallback scoring"})
                fallback_result["grade"] = compute_grade(fallback_result["score"])
                
                # Continue with explainer using fallback score
                context["scoring_result"] = fallback_result
                explainer = self.agents["explainer"]
                explainer_result = await explainer.process({}, context)
                
                if explainer_result.status == AgentStatus.SUCCESS:
                    context["explanation_result"] = explainer_result.data
                    communicator = self.agents["communicator"]
                    comm_result = await communicator.process({}, context)
                    
                    return AgentResult(
                        status=AgentStatus.SUCCESS,
                        data={
                            **comm_result.data,
                            "fallback_used": True,
                            "fallback_agent": "scorer"
                        },
                        confidence=0.6  # Lower confidence due to fallback
                    )
            except Exception as e:
                pass
        
        # If all fallbacks fail, return error
        return AgentResult(
            status=AgentStatus.ERROR,
            data={
                "error": f"Agent {failed_agent} failed and no fallback available",
                "details": error_result.data,
                "pipeline_stage": failed_agent
            },
            error=error_result.error
        )
    
    async def _compile_final_result(self, agent_results: Dict[str, Any], 
                                  processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile results from all agents into final response"""
        
        # Extract key results
        validation_result = agent_results.get("validator_result", {})
        scoring_result = agent_results.get("scorer_result", {})
        explanation_result = agent_results.get("explainer_result", {})
        communication_result = agent_results.get("communicator_result", {})
        
        # Compile comprehensive result
        final_result = {
            # Core scoring data
            "score": scoring_result.get("score", 0),
            "grade": scoring_result.get("grade", "D"),
            "confidence": scoring_result.get("confidence", 0),
            
            # Validation insights
            "data_quality": validation_result.get("quality_score", 0),
            "missing_fields": validation_result.get("missing_fields", []),
            
            # Explanations
            "technical_explanation": explanation_result.get("technical_explanation", ""),
            "user_explanation": explanation_result.get("user_explanation", ""),
            "recommendations": explanation_result.get("recommendations", []),
            "next_steps": explanation_result.get("next_steps", []),
            
            # Communication
            "final_response": communication_result.get("final_response", ""),
            "summary": communication_result.get("summary", ""),
            "follow_up_questions": communication_result.get("follow_up_questions", []),
            
            # Metadata
            "processing_metadata": {
                "pipeline_version": "multi_agent_v1",
                "agents_used": list(self.agents.keys()),
                "processing_time": processing_context.get("total_processing_time", 0),
                "data_type": processing_context.get("data_type", "unknown")
            }
        }
        
        return final_result
    
    def _calculate_overall_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall confidence based on all agent results"""
        confidences = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])
        
        if not confidences:
            return 0.5
        
        # Weighted average confidence
        return sum(confidences) / len(confidences)
    
    def get_agent_health_status(self) -> Dict[str, Any]:
        """Get health status of all agents"""
        health_status = {
            "coordinator": self.get_health_status(),
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            health_status["agents"][name] = agent.get_health_status()
        
        return health_status