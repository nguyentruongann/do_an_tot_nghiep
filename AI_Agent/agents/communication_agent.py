# agents/communication_agent.py
import time
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResult, AgentStatus

class CommunicationAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("communication_agent", config)
        self.response_templates = self.config.get("response_templates", {})
        self.user_preferences = self.config.get("user_preferences", {})
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # Extract results from all previous agents
            validation_result = context.get("validation_result") if context else {}
            scoring_result = context.get("scoring_result") if context else {}
            explanation_result = context.get("explanation_result") if context else {}
            
            # 1. Generate final response
            final_response = await self._generate_final_response(
                validation_result, scoring_result, explanation_result
            )
            
            # 2. Generate summary
            summary = await self._generate_summary(scoring_result, explanation_result)
            
            # 3. Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                validation_result, scoring_result
            )
            
            result = AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "final_response": final_response,
                    "summary": summary,
                    "follow_up_questions": follow_up_questions,
                    "response_format": "structured",
                    "user_friendly": True
                },
                confidence=0.9  # High confidence in communication
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
    
    # agents/communication_agent.py - Sửa dòng 62-90
    async def _generate_final_response(self, validation_result: Dict[str, Any], 
                                    scoring_result: Dict[str, Any], 
                                    explanation_result: Dict[str, Any]) -> str:
        """Generate final user response"""
        # Sửa lỗi NoneType
        validation_result = validation_result or {}
        scoring_result = scoring_result or {}
        explanation_result = explanation_result or {}
        
        score = scoring_result.get("score", 0)
        grade = scoring_result.get("grade", "D")
        user_explanation = explanation_result.get("user_explanation", "")
        recommendations = explanation_result.get("recommendations", [])
        
        response = f"""
            {user_explanation}

            **📊 Kết quả phân tích:**
            - Điểm số: {score:.1f}/100
            - Xếp hạng: {grade}
            - Độ tin cậy: {scoring_result.get('confidence', 0):.1%}

            **🎯 Khuyến nghị hành động:**
            """
        
        for i, rec in enumerate(recommendations[:3], 1):
            response += f"{i}. {rec}\n"
        
        response += "\n**❓ Câu hỏi tiếp theo:**\n"
        follow_ups = explanation_result.get("follow_up_questions", [])
        for i, q in enumerate(follow_ups[:2], 1):
            response += f"{i}. {q}\n"
        
        return response.strip()
    
    async def _generate_summary(self, scoring_result: Dict[str, Any], explanation_result: Dict[str, Any]) -> str:
        """Generate executive summary"""
        scoring_result = scoring_result or {}
        explanation_result = explanation_result or {}
        
        score = scoring_result.get("score", 0)
        grade = scoring_result.get("grade", "D")
        
        if score >= 85:
            summary = f"Lead chất lượng cao (A) - Ưu tiên liên hệ ngay"
        elif score >= 70:
            summary = f"Lead tiềm năng tốt (B) - Liên hệ trong 48h"
        elif score >= 55:
            summary = f"Lead tiềm năng trung bình (C) - Theo dõi và đánh giá"
        else:
            summary = f"Lead tiềm năng thấp (D) - Cần thu thập thêm dữ liệu"
        
        return summary

    async def _generate_follow_up_questions(self, validation_result: Dict[str, Any], 
                                        scoring_result: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions"""
        validation_result = validation_result or {}
        scoring_result = scoring_result or {}
        
        questions = []
        missing_fields = validation_result.get("missing_fields", [])
        score = scoring_result.get("score", 0)
        
        if score < 70 and missing_fields:
            questions.append(f"Bạn có thể cung cấp thêm thông tin về {', '.join(missing_fields[:2])} không?")
        
        if score >= 70:
            questions.append("Bạn có muốn tôi tạo kế hoạch tiếp cận chi tiết cho lead này không?")
        
        questions.append("Bạn có muốn phân tích thêm các lead khác không?")
        
        return questions