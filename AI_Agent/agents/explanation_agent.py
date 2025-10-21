# agents/explanation_agent.py
import time
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResult, AgentStatus

class ExplanationAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("explanation_agent", config)
        self.explanation_templates = self.config.get("explanation_templates", {})
        self.user_language = self.config.get("user_language", "vi")
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # Extract scoring results
            scoring_result = context.get("scoring_result") if context else {}
            validation_result = context.get("validation_result") if context else {}
            
            # 1. Generate technical explanation
            technical_explanation = await self._generate_technical_explanation(scoring_result)
            
            # 2. Generate user-friendly explanation
            user_explanation = await self._generate_user_explanation(scoring_result, validation_result)
            
            # 3. Generate actionable recommendations
            recommendations = await self._generate_recommendations(scoring_result, validation_result)
            
            # 4. Generate next steps
            next_steps = await self._generate_next_steps(scoring_result, validation_result)
            
            result = AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "technical_explanation": technical_explanation,
                    "user_explanation": user_explanation,
                    "recommendations": recommendations,
                    "next_steps": next_steps,
                    "explanation_confidence": self._calculate_explanation_confidence(scoring_result)
                },
                confidence=self._calculate_explanation_confidence(scoring_result)
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
    
    async def _generate_technical_explanation(self, scoring_result: Dict[str, Any]) -> str:
        """Generate technical explanation for developers/analysts"""
        score = scoring_result.get("score", 0)
        rationale = scoring_result.get("rationale", "")
        contrib = scoring_result.get("contrib", [])
        confidence = scoring_result.get("confidence", 0)
        
        explanation = f"""
**Technical Analysis:**
- Score: {score:.1f}/100 (Confidence: {confidence:.2f})
- Primary Factors: {rationale}
- Feature Contributions: {len(contrib)} features analyzed
"""
        
        if contrib:
            explanation += "\n**Top Contributing Factors:**\n"
            for i, factor in enumerate(contrib[:5], 1):
                name = factor.get("name", "unknown")
                impact = factor.get("impact", 0)
                explanation += f"{i}. {name}: {impact:.2f}\n"
        
        return explanation.strip()
    
    async def _generate_user_explanation(self, scoring_result: Dict[str, Any], validation_result: Dict[str, Any]) -> str:
        """Generate user-friendly explanation"""
        score = scoring_result.get("score", 0)
        grade = scoring_result.get("grade", "D")
        
        if score >= 85:
            explanation = f"🎯 **Khách hàng có tiềm năng rất cao** (Điểm: {score:.0f}/100)\n\n"
            explanation += "Đây là một lead chất lượng cao với khả năng chuyển đổi rất tốt. "
            explanation += "Khuyến nghị ưu tiên liên hệ ngay lập tức."
        elif score >= 70:
            explanation = f"✅ **Khách hàng có tiềm năng tốt** (Điểm: {score:.0f}/100)\n\n"
            explanation += "Lead này có triển vọng tích cực và đáng để theo đuổi. "
            explanation += "Nên liên hệ trong thời gian sớm."
        elif score >= 55:
            explanation = f"⚠️ **Khách hàng có tiềm năng trung bình** (Điểm: {score:.0f}/100)\n\n"
            explanation += "Lead này có một số điểm tích cực nhưng cần đánh giá thêm. "
            explanation += "Có thể theo dõi và liên hệ khi có cơ hội."
        else:
            explanation = f"❌ **Khách hàng có tiềm năng thấp** (Điểm: {score:.0f}/100)\n\n"
            explanation += "Lead này có nhiều rủi ro hoặc thiếu thông tin quan trọng. "
            explanation += "Không nên ưu tiên cao trong danh sách liên hệ."
        
        return explanation
    
    async def _generate_recommendations(self, scoring_result: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        score = scoring_result.get("score", 0)
        contrib = scoring_result.get("contrib", [])
        missing_fields = validation_result.get("missing_fields", [])
        
        # Score-based recommendations
        if score >= 85:
            recommendations.append("🚀 Ưu tiên cao: Liên hệ ngay trong 24h")
            recommendations.append("💼 Chuẩn bị offer đặc biệt cho khách hàng VIP")
        elif score >= 70:
            recommendations.append("📞 Liên hệ trong 48h")
            recommendations.append("📋 Chuẩn bị script phù hợp với profile khách hàng")
        elif score >= 55:
            recommendations.append("⏰ Liên hệ trong 1 tuần")
            recommendations.append("🔍 Thu thập thêm thông tin trước khi tiếp cận")
        else:
            recommendations.append("📝 Đánh giá lại sau khi có thêm dữ liệu")
            recommendations.append("🎯 Tập trung vào các lead có điểm cao hơn")
        
        # Feature-based recommendations
        for factor in contrib[:3]:
            name = factor.get("name", "")
            impact = factor.get("impact", 0)
            
            if "duration" in name.lower() and impact > 0:
                recommendations.append("⏱️ Cuộc gọi dài là điểm tích cực - tiếp tục chiến lược này")
            elif "campaign" in name.lower() and impact < 0:
                recommendations.append("📞 Giảm số lần liên hệ để tránh làm phiền khách hàng")
            elif "balance" in name.lower() and impact > 0:
                recommendations.append("💰 Khách hàng có khả năng tài chính tốt - đề xuất sản phẩm cao cấp")
        
        # Missing data recommendations
        if missing_fields:
            recommendations.append(f"📊 Thu thập thêm thông tin: {', '.join(missing_fields)}")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _generate_next_steps(self, scoring_result: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate next steps"""
        next_steps = []
        score = scoring_result.get("score", 0)
        
        if score >= 70:
            next_steps.append("1. Assign to top sales representative")
            next_steps.append("2. Schedule immediate follow-up call")
            next_steps.append("3. Prepare personalized proposal")
        elif score >= 55:
            next_steps.append("1. Add to weekly follow-up list")
            next_steps.append("2. Send relevant marketing materials")
            next_steps.append("3. Monitor for additional signals")
        else:
            next_steps.append("1. Add to nurture campaign")
            next_steps.append("2. Collect additional data points")
            next_steps.append("3. Re-evaluate in 30 days")
        
        return next_steps
    
    def _calculate_explanation_confidence(self, scoring_result: Dict[str, Any]) -> float:
        """Calculate confidence in explanation"""
        base_confidence = scoring_result.get("confidence", 0.5)
        
        # Increase confidence if we have detailed rationale
        rationale = scoring_result.get("rationale", "")
        if len(rationale) > 50:
            base_confidence += 0.1
        
        # Increase confidence if we have feature contributions
        contrib = scoring_result.get("contrib", [])
        if len(contrib) >= 3:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)