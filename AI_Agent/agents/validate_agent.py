# agents/validation_agent.py
import time
from typing import Dict, Any, List, Tuple, Optional
from .base_agent import BaseAgent, AgentResult, AgentStatus
from services.validator import validate_minimal 


class ValidationAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("validation_agent", config)
        self.validation_rules = self.config.get("validation_rules", {})
    
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # 1. Basic structure validation
            payload = input_data.get("payload", {})
            basic_valid, basic_errors = validate_minimal(payload)
            
            if not basic_valid:
                result = AgentResult(
                    status=AgentStatus.ERROR,
                    data={"errors": basic_errors, "stage": "basic_validation"},
                    error=f"Basic validation failed: {basic_errors}"
                )
                self.update_metrics(result, time.time() - start_time)
                return result
            
            # 2. Business rule validation
            business_valid, business_errors = await self._validate_business_rules(input_data)
            if not business_valid:
                result = AgentResult(
                    status=AgentStatus.ERROR,
                    data={"errors": business_errors, "stage": "business_validation"},
                    error=f"Business validation failed: {business_errors}"
                )
                self.update_metrics(result, time.time() - start_time)
                return result
            
            # 3. Data quality assessment
            quality_score = await self._assess_data_quality(input_data)
            
            # 4. Missing field detection
            missing_fields = await self._detect_missing_fields(input_data)
            
            result = AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "is_valid": True,
                    "quality_score": quality_score,
                    "missing_fields": missing_fields,
                    "validation_stage": "complete"
                },
                confidence=quality_score / 100.0
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
    
    async def _validate_business_rules(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Advanced business rule validation"""
        errors = []
        
        # Age validation
        age = data.get("age")
        if age is not None:
            if not isinstance(age, (int, float)) or age < 0 or age > 120:
                errors.append("invalid_age_range")
        
        # Duration validation
        duration = data.get("duration")
        if duration is not None:
            if not isinstance(duration, (int, float)) or duration < 0:
                errors.append("invalid_duration")
        
        # Campaign validation
        campaign = data.get("campaign")
        if campaign is not None:
            if not isinstance(campaign, (int, float)) or campaign < 0:
                errors.append("invalid_campaign_count")
        
        # Balance validation
        balance = data.get("balance")
        if balance is not None:
            if not isinstance(balance, (int, float)):
                errors.append("invalid_balance_type")
        
        return len(errors) == 0, errors
    
    async def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess data quality score (0-100)"""
        score = 100.0
        total_fields = len(data)
        
        if total_fields == 0:
            return 0.0
        
        # Check for null/empty values
        null_count = sum(1 for v in data.values() if v is None or v == "")
        score -= (null_count / total_fields) * 30
        
        # Check for reasonable value ranges
        if data.get("age") and (data["age"] < 18 or data["age"] > 100):
            score -= 20
        
        if data.get("duration") and data["duration"] > 10000:  # Unrealistic call duration
            score -= 15
        
        return max(0.0, score)
    
    async def _detect_missing_fields(self, data: Dict[str, Any]) -> List[str]:
        """Detect important missing fields"""
        important_fields = ["age", "job", "balance", "duration", "campaign"]
        missing = []
        
        for field in important_fields:
            if field not in data or data[field] is None:
                missing.append(field)
        
        return missing