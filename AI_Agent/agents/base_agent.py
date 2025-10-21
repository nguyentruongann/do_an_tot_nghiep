# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class AgentResult:
    status: AgentStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0

class BaseAgent(ABC):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.metrics = {
            "total_requests": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_processing_time": 0.0
        }
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Main processing method for each agent"""
        pass
    
    def update_metrics(self, result: AgentResult, processing_time: float):
        """Update agent metrics"""
        self.metrics["total_requests"] += 1
        if result.status == AgentStatus.SUCCESS:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1
        
        # Update average processing time
        total_time = self.metrics["avg_processing_time"] * (self.metrics["total_requests"] - 1)
        self.metrics["avg_processing_time"] = (total_time + processing_time) / self.metrics["total_requests"]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        success_rate = self.metrics["success_count"] / max(1, self.metrics["total_requests"])
        return {
            "name": self.name,
            "status": self.status.value,
            "success_rate": success_rate,
            "metrics": self.metrics
        }