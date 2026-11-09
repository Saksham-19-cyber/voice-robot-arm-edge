"""NLU data schemas and result types."""



from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class IntentResult:
    intent: str
    target: str
    goal_xyz: Optional[List[float]] = None
    relative_offset: Optional[List[float]] = None
    confidence: float = 1.0
    clarification_needed: bool = False
    message: str = ""
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "target": self.target,
            "goal_xyz": self.goal_xyz,
            "relative_offset": self.relative_offset,
            "confidence": round(self.confidence, 4),
            "clarification_needed": self.clarification_needed,
            "message": self.message,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }
