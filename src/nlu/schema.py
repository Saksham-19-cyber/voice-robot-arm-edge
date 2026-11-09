"""
NLU Schema definition for voice-commanded robot manipulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class IntentResult:
    intent: str  # reach | pick | move | stop | unknown
    target: str  # region or object description (e.g., 'left', 'upper right', 'block')
    goal_xyz: Optional[List[float]] = None  # [x, y, z] target in physical workspace
    relative_offset: Optional[List[float]] = None  # [dx, dy, dz] in meters
    confidence: float = 1.0  # [0.0, 1.0]
    clarification_needed: bool = False
    message: str = ""  # Natural language confirmation/clarification
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

