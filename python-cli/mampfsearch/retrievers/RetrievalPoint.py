from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class RetrievalPoint:
    """Immutable representation of a single retrieval result."""
    score: float
    text: str
    lecture: str
    lecture_position: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    position: Optional[int] = None
    
    @classmethod
    def from_qdrant_point(cls, point):
        
        return cls(
            score               = float(point.score),
            text                = str(point.payload["text"]),
            lecture             = str(point.payload["lecture_name"]),
            lecture_position    = str(point.payload["lecture_position"]),
            start_time          = str(point.payload["start_time"]),
            end_time            = str(point.payload["end_time"]),
            position            = int(point.payload["position"]),
        )