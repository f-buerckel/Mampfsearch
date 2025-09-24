from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional
from datetime import timedelta
from pathlib import Path

class Chunk(BaseModel):
    text: str
    lecture_name: str
    lecture_position: int
    position: int
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None

class TranscriptionRequest(BaseModel):
    lecture_name: str
    lecture_position: int = 0
    audio_filename: str
    whisper_model: str = "base"

class IngestRequest(BaseModel):
    srt_file : Path
    lecture_name: str
    lecture_position: int = 0
    collection_name: str = "Lectures"
    min_chunk_size: int = 350
    max_chunk_size: int = 850
    overlap: bool = True

class RetrieverTypeEnum(str, Enum):
    dense = "dense"
    hybrid = "hybrid"
    hybrid_colbert = "hybrid+colbert"

class SearchRequest(BaseModel):
    query: str
    retriever_type: RetrieverTypeEnum = RetrieverTypeEnum.hybrid  # dense | hybrid | hybrid+colbert
    limit: int = 5
    reranking: bool = False
    collection_name: str = "Lectures"

class RetrievalItem(BaseModel):
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
            score=float(point.score),
            text=str(point.payload["text"]),
            lecture=str(point.payload["lecture_name"]),
            lecture_position=str(point.payload["lecture_position"]),
            start_time=str(point.payload["start_time"]),
            end_time=str(point.payload["end_time"]),
            position=int(point.payload["position"]),
        )

class AskRequest(BaseModel):
    question: str
    retriever_type: str = "hybrid"
    limit: int = 5
    model: str = "gemma3n:e4b"
    temperature: float = 0.5
    collection_name: str = "Lectures"

class SearchResult(BaseModel):
    items: List[RetrievalItem]

class Answer(BaseModel):
    answer: str
    confidence_score: float
    source_snippets: Dict[str, float]