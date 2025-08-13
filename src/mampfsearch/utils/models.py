from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import timedelta

class Chunk(BaseModel):
    text: str
    lecture_name: str
    lecture_position: int
    position: int
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None

class IngestRequest(BaseModel):
    lecture_name: str
    lecture_position: int = 0
    audio_filename: str
    whisper_model: str = "base"
    min_chunk_size: int = 350
    overlap: bool = True
    collection_name: str = "Lectures"

class SearchRequest(BaseModel):
    query: str
    retriever_type: str = "hybrid"  # dense | hybrid | hybrid+colbert
    limit: int = 5
    reranking: bool = False
    collection_name: str = "Lectures"

class AskRequest(BaseModel):
    question: str
    retriever_type: str = "hybrid"
    limit: int = 5
    model: str = "gemma3n:e4b"
    temperature: float = 0.5
    collection_name: str = "Lectures"

class RetrievalItem(BaseModel):
    score: float
    text: str
    lecture: str
    lecture_position: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    position: Optional[int] = None

class SearchResult(BaseModel):
    items: List[RetrievalItem]

class Answer(BaseModel):
    answer: str
    confidence_score: float
    source_snippets: Dict[str, float]