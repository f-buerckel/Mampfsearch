import logging

from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional, Union
from datetime import timedelta
from pathlib import Path

class TranscriptChunk(BaseModel):
    text: str
    lecture_name: str
    lecture_position: int
    position: int
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None

class TranscriptionRequest(BaseModel):
    audio_file: Path

class IngestRequest(BaseModel):
    srt_file : Path
    lecture_name: str
    lecture_position: int = 0
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

class LectureRetrievalItem(BaseModel):
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

class VideoLocation(BaseModel):
    courseId: str
    fileId : str
    lectureId: str
    timestamp: str

class FileLocation(BaseModel):
    courseId: str
    fileId : str

class EntityCandidate(BaseModel):
    """ 
    Entity candidates are single extracted entities from a document that may or may not already be in the knowledge base. 
    Entities are already in the knowledge base and contain a unique identifier along every occurrence of the entity across all documents.
    """
    text: str
    label: str
    Location: Union[VideoLocation, FileLocation, None] = None

class Entity(BaseModel):
    name: str
    label: str
    entity_instances: Optional[List[EntityCandidate]] = []

    @classmethod
    def from_entity_candidate(cls, entity_candidate):
        return cls(
            name = entity_candidate.text.lower(),
            label = entity_candidate.label,
            entity_instances = [entity_candidate],
        )


class EntityRetrievalItem(BaseModel):
    id : str
    score : float
    entity : Entity

    @classmethod
    def from_qdrant_point(cls, point):
        return cls(
            id =str(point.id),
            score=float(point.score),
            entity=Entity(**point.payload)
        )



class Response(BaseModel):
    answer: str
    confidence_score: float
    source_snippets: Dict[str, float]

class AskRequest(BaseModel):
    question: str
    retriever_type: RetrieverTypeEnum = RetrieverTypeEnum.hybrid
    limit: int = 5

class SearchResult(BaseModel):
    items: List[LectureRetrievalItem]

class Answer(BaseModel):
    answer: str
    confidence_score: float
    source_snippets: Dict[str, float]

class ExtractionInfo(BaseModel):
    num_extracted_entities: int
    num_new_inserted_entities: int
    num_merged_entities: int