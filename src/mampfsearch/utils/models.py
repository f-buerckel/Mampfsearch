import logging

from pydantic import BaseModel, field_serializer
from enum import Enum
from typing import List, Dict, Optional, Union
from datetime import timedelta
from pathlib import Path

class VideoLocation(BaseModel):
    courseId: str
    lectureId: str
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None

    # format timestamp readable when using model_dump.
    # https://docs.pydantic.dev/latest/concepts/serialization/#using-the-annotated-pattern
    @field_serializer('start_time', 'end_time')
    def serialize_timedelta(self, td: Optional[timedelta], _info) -> Optional[str]:
        """Convert timedelta to HH:MM:SS format."""
        if td is None:
            return None
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class FileLocation(BaseModel):
    courseId: str
    fileId : str

class Chunk(BaseModel):
    text: str
    location: Union[VideoLocation, FileLocation, None] = None

class TranscriptionRequest(BaseModel):
    audio_file: Path

class IngestRequest(BaseModel):
    srt_file : Path
    course_id: str
    lecture_id: str
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
    video_location: Optional[VideoLocation] = None

    @classmethod
    def from_qdrant_point(cls, point):
        return cls(
            score=float(point.score),
            text=str(point.payload["text"]),
            video_location=VideoLocation(
                courseId=point.payload["course_id"],
                lectureId=point.payload["lecture_id"],
                start_time=str(point.payload["start_time"]),
                end_time=str(point.payload["end_time"]),
            ) if "course_id" in point.payload and "lecture_id" in point.payload else None
        )


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