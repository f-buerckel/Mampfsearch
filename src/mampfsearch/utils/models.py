from pydantic import BaseModel, field_serializer, ConfigDict, Field
from enum import Enum
from typing import List, Dict, Optional, Set
from datetime import timedelta
from pathlib import Path
from mampfsearch.utils.schema import nodeLabels

from gqlalchemy import Node, Relationship


from spacy.tokens import Span, Doc


class VideoLocation(BaseModel):
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None

    # format timestamp readable when using model_dump.
    # https://docs.pydantic.dev/latest/concepts/serialization/#using-the-annotated-pattern
    @field_serializer("start_time", "end_time")
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
    filename: str
    page_number: Optional[int] = None
    word_start: Optional[int] = None
    word_end: Optional[int] = None


class Segment(Node):
    text: str
    location: VideoLocation
    position: int

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["segment"]


class Passage(Node):
    text: str
    location: FileLocation
    position: int

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["passage"]


class Course(Node):
    name: str
    description: Optional[str] = None
    instructor: Optional[str] = None

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["course"]


class Lecture(Node):
    name: str
    position: Optional[int] = None
    description: Optional[str] = None
    upload_date: Optional[str] = None

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["lecture"]


class HasLecture(Relationship, type="HAS_LECTURE"):
    pass


class PdfFile(Node):
    filename: str
    upload_date: Optional[str] = None
    description: Optional[str] = None

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["pdf_file"]


class Topic(Node):
    name: str
    uri: Optional[str] = None
    description: Optional[str] = None
    wikipedia_url: Optional[str] = None

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["topic"]


class MathEntityCandidate(BaseModel):
    """
    Entity candidates are single extracted entities from a document that may or may not already be in the knowledge base.
    Entities are already in the knowledge base and contain a unique identifier along every occurrence of the entity across all documents.
    """

    text: str
    label: str


class MathEntity(Node):
    name: str
    uri: Optional[str] = None
    description: Optional[str] = None
    formula: Optional[str] = None
    wikipedia_url: Optional[str] = None
    entity_instances: Optional[List[MathEntityCandidate]] = []
    # number of mentions / total mentions in the kg
    global_density: Optional[float] = None

    @classmethod
    def from_entity_candidate(cls, entity_candidate):
        return cls(
            name=entity_candidate.text.lower(),
            label=entity_candidate.label,
            entity_instances=[entity_candidate],
        )

    @classmethod
    def get_identifying_label(self) -> str:
        return nodeLabels["math_entity"]


class TranscriptionRequest(BaseModel):
    audio_file: Path


class IngestRequest(BaseModel):
    srt_file: Path
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
    retriever_type: RetrieverTypeEnum = (
        RetrieverTypeEnum.hybrid
    )  # dense | hybrid | hybrid+colbert
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
            )
            if "course_id" in point.payload and "lecture_id" in point.payload
            else None,
        )


class EntityRetrievalItem(BaseModel):
    id: str
    score: float
    entity: MathEntity

    @classmethod
    def from_qdrant_point(cls, point):
        entity = MathEntity(**point.payload)
        # If the entity has a URI (from Wikidata), use it as the ID.
        # Otherwise use the Qdrant point ID (UUID) as original inserted entities use qdrant uuid in graph storage
        entity_id = entity.uri if entity.uri else str(point.id)
        return cls(id=entity_id, score=float(point.score), entity=entity)


class RelationshipCandidate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_1: Span
    entity_2: Span
    context: Doc


class RelationshipReturn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_1: Span
    entity_2: Span
    relationship: str
    reasoning: str
    context: Doc


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
