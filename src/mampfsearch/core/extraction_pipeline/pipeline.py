import spacy
import logging

from pathlib import Path
from langdetect import detect
from typing import Optional

from spacy.tokens import Doc

from mampfsearch.core.chunking import chunk_file
from mampfsearch.utils import config
from mampfsearch.utils.models import Lecture, Course, Segment

# import declared as redundant so that linter does not "fix" unused imports
# needed so that spaCy registers the custom components
from mampfsearch.core import named_entity_recognition as named_entity_recognition
from mampfsearch.core import linking as linking
from mampfsearch.core import relationship_extraction as relationship_extraction

logger = logging.getLogger(__name__)

if not Doc.has_extension("course"):
    Doc.set_extension("course", default=None)

if not Doc.has_extension("lecture"):
    Doc.set_extension("lecture", default=None)

if not Doc.has_extension("segment"):
    Doc.set_extension("segment", default=None)


def extract(
    file_path: Path,
    course_name: str,
    lecture_name: str,
    lecture_position: int,
    lecture_description: Optional[str] = None,
):
    # add course node if not exists
    # add lecture node if not exists
    graph_storage = config.get_graph_storage()

    lectureNode = graph_storage.get_lecture_node(name=lecture_name)
    if lectureNode:
        logger.info(
            f"Lecture node already exists for file: {file_path.name}, skipping extraction."
        )
        return

    courseNode = graph_storage.get_course_node(name=course_name)
    if courseNode is None:
        course = Course(name=course_name)
        graph_storage.add_course_node(course)

    lecture = Lecture(
        name=lecture_name,
        position=lecture_position,
        description=lecture_description,
    )

    graph_storage.add_lecture_node(lecture=lecture, courseNode=courseNode)

    chunks = chunk_file(
        file_path=file_path,
    )

    language = detect(" ".join([chunk.text for chunk in chunks[0:2]]))
    logger.info(f"Detected language: {language}")

    if language == "de":
        nlp = spacy.blank("de")
    else:
        nlp = spacy.blank("en")

    nlp.add_pipe("sentencizer")
    nlp.add_pipe("llm_ner_v2")
    nlp.add_pipe("llm_ner_validation")
    nlp.add_pipe("embedding_entity_linker")
    nlp.add_pipe("llm_relationship_extraction")
    nlp.add_pipe("llm_relationship_validation")
    nlp.add_pipe("simple_relationship_linker")

    logger.debug(f"Pipeline problems: {nlp.analyze_pipes()['problems']}")

    docs = []
    for i, chunk in enumerate(chunks):
        segment = Segment(
            text=chunk.text,
            location=chunk.location,
            position=i,
        )
        segmentNode = graph_storage.add_segment_node(
            segment=segment, lectureNode=lectureNode
        )
        doc = nlp.make_doc(chunk.text)
        doc._.course = courseNode
        doc._.lecture = lectureNode
        doc._.segment = segmentNode
        docs.append(doc)

    # Proccess chunks in batches like this is usually more efficient, see: https://spacy.io/usage/processing-pipelines#processing
    list(nlp.pipe(docs))

    # update global mention ratios
    graph_storage.update_global_mention_ratio()

    logger.info(f"Extraction pipeline completed for file: {file_path.name}")
