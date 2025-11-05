import spacy
import logging

from pathlib import Path
from langdetect import detect
from typing import Optional

from spacy.tokens import Doc

from mampfsearch.core.chunking import chunk_file
from mampfsearch.core import named_entity_recognition
from mampfsearch.core import entity_linking

logger = logging.getLogger(__name__)

if not Doc.has_extension("location"):
    Doc.set_extension("location", default=None)

def extract(
    file_path: Path,
    course_id: str,
    lecture_id: Optional[str] = None,
    print_chunks: bool = False
):

    chunks = chunk_file(
        file_path=file_path,
        course_id=course_id,
        lecture_id=lecture_id,
    )
    
    language = detect(" ".join([chunk.text for chunk in chunks[0:2]]))
    logger.info(f"Detected language: {language}")

    if language == 'de':
        nlp = spacy.blank("de")
    else:
        nlp = spacy.blank("en")

    nlp.add_pipe("llm_ner")
    nlp.add_pipe("embedding_entity_linker")
    
    docs = []
    for chunk in chunks:
        doc = nlp.make_doc(chunk.text)
        doc._.location = chunk.location
        docs.append(doc)
    
    # Proccess chunks in batches like this is usually more efficient, see: https://spacy.io/usage/processing-pipelines#processing
    final_docs = list(nlp.pipe(docs))