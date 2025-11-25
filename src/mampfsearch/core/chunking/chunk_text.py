"""Plain text chunking."""

import logging
from pathlib import Path
from typing import List

from spacy.lang.en import English

from mampfsearch.utils.models import Passage, FileLocation

logger = logging.getLogger(__name__)


def chunk_text_by_sentences(
    text: str,
    max_sentences_per_chunk: int = 5,
) -> List[Passage]:
    """
    Chunk plain text into groups of sentences.

    Args:
        text: The text to chunk
        max_sentences_per_chunk: Maximum sentences per chunk

    Returns:
        List of Chunk objects
    """
    logger.debug(
        f"Chunking text ({len(text)} chars) with {max_sentences_per_chunk} sentences/chunk"
    )

    nlp = English()
    nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk_sentences = sentences[i : i + max_sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)

        chunk = Passage(text=chunk_text)
        chunks.append(chunk)

    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def chunk_text_file(
    file_path: Path,
    max_sentences_per_chunk: int = 5,
) -> List[Passage]:
    """
    Read and chunk a plain text file.

    Args:
        file_path: Path to the .txt file
        max_sentences_per_chunk: Maximum sentences per chunk
        course_id: Course identifier for metadata

    Returns:
        List of Chunk objects with FileLocation metadata
    """
    logger.info(f"Chunking text file: {file_path.name}")

    text = file_path.read_text(encoding="utf-8")

    return chunk_text_by_sentences(text, max_sentences_per_chunk)
