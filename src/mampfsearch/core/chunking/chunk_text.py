"""Plain text chunking."""
import logging
from pathlib import Path
from typing import List, Union

from spacy.lang.en import English

from mampfsearch.utils.models import Chunk, FileLocation, VideoLocation

logger = logging.getLogger(__name__)


def chunk_text_by_sentences(
    text: str,
    location: FileLocation,
    max_sentences_per_chunk: int = 5,
) -> List[Chunk]:
    """
    Chunk plain text into groups of sentences.
    
    Args:
        text: The text to chunk
        max_sentences_per_chunk: Maximum sentences per chunk
        location: Optional location metadata for all chunks
        
    Returns:
        List of Chunk objects
    """
    logger.debug(f"Chunking text ({len(text)} chars) with {max_sentences_per_chunk} sentences/chunk")
    
    nlp = English()
    nlp.add_pipe("sentencizer")
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk_sentences = sentences[i:i + max_sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        
        chunk = Chunk(text=chunk_text, location=location)
        chunks.append(chunk)
    
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def chunk_text_file(
    file_path: Path,
    course_id: str,
    max_sentences_per_chunk: int = 5,
) -> List[Chunk]:
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
    
    text = file_path.read_text(encoding='utf-8')
    
    location = FileLocation(
        courseId=course_id,
        fileId=file_path.stem
    )
    
    return chunk_text_by_sentences(text, location, max_sentences_per_chunk)