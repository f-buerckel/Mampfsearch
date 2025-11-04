from pathlib import Path
from typing import List, Optional

from mampfsearch.utils.models import Chunk
from mampfsearch.core.chunking import chunk_text_file, chunk_pdf_file, chunk_srt_file

def chunk_file(
    file_path: Path,
    course_id: str,
    lecture_id: Optional[str] = None,
    min_chunk_size: int = 40,
    max_chunk_size: int = 400, # the high max_chunk_size encouages to keep sentences in one chunk.
) -> List[Chunk]:
    
    chunks = []
    if file_path.suffix == ".txt":
        max_sentences_per_chunk = 3
        chunks = chunk_text_file(
            file_path=file_path,
            course_id=course_id,
            max_sentences_per_chunk=max_sentences_per_chunk,
        )

    elif file_path.suffix == ".pdf":
        chunks = chunk_pdf_file(
            pdf_file_path=file_path,
            course_id=course_id,
            enable_formula_enrichment=False
        )
    
    elif file_path.suffix == ".srt":
        chunks = chunk_srt_file(
            srt_file=file_path,
            course_id=course_id,
            lecture_id=lecture_id if lecture_id else "unknown",
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
    
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    return chunks