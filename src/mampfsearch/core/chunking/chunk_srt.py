"""SRT subtitle file chunking."""

import srt
import logging
from pathlib import Path
from typing import List

from mampfsearch.utils.models import Chunk, VideoLocation
from mampfsearch.core.chunking._helpers import (
    split_subtitle_at_periods,
    merge_until_sentence_complete,
    merge_until_min_size,
    split_large_chunks
)

logger = logging.getLogger(__name__)


def chunk_srt_file(
    srt_file: Path,
    course_id: str,
    lecture_id: str,
    min_chunk_size: int = 350,
    max_chunk_size: int = 750,
    overlap: bool = True,
    output_file: Path = None,
) -> List[Chunk]:
    """
    Chunk an SRT subtitle file into semantically coherent blocks.
    
    The chunking process:
    1. Split subtitles at sentence boundaries (periods)
    2. Merge consecutive sentences into complete blocks
    -- now every block ends with a full sentence --
    3. Grow blocks to reach min_chunk_size (with optional overlap)
    4. Split any blocks exceeding max_chunk_size
    
    Args:
        srt_file: Path to the .srt file
        course_id: Course identifier for metadata
        lecture_id: Lecture identifier for metadata
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
        overlap: If True, add context from adjacent subtitles
        output_file: Optional path to save final SRT for inspection
        
    Returns:
        List of Chunk objects with VideoLocation metadata
        
    Raises:
        ValueError: If max_chunk_size < min_chunk_size or file is not .srt
    """
    if max_chunk_size < min_chunk_size:
        raise ValueError("max_chunk_size must be >= min_chunk_size")
    
    logger.info(f"Chunking SRT file: {srt_file.name}")
    
    subs = list(_parse_srt_file(srt_file))
    logger.debug(f"Loaded {len(subs)} raw subtitles")

    # 1. Split so that no period is in the middle of a subtitle
    sentence_subs = []
    for sub in subs:
        sentence_subs.extend(split_subtitle_at_periods(sub))
    logger.debug(f"Split into {len(sentence_subs)} sentence-level subtitles")

    # 2. Merge consecutive sentences into sentence-complete blocks
    merged_sentences = merge_until_sentence_complete(sentence_subs)
    logger.debug(f"Merged into {len(merged_sentences)} sentence-complete blocks")

    # 3. Grow blocks to min_chunk_size (with optional overlap)
    grown = merge_until_min_size(merged_sentences, min_chunk_size, overlap)
    logger.debug(f"Grown to {len(grown)} chunks (min size: {min_chunk_size})")

    # 4. Split overly large blocks
    final_subs = split_large_chunks(grown, max_chunk_size)
    logger.info(f"Final chunk count: {len(final_subs)}")

    if output_file:
        _save_srt_file(final_subs, output_file)

    # 5. Convert to Chunk models
    return _subtitles_to_chunks(final_subs, course_id, lecture_id)


def _parse_srt_file(file_path: Path) -> List[srt.Subtitle]:
    """Parse an SRT file into subtitle objects."""
    if file_path.suffix != ".srt":
        raise ValueError(f"Not a valid SRT file: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    return srt.parse(content)


def _subtitles_to_chunks(
    subtitles: List[srt.Subtitle], 
    course_id: str, 
    lecture_id: str
) -> List[Chunk]:
    """Convert subtitle objects to Chunk models with VideoLocation."""
    chunks = []
    for sub in subtitles:
        chunk = Chunk(
            text=sub.content.strip(),
            location=VideoLocation(
                courseId=course_id,
                lectureId=lecture_id,
                start_time=sub.start,
                end_time=sub.end
            ),
        )
        chunks.append(chunk)
    return chunks


def _save_srt_file(subtitles: List[srt.Subtitle], output_path: Path) -> None:
    srt_content = srt.compose(subtitles, reindex=True)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.debug(f"Debug SRT saved to: {output_path}")