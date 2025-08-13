from copy import copy
from typing import List
from mampfsearch.utils.models import Chunk
import re
import srt
import logging

logger = logging.getLogger(__name__)


def chunk_srt(
    srt_file: str,
    lecture_name: str,
    lecture_position: int,
    min_chunk_size: int,
    overlap: bool,
    max_chunk_size: int = 750,
) -> List[Chunk]:
    """
    Chunk an SRT file and return a list of Chunk models (no file I/O).
    """
    subs = list(get_srt(srt_file))

    # 1) split into sentence-level subtitles
    sentence_subs: List[srt.Subtitle] = []
    for sub in subs:
        sentence_subs.extend(split_subtitle_at_periods(sub))

    # 2) merge consecutive sentences into sentence-complete blocks
    merged_sentences = merge_subtitles_by_sentence(sentence_subs)

    # 3) grow blocks to min_chunk_size (optionally with overlap context)
    grown = merge_subtitles_by_size(merged_sentences, min_chunk_size, overlap)

    # 4) split overly large blocks
    if max_chunk_size < min_chunk_size:
        raise ValueError("max_chunk_size must be >= min_chunk_size")
    final_subtitles = split_large_chunks(grown, max_chunk_size)

    # 5) map to Chunk models
    return subtitles_to_chunks(
        final_subtitles, lecture_name=lecture_name, lecture_position=lecture_position
    )

def get_srt(file_path):
    if not file_path.endswith(".srt"):
        raise ValueError(f"The file {file_path} is not a valid SRT file.")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return srt.parse(content)

def subtitles_to_chunks(final_subtitles: List[srt.Subtitle], lecture_name: str, lecture_position: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, sub in enumerate(final_subtitles):
        chunks.append(
            Chunk(
                text=sub.content.strip(),
                lecture_name=lecture_name,
                lecture_position=lecture_position,
                position=i,
                start_time=sub.start,
                end_time=sub.end,
            )
        )
    return chunks

def merge_subtitles_by_sentence(subtitles):
    merged_subtitles = []
    current_index = 0
    while current_index < len(subtitles):
        merged_subtitle = copy(subtitles[current_index])
        end_index = current_index
        while end_index < len(subtitles)-1:
            if subtitles[end_index].content.endswith("."):
                break
            end_index += 1
            merged_subtitle.content += " " + subtitles[end_index].content
            merged_subtitle.end = subtitles[end_index].end

        current_index = end_index + 1
        merged_subtitles.append(merged_subtitle)

    return merged_subtitles

def merge_subtitles_by_size(subtitles, min_chunk_size, overlap):
    merged_subtitles = []
    current_index = 0
    while current_index < len(subtitles):
        merged_subtitle = copy(subtitles[current_index])
        end_index = current_index
        while end_index < len(subtitles)-1:
            if len(merged_subtitle.content) > min_chunk_size:   
                break

            end_index += 1
            merged_subtitle.content += " " + subtitles[end_index].content 
            merged_subtitle.end = subtitles[end_index].end

        if overlap:
            # Sliding windows 
            if current_index > 0:
                merged_subtitle.content = subtitles[current_index-1].content + " " + merged_subtitle.content 
                merged_subtitle.start = subtitles[current_index-1].start

            if end_index < len(subtitles) - 1:
                merged_subtitle.content += " " + subtitles[end_index + 1].content
                merged_subtitle.end = subtitles[end_index + 1].end

        current_index = end_index + 1
        merged_subtitles.append(merged_subtitle)

    return merged_subtitles

def split_subtitle_at_periods(subtitle):
    sentences = re.split(r'(?<=\.)\s+', subtitle.content)
    if len(sentences) == 0:
        return subtitle

    subtitles = []
    # calculate estimated start and end time
    # not optimal but chunks get merged anyways later
    for index, sentence in enumerate(sentences):
        if index == 0:
            start_time = subtitle.start
        else:
            start_time = subtitles[-1].end

        if index == len(sentences) - 1:
            end_time = subtitle.end
        else:
            end_time = start_time + ( (subtitle.end - subtitle.start) / len(subtitle.content) ) / len(sentence)

        # Create new subtitle object
        new_subtitle = srt.Subtitle(index, start_time, end_time, sentence)
        subtitles.append(new_subtitle)

    return subtitles

def split_large_chunks(subtitles, max_chunk_size):
    """Split chunks that exceed max_chunk_size at word boundaries"""
    result_subtitles = []
    
    for subtitle in subtitles:
        split_chunks = split_if_too_large(subtitle, max_chunk_size)
        result_subtitles.extend(split_chunks)
    
    return result_subtitles

def split_if_too_large(subtitle, max_chunk_size):
    """Recursively split subtitle if it's too large"""
    if len(subtitle.content) <= max_chunk_size:
        return [subtitle]
    
    # Find middle point at word boundary
    words = subtitle.content.split()
    if len(words) <= 1:
        return [subtitle]  # Can't split further
    
    # Split roughly in the middle
    middle_index = len(words) // 2
    
    first_half_text = " ".join(words[:middle_index])
    second_half_text = " ".join(words[middle_index:])
    
    # Calculate timing (split duration proportionally)
    total_duration = subtitle.end - subtitle.start
    first_half_duration = total_duration * (len(first_half_text) / len(subtitle.content))
    split_time = subtitle.start + first_half_duration
    
    # Create two subtitle objects
    first_half = srt.Subtitle(
        index=subtitle.index,
        start=subtitle.start,
        end=split_time,
        content=first_half_text
    )
    
    second_half = srt.Subtitle(
        index=subtitle.index,
        start=split_time,
        end=subtitle.end,
        content=second_half_text
    )
    
    # Recursively check if the halves are still too large
    result = []
    result.extend(split_if_too_large(first_half, max_chunk_size))
    result.extend(split_if_too_large(second_half, max_chunk_size))
    
    return result