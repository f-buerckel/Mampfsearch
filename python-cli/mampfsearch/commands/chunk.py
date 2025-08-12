import re
import srt
from copy import copy
import logging

logger = logging.getLogger(__name__)

def chunk_srt_command(srt_file, min_chunk_size, output, overlap):
    """Optimizes chunk size of SRT files for better semantic search results"""
    result = chunk_srt(srt_file, min_chunk_size, output, overlap)
    if result == -1:
        logger.info("Error processing SRT file", err=True)
    else:
        logger.info(f"Successfully chunked SRT to {output}")

def get_srt(file_path):
    if not file_path.endswith(".srt"):
        raise ValueError(f"The file {file_path} is not a valid SRT file.")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return srt.parse(content)

def chunk_srt(srt_file, min_chunk_size, output_file, overlap):
    try: 
        subs = list(get_srt(srt_file))
    except (FileNotFoundError, ValueError) as e:
        logger.info(e, err=True)
        return -1
    
    # Split subtitles at periods
    sentence_subs = []
    for sub in subs:
        sentences = split_subtitle_at_periods(sub)
        [sentence_subs.append(s) for s in sentences]
    
    # Merge subtitles by sentences (each subtitle is a full sentence)
    merged_senteces = merge_subtitles_by_sentence(sentence_subs)

    # Merge sentences to reach min_chunk_size
    merged_subtitles = merge_subtitles_by_size(merged_senteces, min_chunk_size, overlap)

    # Split large chunks if they exceed max_chunk_size
    max_chunk_size = 750
    if max_chunk_size < min_chunk_size:
        logger.info(f"Warning: max_chunk_size ({max_chunk_size}) is smaller than min_chunk_size ({min_chunk_size}).", err=True)
        raise ValueError("max_chunk_size must be greater than or equal to min_chunk_size.")
    final_subtitles = split_large_chunks(merged_subtitles, max_chunk_size)

    final_srt = srt.compose(final_subtitles, reindex=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(final_srt)
    return 0

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
    # not optimal but get merged anyways later
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