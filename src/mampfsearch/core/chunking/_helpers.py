import re
import srt
from copy import copy
from typing import List

def split_subtitle_at_periods(subtitle: srt.Subtitle) -> List[srt.Subtitle]:
    """
    Split a subtitle into multiple subtitles at sentence boundaries (periods).
    Time stamps are estimated proportionally based on character count.
    """
    sentences = re.split(r'(?<=\.)\s+', subtitle.content)
    if len(sentences) <= 1:
        return [subtitle]

    subtitles = []
    for index, sentence in enumerate(sentences):
        if index == 0:
            start_time = subtitle.start
        else:
            start_time = subtitles[-1].end

        if index == len(sentences) - 1:
            end_time = subtitle.end
        else:
            char_duration = (subtitle.end - subtitle.start) / len(subtitle.content)
            end_time = start_time + (char_duration * len(sentence))

        new_subtitle = srt.Subtitle(index, start_time, end_time, sentence)
        subtitles.append(new_subtitle)

    return subtitles


def merge_until_sentence_complete(subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
    """
    Merge consecutive subtitles until each block ends with a complete sentence.
    """
    merged = []
    current_index = 0
    
    while current_index < len(subtitles):
        merged_sub = copy(subtitles[current_index])
        end_index = current_index
        
        while end_index < len(subtitles) - 1:
            if subtitles[end_index].content.endswith("."):
                break
            end_index += 1
            merged_sub.content += " " + subtitles[end_index].content
            merged_sub.end = subtitles[end_index].end

        current_index = end_index + 1
        merged.append(merged_sub)

    return merged


def merge_until_min_size(
    subtitles: List[srt.Subtitle], 
    min_size: int, 
    overlap: bool
) -> List[srt.Subtitle]:
    """
    Grow subtitle blocks to at least min_size characters.
    
    If overlap=True, add context from adjacent subtitles.
    """
    merged = []
    current_index = 0
    
    while current_index < len(subtitles):
        merged_sub = copy(subtitles[current_index])
        end_index = current_index
        
        # Grow until we reach min_size
        while end_index < len(subtitles) - 1:
            if len(merged_sub.content) > min_size:
                break
            end_index += 1
            merged_sub.content += " " + subtitles[end_index].content
            merged_sub.end = subtitles[end_index].end

        # Add overlap context if requested
        if overlap:
            # Add previous subtitle as context
            if current_index > 0:
                merged_sub.content = (
                    subtitles[current_index - 1].content + " " + merged_sub.content
                )
                merged_sub.start = subtitles[current_index - 1].start

            # Add next subtitle as context
            if end_index < len(subtitles) - 1:
                merged_sub.content += " " + subtitles[end_index + 1].content
                merged_sub.end = subtitles[end_index + 1].end

        current_index = end_index + 1
        merged.append(merged_sub)

    return merged


def split_at_word_boundary(subtitle: srt.Subtitle, max_size: int) -> List[srt.Subtitle]:
    """
    Recursively split a subtitle at word boundaries if it exceeds max_size.
    """
    if len(subtitle.content) <= max_size:
        return [subtitle]
    
    words = subtitle.content.split()
    if len(words) <= 1:
        return [subtitle]  # Cannot split further
    
    # Split at the middle
    middle_idx = len(words) // 2
    first_text = " ".join(words[:middle_idx])
    second_text = " ".join(words[middle_idx:])
    
    # Calculate proportional timing
    total_duration = subtitle.end - subtitle.start
    first_duration = total_duration * (len(first_text) / len(subtitle.content))
    split_time = subtitle.start + first_duration
    
    first_half = srt.Subtitle(
        index=subtitle.index,
        start=subtitle.start,
        end=split_time,
        content=first_text
    )
    
    second_half = srt.Subtitle(
        index=subtitle.index,
        start=split_time,
        end=subtitle.end,
        content=second_text
    )
    
    # Recursively split if needed
    result = []
    result.extend(split_at_word_boundary(first_half, max_size))
    result.extend(split_at_word_boundary(second_half, max_size))
    return result


def split_large_chunks(
    subtitles: List[srt.Subtitle], 
    max_size: int
) -> List[srt.Subtitle]:
    """Split any subtitles that exceed max_size."""
    result = []
    for subtitle in subtitles:
        result.extend(split_at_word_boundary(subtitle, max_size))
    return result