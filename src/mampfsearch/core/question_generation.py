from mampfsearch.utils.config import get_graph_storage, get_llm_client
from mampfsearch.core.entity_selection import find_relevant_entities_in_lecture
from mampfsearch.utils.models import SegmentNode
from mampfsearch.utils import prompts
from typing import List, Optional, Dict

import re
import logging
import json

logger = logging.getLogger(__name__)


def parse_llm_response(response: str, keys: tuple) -> Optional[Dict[str, str]]:
    # Returns either none if not all keys are present or the parsed dict.

    # Escape backslashes that aren't part of valid JSON escape sequences
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    # Avoid touching already-escaped sequences like `\\_` (which would become invalid `\\\_`).
    response = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", response)
    try:
        data = json.loads(response)

        if all(key in data for key in keys):
            return {key: data.get(key) for key in keys}
        else:
            logger.debug(
                f"Not all required keys found in LLM response. Required: {keys}, Found: {data.keys()}"
            )
            return None
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response as JSON. Response: {response}")
        return None


def create_factual_questions(segments: List[SegmentNode], entity_name: str):
    llm_client = get_llm_client()
    questions = []
    for segmentNode in segments:
        segment = segmentNode.segment
        prompt = prompts.CREATE_FACTUAL_QUESTION_PROMPT.format(
            entity=entity_name, segment=segment.text
        )

        try:
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
            )
            response = response.choices[0].message.content
            response = parse_llm_response(
                response=response, keys=("question", "answer", "contains_fact")
            )
            if response:
                question = {
                    "segment_text": segment.text,
                    "question": response["question"],
                    "answer": response["answer"],
                }
                questions.append(question)

        except Exception as e:
            logger.error(f"Error during factual question LLM call: {e}")
            continue

    return questions


# TODO: Dry it up
def create_multiple_choice_question(segments: List[SegmentNode], entity_name: str):
    llm_client = get_llm_client()
    questions = []
    for segmentNode in segments:
        segment = segmentNode.segment
        prompt = prompts.CREATE_MULTIPLE_CHOICE_QUESTION_PROMPT.format(
            entity=entity_name, segment=segment.text
        )

        try:
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
            )
            response = response.choices[0].message.content
            response = parse_llm_response(
                response,
                keys=(
                    "question",
                    "answer",
                    "distractor1",
                    "distractor2",
                    "distractor3",
                ),
            )
            if response:
                question = {
                    "segment_text": segment.text,
                    "question": response["question"],
                    "answer": response["answer"],
                    "distractor1": response["distractor1"],
                    "distractor2": response["distractor2"],
                    "distractor3": response["distractor3"],
                }
                questions.append(question)

        except Exception as e:
            logger.error(f"Error during factual question LLM call: {e}")
            continue

    return questions
