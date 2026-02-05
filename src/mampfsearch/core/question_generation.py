from mampfsearch.utils.config import get_graph_storage, get_llm_client
from mampfsearch.core.entity_selection import find_relevant_entities_in_lecture
from mampfsearch.utils.models import SegmentNode, MathEntity
from mampfsearch.utils import prompts, config
from typing import List, Optional, Dict

import re
import logging
import json

logger = logging.getLogger(__name__)


def parse_llm_response(response: str, keys: tuple) -> Optional[Dict[str, str]]:
    # Returns either none if not all keys are present or the parsed dict.

    # 1. Strip Markdown code blocks if present (e.g., ```json ... ```)
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_pattern, response)
    if match:
        response = match.group(1)
    else:
        # Fallback: Try to find the outermost JSON braces in the raw string
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            response = response[start : end + 1]

    # We escape all backslashes that are NOT followed by a double quote.
    # This preserves \" (escaped quote) but turns \theta into \\theta, \n into \\n, etc.
    # This ensures json.loads sees them as literal backslash characters in the string.
    response_sanitized = re.sub(r'\\(?!")', r"\\\\", response)

    try:
        # strict=False allows control characters (like literal newlines) inside strings
        data = json.loads(response_sanitized, strict=False)

        if all(key in data for key in keys):
            return {key: data.get(key) for key in keys}
        else:
            logger.debug(
                f"Not all required keys found in LLM response. Required: {keys}, Found: {data.keys()}"
            )
            return None
    except json.JSONDecodeError:
        # Attempt to recover from truncated JSON
        try:
            clean_str = response_sanitized.strip()
            # If the string ends with a quote, it might just be missing the closing brace
            if clean_str.endswith('"'):
                data = json.loads(clean_str + "}", strict=False)
                if all(key in data for key in keys):
                    return {key: data.get(key) for key in keys}
        except json.JSONDecodeError:
            pass

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


def create_multiple_segment_spanning_question(
    segments: List[SegmentNode], entity_name: str
):
    # Sort segments by position
    segments = sorted(segments, key=lambda s: s.segment.position)

    spans = []
    current_span = []

    for i, seg_node in enumerate(segments):
        labels = seg_node.labels
        classification_label = None
        for label in labels:
            if label.startswith("B-") or label.startswith("I-"):
                classification_label = label
                break

        if not classification_label:
            if current_span:
                spans.append(current_span)
                current_span = []
            continue

        type_part = classification_label.split("-")[1]
        prefix = classification_label.split("-")[0]

        if prefix == "B":
            if current_span:
                spans.append(current_span)
            current_span = [seg_node]
        elif prefix == "I":
            if current_span:
                last_seg = current_span[-1]
                last_labels = last_seg.labels
                last_class = next(
                    (
                        l
                        for l in last_labels
                        if l.startswith("B-") or l.startswith("I-")
                    ),
                    "",
                )
                last_type = last_class.split("-")[1] if "-" in last_class else ""

                if (
                    last_type == type_part
                    and seg_node.segment.position == last_seg.segment.position + 1
                ):
                    current_span.append(seg_node)
                else:
                    spans.append(current_span)
                    current_span = [seg_node]
            else:
                current_span = [seg_node]

    if current_span:
        spans.append(current_span)

    valid_spans = []
    for span in spans:
        if not span:
            continue
        first_seg = span[0]
        # Check about_entity.
        about = first_seg.segment.about_entity
        if about and entity_name and about.lower() == entity_name.lower():
            valid_spans.append(span)

    llm_client = get_llm_client()
    graph_storage = get_graph_storage()
    questions = []

    for span in valid_spans:
        # Construct Context
        span_text = "\n".join([s.segment.text for s in span])

        # Get entities in segments
        entities = graph_storage.get_entities_in_segments(span)
        # Unique entity descriptions
        descriptions = {}
        for ent_node in entities:
            ent = ent_node.math_entity
            if ent.name.lower() != entity_name.lower() and ent.description:
                descriptions[ent.name] = ent.description

        description_text = "\n".join(
            [f"{name}: {desc}" for name, desc in descriptions.items()]
        )

        full_context = (
            f"Main Text:\n{span_text}\n\nRelated Entity Descriptions:\n{description_text}"
            if descriptions
            else f"Main Text:\n{span_text}"
        )

        prompt = prompts.CREATE_SPANNING_QUESTION_PROMPT.format(
            entity=entity_name, context=full_context
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
            response_content = response.choices[0].message.content
            response_dict = parse_llm_response(
                response=response_content, keys=("question", "answer", "explanation")
            )
            if response_dict:
                logger.debug(f"Spanning Question Prompt:\n{full_context}")
                logger.debug(f"Question: {response_dict['question']}")
                logger.debug(f"Answer: {response_dict['answer']}")
                logger.debug(f"Explanation: {response_dict['explanation']}")

                scores = evaluate_question_with_llm(
                    context=full_context,
                    question=response_dict["question"],
                    answer=response_dict["answer"],
                    entity_name=entity_name,
                )
                logger.debug(f"Evaluation Scores: {scores}")

                questions.append(
                    {
                        "span_text": span_text,
                        "question": response_dict["question"],
                        "answer": response_dict["answer"],
                        "explanation": response_dict["explanation"],
                        "span_ids": [s.graph_id for s in span],
                        "context": full_context,
                        "evaluation": scores,
                    }
                )

        except Exception as e:
            logger.error(f"Error during spanning question LLM call: {e}")
            continue

    return questions


def evaluate_question(
    context: str,
    question: str,
    answer: str,
    entities: List[MathEntity],
    entity_name: str,
):
    similarity = similarity_to_context(context, question)

    llm_score = evaluate_question_with_llm(
        context=context, question=question, answer=answer, entity_name=entity_name
    )

    return llm_score


def generate_unstructured_questions(
    context: str, n_questions: int = 10, entity_name: str = ""
):
    llm_client = get_llm_client()
    prompt = prompts.CREATE_UNSTRUCTURED_QUESTION_PROMPT.format(
        context=context, n_questions=n_questions, entity=entity_name
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
        content = response.choices[0].message.content
        response = parse_llm_response(response=content, keys=("questions", "answers"))
        if response:
            return [
                {"question": q, "answer": a}
                for q, a in zip(response["questions"], response["answers"])
            ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error during unstructured question generation LLM call: {e}")
        return []


def evaluate_question_with_llm(
    context: str, question: str, answer: str, entity_name: str
) -> float:
    llm_client = get_llm_client()
    prompt = prompts.QUESTION_EVALUATION_PROMPT.format(
        context=context, entity_name=entity_name, question=question, answer=answer
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
        response_dict = parse_llm_response(
            response=response,
            keys=(
                "reasoning_steps",
                "faithfulness",
                "entity_alignment",
                "math_correctness",
                "pedagogical_utility",
                "verdict",
            ),
        )
        if response_dict:
            # scores = {key: int(response_dict[key]) for key in response_dict.keys()}

            # return the average score:
            return response_dict
            # return sum(scores.values()) / len(scores)

        else:
            logger.debug("LLM response did not contain all required evaluation keys.")
            return 0.0

    except Exception as e:
        logger.error(f"Error during question evaluation LLM call: {e}")
        return 0.0


def similarity_to_context(context: str, question: str):
    model = config.get_embedding_model()
    context_embedding = model.encode(context, return_dense=True)
    question_embedding = model.encode(question, return_dense=True)
    similarity = context_embedding @ question_embedding.T

    return similarity
