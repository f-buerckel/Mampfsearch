from mampfsearch.utils.config import get_graph_storage, get_llm_client
from mampfsearch.utils.models import SegmentNode, MathEntity
from mampfsearch.utils import prompts, config
from typing import List, Optional, Dict, Iterable, Tuple, Set

import re
import logging
import json

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = "You are a careful assistant. Follow the instructions exactly and respond with only the requested JSON."


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words]).rstrip()
    return truncated + " ..."


def _build_separated_context_blocks(
    blocks: List[Tuple[str, str]],
    max_words: int,
) -> str:
    """Combine titled blocks into a single context string under a word budget.

    Blocks are kept separate via headers. If the total exceeds max_words, later blocks
    are truncated first; the first block should be the main span.
    """

    remaining = max_words
    rendered_parts: List[str] = []

    for title, content in blocks:
        header = f"\n\n### {title}\n"
        header_words = _word_count(header)
        if remaining <= header_words:
            break
        remaining -= header_words

        trimmed = _truncate_to_word_limit(content or "", remaining)
        rendered_parts.append(header + trimmed)
        remaining -= _word_count(trimmed)
        if remaining <= 0:
            break

    return "".join(rendered_parts).lstrip()


def _group_and_format_segments(segments: List[SegmentNode], separator: str = "\n[...]\n") -> str:
    """Group consecutive segments into paragraphs and separate non-consecutive ones.

    Segments are considered consecutive if their positions are adjacent (diff == 1).
    """
    if not segments:
        return ""

    sorted_segments = sorted(segments, key=lambda s: s.segment.position)
    
    formatted_groups = []
    current_group = []
    
    for seg in sorted_segments:
        if not current_group:
            current_group.append(seg)
        else:
            last_seg = current_group[-1]
            if seg.segment.position == last_seg.segment.position + 1:
                current_group.append(seg)
            else:
                # Close current group
                group_text = " ".join([s.segment.text for s in current_group])
                formatted_groups.append(group_text)
                current_group = [seg]
    
    if current_group:
        group_text = " ".join([s.segment.text for s in current_group])
        formatted_groups.append(group_text)
        
    return separator.join(formatted_groups)


def _get_bio_classification_label(labels: Iterable[str]) -> Optional[str]:
    for label in sorted(labels or []):
        lower = label.lower()
        if lower.startswith("b-") or lower.startswith("i-"):
            return label
    return None


def _has_classification_label(seg_node: SegmentNode, type_name: str) -> bool:
    type_lower = (type_name or "").lower()
    if not type_lower:
        return False
    for label in seg_node.labels or []:
        label_lower = label.lower()
        if label_lower == f"b-{type_lower}" or label_lower == f"i-{type_lower}":
            return True
    return False


def create_multi_entity_segment_spanning_question(
    segments: List[SegmentNode],
    entity_name: str,
    n_questions_per_span: int = 3,
    max_context_words: int = 20000,
    max_definition_words: int = 500,
    max_comention_words: int = 1500,
    max_comention_entities: int = 5,
    definition_label: str = "definition",
    allowed_related_entities: Optional[Set[str]] = None,
    max_mentioned_spans: int = 2,
):
    """Generate multi-entity spanning questions with enriched context.

    Similar to `create_multiple_segment_spanning_question`, but additionally:
    - Extracts all entities mentioned in the main span.
    - For each mentioned entity (excluding the target), adds:
      1) a combined 'definition' span about that entity (case-insensitive)
      2) a combined co-mention span containing segments that mention BOTH the target
         entity and that related entity.
    - Keeps these context blocks separate and instructs the LLM accordingly.
    - Caps the context length to ~max_context_words.
    """

    target_lower = (entity_name or "").strip().lower()
    if not target_lower:
        logger.warning(f"Skipping question generation: Invalid entity name '{entity_name}'")
        return []

    segments = sorted(segments, key=lambda s: s.segment.position)
    all_segment_ids = [s.graph_id for s in segments]

    spans: List[Tuple[str, List[SegmentNode]]] = []
    current_span: List[SegmentNode] = []
    current_type_lower: Optional[str] = None

    for seg_node in segments:
        classification_label = _get_bio_classification_label(seg_node.labels)
        if not classification_label:
            if current_span:
                spans.append((current_type_lower or "", current_span))
                current_span = []
                current_type_lower = None
            continue

        parts = classification_label.split("-", 1)
        prefix = parts[0].strip().lower()
        type_part_lower = (parts[1] if len(parts) > 1 else "").strip().lower()

        if prefix == "b":
            if current_span:
                spans.append((current_type_lower or "", current_span))
            current_span = [seg_node]
            current_type_lower = type_part_lower
        elif prefix == "i":
            if current_span:
                last_seg = current_span[-1]
                last_class = _get_bio_classification_label(last_seg.labels) or ""
                last_parts = last_class.split("-", 1)
                last_type_lower = (
                    (last_parts[1] if len(last_parts) > 1 else "").strip().lower()
                )
                if (
                    last_type_lower == type_part_lower
                    and seg_node.segment.position == last_seg.segment.position + 1
                ):
                    current_span.append(seg_node)
                    current_type_lower = type_part_lower
                else:
                    spans.append((current_type_lower or "", current_span))
                    current_span = [seg_node]
                    current_type_lower = type_part_lower
            else:
                current_span = [seg_node]
                current_type_lower = type_part_lower

    if current_span:
        spans.append((current_type_lower or "", current_span))

    graph_storage = get_graph_storage()

    # Pre-fetch mentioned entity names per segment for efficient filtering.
    try:
        mentions_by_segment_id = graph_storage.get_entity_mentions_for_segments(
            all_segment_ids
        )
    except Exception as e:
        logger.error(f"Failed to fetch entity mentions for segments: {e}")
        mentions_by_segment_id = {}

    import random

    # Separate spans into 'about' and 'mentioned'
    about_spans = []
    mentioned_spans = []

    for span_type, span in spans:
        if not span:
            continue
        
        # Check if span is about target
        is_about_target = (span[0].segment.about_entity or "").strip().lower() == target_lower
        
        if is_about_target:
            about_spans.append((span_type, span))
        else:
            # Check if target is mentioned
            is_mentioned = False
            for s in span:
                if target_lower in mentions_by_segment_id.get(s.graph_id, set()):
                    is_mentioned = True
                    break
            
            if is_mentioned:
                mentioned_spans.append((span_type, span))

    # Sample mentioned spans if needed
    if len(mentioned_spans) > max_mentioned_spans:
        logger.info(f"Sampling {max_mentioned_spans} mentioned spans from {len(mentioned_spans)} total.")
        mentioned_spans = random.sample(mentioned_spans, max_mentioned_spans)
    
    # Combine valid spans
    valid_spans = about_spans + mentioned_spans

    # Performance Logging
    logger.info(f"Entity '{entity_name}': Found {len(valid_spans)} valid spans "
                f"({len(about_spans)} 'about', {len(mentioned_spans)} 'mentioned').")

    if not valid_spans:
        logger.info(f"No valid spans found for entity '{entity_name}' (target_lower='{target_lower}'). "
                    f"This entity is neither the subject nor mentioned in any segment.")
        return []

    llm_client = get_llm_client()
    questions = []

    for span_type, span in valid_spans:
        span_text = "\n".join([s.segment.text for s in span])

        # Entities mentioned in main span
        entities = graph_storage.get_entities_in_segments(span)
        related_entities: Dict[str, str] = {}
        descriptions: Dict[str, str] = {}

        for ent_node in entities:
            ent = ent_node.math_entity
            name = (ent.name or "").strip()
            if not name:
                continue
            name_lower = name.lower()
            if name_lower == target_lower:
                continue

            # Filter non-allowed entities if allowlist is provided
            if allowed_related_entities is not None:
                 if name_lower not in allowed_related_entities:
                     continue

            # Keep a stable original casing for display
            related_entities[name_lower] = name
            if ent.description:
                descriptions[name] = ent.description

        # 1. Filter related entities based on co-mention frequency
        # Count co-mentions for each related entity
        comention_counts = {}
        for related_lower in related_entities.keys():
            count = 0
            for s in segments:
                mentioned = mentions_by_segment_id.get(s.graph_id, set())
                if target_lower in mentioned and related_lower in mentioned:
                    count += 1
            comention_counts[related_lower] = count
        
        # Sort by count (descending), then name (ascending) for stability
        sorted_related_keys = sorted(
            related_entities.keys(),
            key=lambda k: (-comention_counts.get(k, 0), k)
        )
        
        # Keep top K
        top_k_keys = sorted_related_keys[:max_comention_entities]
                
        other_entities_list = [related_entities[k] for k in top_k_keys]
        # Sort alphabetically for display in the prompt list
        other_entities_list.sort(key=lambda s: s.lower())

        mentioned_entities_block = (
            "\n".join([f"- {n}" for n in other_entities_list])
            if other_entities_list
            else "(none detected)"
        )

        description_text = "\n".join(
            [f"{related_entities[k]}: {descriptions.get(related_entities[k], '')}" for k in top_k_keys if descriptions.get(related_entities[k])]
        )

        blocks: List[Tuple[str, str]] = [
            (f"Main Text (span type: {span_type or 'unknown'})", span_text),
            ("Mentioned Entities In Main Span", mentioned_entities_block),
        ]
        if description_text.strip():
            blocks.append(("Related Entity Descriptions", description_text))

        # Consolidate Definition + Co-mention contexts
        
        # Track seen segments to avoid duplication across blocks
        seen_segment_ids = set()
        
        # Mark main span segments as seen
        for s in span:
            seen_segment_ids.add(s.graph_id)

        # Iterate only over the filtered top K entities
        for related_lower in top_k_keys:
            related_name = related_entities[related_lower]
            
            # 1. Definition segments
            def_segments = []
            for s in segments:
                if (s.segment.about_entity or "").strip().lower() == related_lower and _has_classification_label(s, definition_label):
                    if s.graph_id not in seen_segment_ids:
                        def_segments.append(s)
                        seen_segment_ids.add(s.graph_id)

            def_segments.sort(key=lambda s: s.segment.position)
            if def_segments:
                def_text = _group_and_format_segments(def_segments)
                # Truncate definition text
                def_text = _truncate_to_word_limit(def_text, max_definition_words)
                blocks.append((f"Definition Context — {related_name}", def_text))

            # 2. Co-mention segments (separate blocks per entity, strictly deduplicated)
            comention_segments = []
            for s in segments:
                mentioned = mentions_by_segment_id.get(s.graph_id, set())
                if target_lower in mentioned and related_lower in mentioned:
                    if s.graph_id not in seen_segment_ids:
                        comention_segments.append(s)
                        seen_segment_ids.add(s.graph_id)
            
            
            comention_segments.sort(key=lambda s: s.segment.position)
            if comention_segments:
                # Use coherent grouping
                comention_text = _group_and_format_segments(comention_segments)
                # Truncate co-mention text
                comention_text = _truncate_to_word_limit(comention_text, max_comention_words)
                blocks.append(
                    (
                        f"Co-mention Context — {entity_name} + {related_name}",
                        comention_text,
                    )
                )

        full_context = _build_separated_context_blocks(
            blocks=blocks,
            max_words=max_context_words,
        )

        prompt = prompts.CREATE_MULTI_ENTITY_SPANNING_QUESTION_PROMPT.format(
            entity=entity_name,
            context=full_context,
            n_questions=n_questions_per_span,
            other_entities=other_entities_list,
        )

        try:
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": _DEFAULT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            response_content = response.choices[0].message.content
            response_dict = parse_llm_response(
                response=response_content,
                keys=("reasoning", "blooms_level", "questions", "answers", "explanations"),
            )
            if not response_dict:
                continue

            reasoning = response_dict.get("reasoning", "")
            blooms_level = response_dict.get("blooms_level", 0)
            logger.info(f"Generated questions with Bloom's Level {blooms_level}. Reasoning: {reasoning}")

            generated_questions = response_dict.get("questions") or []
            generated_answers = response_dict.get("answers") or []
            generated_explanations = response_dict.get("explanations") or []

            for q, a, e in zip(
                generated_questions, generated_answers, generated_explanations
            ):
                if not q or not a:
                    continue

                scores = evaluate_question_with_llm(
                    context=full_context,
                    question=q,
                    answer=a,
                    entity_name=entity_name,
                )

                questions.append(
                    {
                        "span_type": span_type,
                        "span_text": span_text,
                        "question": q,
                        "answer": a,
                        "explanation": e,
                        "span_ids": [s.graph_id for s in span],
                        "context": full_context,
                        "evaluation": scores,
                    }
                )

        except Exception as e:
            logger.error(f"Error during multi-entity spanning question LLM call: {e}")
            continue

    return questions


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
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": _DEFAULT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
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
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": _DEFAULT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
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
    segments: List[SegmentNode],
    entity_name: str,
    n_questions_per_span: int = 5,
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
                        label
                        for label in last_labels
                        if label.startswith("B-") or label.startswith("I-")
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
            entity=entity_name,
            context=full_context,
            n_questions=n_questions_per_span,
        )

        try:
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": _DEFAULT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            response_content = response.choices[0].message.content
            response_dict = parse_llm_response(
                response=response_content,
                keys=("reasoning", "blooms_level", "questions", "answers", "explanations"),
            )
            if response_dict:
                logger.debug(f"Spanning Question Prompt:\n{full_context}")
                reasoning = response_dict.get("reasoning") or "No reasoning provided."
                blooms_level = response_dict.get("blooms_level", 0)
                logger.info(f"Generated questions with Bloom's Level {blooms_level}. Reasoning: {reasoning}")
                generated_questions = response_dict.get("questions") or []
                generated_answers = response_dict.get("answers") or []
                generated_explanations = response_dict.get("explanations") or []

                for q, a, e in zip(
                    generated_questions, generated_answers, generated_explanations
                ):
                    if not q or not a:
                        continue

                    logger.debug(f"Question: {q}")
                    logger.debug(f"Answer: {a}")
                    logger.debug(f"Explanation: {e}")

                    scores = evaluate_question_with_llm(
                        context=full_context,
                        question=q,
                        answer=a,
                        entity_name=entity_name,
                    )
                    logger.debug(f"Evaluation Scores: {scores}")

                    questions.append(
                        {
                            "span_text": span_text,
                            "question": q,
                            "answer": a,
                            "explanation": e,
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
    llm_score = evaluate_question_with_llm(
        context=context, question=question, answer=answer, entity_name=entity_name
    )

    return llm_score


def generate_unstructured_questions(context: str, n_questions: int = 10):
    llm_client = get_llm_client()

    prompt = prompts.CREATE_UNSTRUCTURED_QUESTION_PROMPT_NO_ENTITY.format(
        context=context, n_questions=n_questions
    )

    try:
        response = llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": _DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
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
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": _DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
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
