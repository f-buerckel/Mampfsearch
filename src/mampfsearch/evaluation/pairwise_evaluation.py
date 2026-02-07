import argparse
import csv
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from mampfsearch.utils.config import get_llm_client

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = "You are a careful assistant. Follow the instructions exactly and respond with only the requested JSON."


PAIRWISE_JUDGE_PROMPT = """
You are an expert university pedagogue and NLP evaluator.

Task:
You will compare two candidate question-answer pairs (A and B). Choose the better one according to educational objectives.

Judging criteria (prioritize in this order):
1) Answer consistency: The answer must actually answer the question asked.
2) Relevance: The question targets a meaningful educational objective (not trivia).
3) Educational value: Prefer questions that test understanding/integration over trivial recall.
4) Clarity & conciseness.
5) Independence: Avoid answer leakage.

Decision rules:
- If one candidate is clearly better overall, pick it.
- If both are similarly good/bad, choose TIE.
- If one candidate looks internally inconsistent or nonsensical, it should lose.

Return ONLY valid JSON with exactly these keys:
Return a single JSON object with exactly these keys (and no other text):
{{
    "reasoning": "<short overall rationale>",
    "winner": "A" | "B" | "TIE"
}}

Candidate A:
Lecture: {lecture_a}
Question: {question_a}
Answer: {answer_a}

Candidate B:
Lecture: {lecture_b}
Question: {question_b}
Answer: {answer_b}
""".strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + " ..."


def _require_columns(df: pd.DataFrame, columns: Sequence[str], csv_path: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing}")


def _extract_json_candidate_text(response: str) -> str:
    """Extract a likely JSON substring from a raw LLM response.

    This is intentionally best-effort: it strips markdown fences or slices the outermost braces.
    """

    if not response:
        return ""

    # Strip Markdown code blocks if present
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_pattern, response)
    if match:
        return match.group(1)

    # Fallback: attempt to slice outermost braces
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    return response


def parse_llm_json(response: str) -> Optional[Dict[str, Any]]:
    data, _err = parse_llm_json_with_error(response)
    return data if isinstance(data, dict) else None


def parse_llm_json_with_error(response: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse JSON from an LLM response and return (data, error).

    Returns:
      - (parsed_obj, None) on success
      - (None, "...") on failure, with a human-readable error string
    """

    if response is None:
        return None, "response_is_none"

    extracted = _extract_json_candidate_text(str(response))
    if not extracted.strip():
        return None, "empty_response"

    # Make backslashes safer for json.loads
    sanitized = re.sub(r'\\(?!")', r"\\\\", extracted)
    try:
        return json.loads(sanitized, strict=False), None
    except json.JSONDecodeError as e:
        return None, f"{e.msg} (line {e.lineno}, col {e.colno})"


@dataclass(frozen=True)
class Candidate:
    source: str
    row_index: int
    lecture_name: str
    entity_name: str
    context: str
    question: str
    answer: str


def _randomize_ab(
    cand_from_csv_a: Candidate, cand_from_csv_b: Candidate, rng: random.Random
) -> Tuple[Candidate, Candidate]:
    """Randomly assign which underlying method becomes A vs B.

    This combats positional/label bias. The model only ever sees the labels A/B.
    """

    if rng.random() < 0.5:
        return cand_from_csv_a, cand_from_csv_b
    return cand_from_csv_b, cand_from_csv_a


def _row_to_candidate(
    row: pd.Series, source: str, row_index: int, context_words: int
) -> Candidate:
    lecture = str(row.get("lecture_name", "") or "")
    entity = str(row.get("entity_name", "") or "")
    context = str(row.get("context", "") or "")
    question = str(row.get("question", "") or "")
    answer = str(row.get("answer", "") or "")

    return Candidate(
        source=source,
        row_index=int(row_index),
        lecture_name=lecture,
        entity_name=entity,
        context=_truncate_to_word_limit(context, context_words),
        question=question.strip(),
        answer=answer.strip(),
    )


def _sample_pairs_none(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_pairs: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    idx_a = list(df_a.index)
    idx_b = list(df_b.index)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    if n_pairs <= min(len(idx_a), len(idx_b)):
        return list(zip(idx_a[:n_pairs], idx_b[:n_pairs]))

    # If asked for more than available, sample with replacement
    pairs: List[Tuple[int, int]] = []
    for _ in range(n_pairs):
        pairs.append((rng.choice(idx_a), rng.choice(idx_b)))
    return pairs


def _sample_pairs_match(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_pairs: int,
    rng: random.Random,
    match_on: str,
) -> List[Tuple[int, int]]:
    if match_on == "lecture":
        key_cols = ["lecture_name"]
    elif match_on == "lecture_entity":
        key_cols = ["lecture_name", "entity_name"]
    else:
        raise ValueError(f"Unknown match_on: {match_on}")

    grouped_a: Dict[Tuple[str, ...], List[int]] = {}
    grouped_b: Dict[Tuple[str, ...], List[int]] = {}

    for idx, row in df_a.iterrows():
        key = tuple(str(row.get(c, "") or "") for c in key_cols)
        grouped_a.setdefault(key, []).append(idx)
    for idx, row in df_b.iterrows():
        key = tuple(str(row.get(c, "") or "") for c in key_cols)
        grouped_b.setdefault(key, []).append(idx)

    common_keys = sorted(set(grouped_a.keys()) & set(grouped_b.keys()))
    if not common_keys:
        raise ValueError(
            f"No overlap between CSVs for match_on={match_on} using columns {key_cols}."
        )

    pairs: List[Tuple[int, int]] = []
    for _ in range(n_pairs):
        key = rng.choice(common_keys)
        pairs.append((rng.choice(grouped_a[key]), rng.choice(grouped_b[key])))
    return pairs


def judge_pair(
    candidate_a: Candidate,
    candidate_b: Candidate,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        lecture_a=candidate_a.lecture_name,
        question_a=candidate_a.question,
        answer_a=candidate_a.answer,
        lecture_b=candidate_b.lecture_name,
        question_b=candidate_b.question,
        answer_b=candidate_b.answer,
    )

    llm_client = get_llm_client()
    response = llm_client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    # Some OpenAI-compatible backends can return content=None (e.g. tool-calls only).
    # In that case, dump as much of the response as possible for debugging.
    choices = getattr(response, "choices", None)
    if not choices:
        try:
            dumped = (
                response.model_dump_json(indent=2)
                if hasattr(response, "model_dump_json")
                else repr(response)
            )
        except Exception:
            dumped = repr(response)
        return None, "llm_no_choices", dumped

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        debug_parts: List[str] = []
        try:
            debug_parts.append(
                f"finish_reason={getattr(choices[0], 'finish_reason', None)!r}"
            )
        except Exception:
            pass
        try:
            debug_parts.append(f"tool_calls={getattr(message, 'tool_calls', None)!r}")
        except Exception:
            pass
        try:
            debug_parts.append(
                f"function_call={getattr(message, 'function_call', None)!r}"
            )
        except Exception:
            pass
        try:
            dumped = (
                response.model_dump_json(indent=2)
                if hasattr(response, "model_dump_json")
                else (
                    json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
                    if hasattr(response, "model_dump")
                    else repr(response)
                )
            )
        except Exception:
            dumped = repr(response)
        debug_text = "\n".join([p for p in debug_parts if p])
        if debug_text:
            debug_text = debug_text + "\n\n--- FULL_RESPONSE ---\n" + dumped
        else:
            debug_text = dumped
        return None, "llm_returned_null_content", debug_text

    data, parse_error = parse_llm_json_with_error(content)
    if parse_error is not None:
        return None, f"json_parse_error: {parse_error}", content
    if not isinstance(data, dict):
        data_type = type(data).__name__
        return None, f"json_not_object(type={data_type})", content

    # Be lenient: allow extra keys; require at least winner+reasoning.
    if "winner" not in data or "reasoning" not in data:
        return None, "missing_required_keys", content

    winner = str(data.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        return None, "invalid_winner_value", content
    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        return None, "empty_reasoning", content

    return {"winner": winner, "reasoning": reasoning}, None, content


def run_pairwise_evaluation(
    csv_a: str,
    csv_b: str,
    name_a: str = "A",
    name_b: str = "B",
    n_pairs: int = 100,
    match_on: str = "none",
    seed: int = 0,
    context_words: int = 1200,
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.0,
    max_tokens: int = 100000,
    debug_invalid: bool = False,
    debug_invalid_dir: Optional[str] = None,
    sleep: float = 0.0,
    output: Optional[str] = None,
) -> Dict[str, int]:
    logging.basicConfig(level=logging.INFO)

    if n_pairs <= 0:
        raise ValueError("--n-pairs must be > 0")

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    # Normalize NaNs so we don't accidentally treat them as non-empty strings like 'nan'.
    for df in (df_a, df_b):
        for col in ("lecture_name", "entity_name", "context", "question", "answer"):
            if col in df.columns:
                df[col] = df[col].fillna("")

    required_cols = ["lecture_name", "entity_name", "context", "question", "answer"]
    _require_columns(df_a, required_cols, csv_a)
    _require_columns(df_b, required_cols, csv_b)

    # Basic cleanup: drop empty Q/A
    def _nonempty(s: Any) -> bool:
        # After fillna(""), this treats empty/whitespace-only as empty.
        return bool(str(s).strip())

    df_a = df_a[
        df_a["question"].apply(_nonempty) & df_a["answer"].apply(_nonempty)
    ].copy()
    df_b = df_b[
        df_b["question"].apply(_nonempty) & df_b["answer"].apply(_nonempty)
    ].copy()

    if df_a.empty or df_b.empty:
        raise ValueError(
            "One of the inputs has no non-empty question/answer rows after filtering."
        )

    rng = random.Random(seed)
    if match_on == "none":
        sampled_pairs = _sample_pairs_none(df_a, df_b, n_pairs, rng)
    else:
        sampled_pairs = _sample_pairs_match(
            df_a, df_b, n_pairs, rng, match_on
        )

    os.makedirs("Results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = output
    else:
        output_path = os.path.join(
            "Results", f"{stamp}_pairwise_{name_a}_vs_{name_b}.csv"
        )

    output_dir = os.path.dirname(str(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    is_json_output = str(output_path).lower().endswith(".json")

    # wins_a / wins_b refer to the input datasets (name_a / name_b),
    # not to the anonymous A/B labels shown to the model.
    wins_a = 0
    wins_b = 0
    ties = 0
    invalid = 0

    metadata = {
        "csv_a": csv_a,
        "csv_b": csv_b,
        "name_a": name_a,
        "name_b": name_b,
        "n_pairs": len(sampled_pairs),
        "match_on": match_on,
        "seed": seed,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "context_words": context_words,
    }

    logger.info(
        f"Judging {len(sampled_pairs)} pairs (match_on={match_on}, model={model}, temperature={temperature})."
    )

    invalid_logged = 0
    invalid_log_limit = len(sampled_pairs) if debug_invalid else 3

    if debug_invalid_dir:
        debug_invalid_dir = str(debug_invalid_dir)
        os.makedirs(debug_invalid_dir, exist_ok=True)

    def _dump_invalid(
        pair_index: int, reason: Optional[str], prompt: str, raw: Optional[str]
    ) -> Optional[str]:
        if not debug_invalid_dir:
            return None

        safe_reason = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(reason or "unknown"))[:80]
        path = os.path.join(
            debug_invalid_dir, f"invalid_pair_{pair_index:04d}_{safe_reason}.txt"
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"pair={pair_index}\n")
                f.write(f"reason={reason}\n")
                f.write("\n--- PROMPT ---\n")
                f.write(prompt)
                f.write("\n\n--- RAW_RESPONSE ---\n")
                f.write(raw or "")
            return path
        except Exception as e:
            logger.warning(f"Failed writing invalid dump for pair {pair_index}: {e}")
            return None

    total_prompt_words = 0
    total_output_words = 0

    if is_json_output:
        # JSON output is buffered in-memory and written at the end.
        matches: List[Dict[str, Any]] = []
        for i, (idx_a, idx_b) in enumerate(sampled_pairs, 1):
            row_a = df_a.loc[idx_a]
            row_b = df_b.loc[idx_b]

            cand_from_csv_a = _row_to_candidate(
                row_a, name_a, idx_a, context_words
            )
            cand_from_csv_b = _row_to_candidate(
                row_b, name_b, idx_b, context_words
            )

            cand_a, cand_b = _randomize_ab(cand_from_csv_a, cand_from_csv_b, rng)

            prompt = PAIRWISE_JUDGE_PROMPT.format(
                lecture_a=cand_a.lecture_name,
                question_a=cand_a.question,
                answer_a=cand_a.answer,
                lecture_b=cand_b.lecture_name,
                question_b=cand_b.question,
                answer_b=cand_b.answer,
            )
            total_prompt_words += _word_count(prompt)

            invalid_reason = None
            raw_response_snippet = None
            try:
                decision, invalid_reason, raw_response = judge_pair(
                    candidate_a=cand_a,
                    candidate_b=cand_b,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if raw_response:
                    total_output_words += _word_count(raw_response)
                if raw_response:
                    raw_response_snippet = (raw_response or "").strip()[:800]
            except Exception as e:
                logger.warning(f"LLM error on pair {i}: {e}")
                invalid_reason = str(e)
                decision = None

            if not decision:
                invalid += 1
                if invalid_logged < invalid_log_limit:
                    invalid_logged += 1
                    logger.warning(
                        "Invalid LLM response on pair %s (%s). Raw (first 1200 chars):\n%s",
                        i,
                        invalid_reason,
                        (raw_response or "").strip()[:1200],
                    )
                dump_path = _dump_invalid(i, invalid_reason, prompt, raw_response)
                if dump_path:
                    logger.warning(f"Wrote invalid debug dump: {dump_path}")
                matches.append(
                    {
                        "pair": i,
                        "winner": None,
                        # Store raw model output (if any) in reasoning for debugging.
                        "reasoning": raw_response_snippet,
                        "a": {
                            "source": cand_a.source,
                            "row_index": cand_a.row_index,
                            "lecture_name": cand_a.lecture_name,
                            "entity_name": cand_a.entity_name,
                            "question": cand_a.question,
                            "answer": cand_a.answer,
                        },
                        "b": {
                            "source": cand_b.source,
                            "row_index": cand_b.row_index,
                            "lecture_name": cand_b.lecture_name,
                            "entity_name": cand_b.entity_name,
                            "question": cand_b.question,
                            "answer": cand_b.answer,
                        },
                        "error": "invalid_or_unparseable_llm_response",
                        "invalid_reason": invalid_reason,
                    }
                )
            else:
                winner = decision["winner"]
                if winner in {"A", "B"}:
                    winner_source = cand_a.source if winner == "A" else cand_b.source
                    if winner_source == name_a:
                        wins_a += 1
                    elif winner_source == name_b:
                        wins_b += 1
                else:
                    ties += 1

                matches.append(
                    {
                        "pair": i,
                        "winner": winner,
                        "reasoning": decision["reasoning"],
                        "a": {
                            "source": cand_a.source,
                            "row_index": cand_a.row_index,
                            "lecture_name": cand_a.lecture_name,
                            "entity_name": cand_a.entity_name,
                            "question": cand_a.question,
                            "answer": cand_a.answer,
                        },
                        "b": {
                            "source": cand_b.source,
                            "row_index": cand_b.row_index,
                            "lecture_name": cand_b.lecture_name,
                            "entity_name": cand_b.entity_name,
                            "question": cand_b.question,
                            "answer": cand_b.answer,
                        },
                    }
                )

            if sleep and sleep > 0:
                time.sleep(sleep)

            if i % 5 == 0 or i == len(sampled_pairs):
                logger.info(
                    f"Processed {i}/{len(sampled_pairs)} | {name_a} wins={wins_a} | {name_b} wins={wins_b} | ties={ties} | invalid={invalid}"
                )

        summary = {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "invalid": invalid,
        }
        result = {
            "metadata": metadata,
            "summary": summary,
            "matches": matches,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Total prompt word count (all pairs): {total_prompt_words}")
        logger.info(f"Saved pairwise results to {output_path}")
        logger.info(f"Saved pairwise results to {output_path}")
        return {"input_words": total_prompt_words, "output_words": total_output_words}

    summary = {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "invalid": invalid,
    }

    # CSV output (default): stream rows as we go so partial runs produce usable files.
    summary_path = re.sub(r"\.csv$", "_summary.csv", output_path, flags=re.IGNORECASE)

    def _write_summary() -> None:
        summary_row = {**metadata, **summary}
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    fieldnames = [
        "pair",
        "winner",
        "reasoning",
        "a_source",
        "a_row_index",
        "a_lecture_name",
        "a_question",
        "a_answer",
        "b_source",
        "b_row_index",
        "b_lecture_name",
        "b_question",
        "b_answer",
        "error",
        "invalid_reason",
    ]

    # Stream match CSV. Also keep a small in-memory list ONLY for JSON mode.
    # Here, for CSV mode, we write each row immediately.
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        f.flush()

        try:
            for i, (idx_a, idx_b) in enumerate(sampled_pairs, 1):
                row_a = df_a.loc[idx_a]
                row_b = df_b.loc[idx_b]

                cand_from_csv_a = _row_to_candidate(
                    row_a, name_a, idx_a, context_words
                )
                cand_from_csv_b = _row_to_candidate(
                    row_b, name_b, idx_b, context_words
                )

                cand_a, cand_b = _randomize_ab(cand_from_csv_a, cand_from_csv_b, rng)

                prompt = PAIRWISE_JUDGE_PROMPT.format(
                    lecture_a=cand_a.lecture_name,
                    question_a=cand_a.question,
                    answer_a=cand_a.answer,
                    lecture_b=cand_b.lecture_name,
                    question_b=cand_b.question,
                    answer_b=cand_b.answer,
                )
                total_prompt_words += _word_count(prompt)

                invalid_reason = None
                raw_response_snippet = None
                try:
                    decision, invalid_reason, raw_response = judge_pair(
                        candidate_a=cand_a,
                        candidate_b=cand_b,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if raw_response:
                        total_output_words += _word_count(raw_response)
                    if raw_response:
                        raw_response_snippet = (raw_response or "").strip()[:800]
                except Exception as e:
                    logger.warning(f"LLM error on pair {i}: {e}")
                    invalid_reason = str(e)
                    decision = None

                if not decision:
                    invalid += 1
                    if invalid_logged < invalid_log_limit:
                        invalid_logged += 1
                        logger.warning(
                            "Invalid LLM response on pair %s (%s). Raw (first 1200 chars):\n%s",
                            i,
                            invalid_reason,
                            (raw_response or "").strip()[:1200],
                        )
                    dump_path = _dump_invalid(i, invalid_reason, prompt, raw_response)
                    if dump_path:
                        logger.warning(f"Wrote invalid debug dump: {dump_path}")
                    match: Dict[str, Any] = {
                        "pair": i,
                        "winner": None,
                        "reasoning": raw_response_snippet,
                        "a": {
                            "source": cand_a.source,
                            "row_index": cand_a.row_index,
                            "lecture_name": cand_a.lecture_name,
                            "entity_name": cand_a.entity_name,
                            "question": cand_a.question,
                            "answer": cand_a.answer,
                        },
                        "b": {
                            "source": cand_b.source,
                            "row_index": cand_b.row_index,
                            "lecture_name": cand_b.lecture_name,
                            "entity_name": cand_b.entity_name,
                            "question": cand_b.question,
                            "answer": cand_b.answer,
                        },
                        "error": "invalid_or_unparseable_llm_response",
                        "invalid_reason": invalid_reason,
                    }
                    winner_ab = None
                else:
                    winner = decision["winner"]
                    if winner in {"A", "B"}:
                        winner_source = cand_a.source if winner == "A" else cand_b.source
                        if winner_source == name_a:
                            wins_a += 1
                        elif winner_source == name_b:
                            wins_b += 1
                    else:
                        ties += 1

                    winner_ab = winner

                    match = {
                        "pair": i,
                        "winner": winner,
                        "reasoning": decision["reasoning"],
                        "a": {
                            "source": cand_a.source,
                            "row_index": cand_a.row_index,
                            "lecture_name": cand_a.lecture_name,
                            "entity_name": cand_a.entity_name,
                            "question": cand_a.question,
                            "answer": cand_a.answer,
                        },
                        "b": {
                            "source": cand_b.source,
                            "row_index": cand_b.row_index,
                            "lecture_name": cand_b.lecture_name,
                            "entity_name": cand_b.entity_name,
                            "question": cand_b.question,
                            "answer": cand_b.answer,
                        },
                    }

                # Write row immediately
                row_dict = {
                    "pair": i,
                    "winner": match.get("winner") if winner_ab == "TIE" else (
                         match["a"]["source"] if winner_ab == "A" else (match["b"]["source"] if winner_ab == "B" else None)
                    ),
                    "reasoning": match.get("reasoning"),
                    "a_source": match["a"].get("source", name_a),
                    "a_row_index": match["a"].get("row_index"),
                    "a_lecture_name": match["a"].get("lecture_name"),
                    "a_question": match["a"].get("question"),
                    "a_answer": match["a"].get("answer"),
                    "b_source": match["b"].get("source", name_b),
                    "b_row_index": match["b"].get("row_index"),
                    "b_lecture_name": match["b"].get("lecture_name"),
                    "b_question": match["b"].get("question"),
                    "b_answer": match["b"].get("answer"),
                    "error": match.get("error"),
                    "invalid_reason": match.get("invalid_reason"),
                }
                # Fix winner field logic above which was slightly broken/complex in dict comprehension:
                # Let's just correct it:
                if winner_ab == "A":
                    row_dict["winner"] = cand_a.source
                elif winner_ab == "B":
                    row_dict["winner"] = cand_b.source
                elif winner_ab == "TIE":
                    row_dict["winner"] = "TIE"
                else:
                    row_dict["winner"] = None

                writer.writerow(row_dict)
                f.flush()

                if sleep and sleep > 0:
                    time.sleep(sleep)

                if i % 5 == 0 or i == len(sampled_pairs):
                    logger.info(
                        f"Processed {i}/{len(sampled_pairs)} | {name_a} wins={wins_a} | {name_b} wins={wins_b} | ties={ties} | invalid={invalid}"
                    )

        except KeyboardInterrupt:
            logger.warning("Interrupted! Saving summary so far...")

    _write_summary()
    print(f"Total prompt word count (all pairs): {total_prompt_words}")
    logger.info(f"Saved pairwise results to {output_path}")
    return {"input_words": total_prompt_words, "output_words": total_output_words}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pairwise evaluation: randomly pits two CSV question-answer datasets against each other and counts wins."
        )
    )
    parser.add_argument("--csv-a", required=True, help="Path to CSV A")
    parser.add_argument("--csv-b", required=True, help="Path to CSV B")
    parser.add_argument("--name-a", default="A", help="Display name for CSV A")
    parser.add_argument("--name-b", default="B", help="Display name for CSV B")
    parser.add_argument(
        "--n-pairs", type=int, default=100, help="Number of random matchups"
    )
    parser.add_argument(
        "--match-on",
        choices=["none", "lecture", "lecture_entity"],
        default="none",
        help="How to sample pairs: none = independent random rows; lecture = same lecture_name; lecture_entity = same lecture_name + entity_name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--context-words",
        type=int,
        default=1200,
        help="Truncate each context to this many words before judging",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="OpenAI-compatible model name served by vLLM",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument(
        "--debug-invalid",
        action="store_true",
        help=(
            "Print detailed debugging info for invalid/unparseable LLM responses. "
            "By default the script only prints details for the first few invalid cases."
        ),
    )
    parser.add_argument(
        "--debug-invalid-dir",
        default=None,
        help=(
            "If set, writes full prompt+raw response for invalid pairs into this directory "
            "(one text file per invalid pair)."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests (useful to reduce load)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. If it ends with .csv, writes CSV (default). If it ends with .json, writes JSON.",
    )

    args = parser.parse_args()
    
    run_pairwise_evaluation(
        csv_a=args.csv_a,
        csv_b=args.csv_b,
        name_a=args.name_a,
        name_b=args.name_b,
        n_pairs=args.n_pairs,
        match_on=args.match_on,
        seed=args.seed,
        context_words=args.context_words,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        debug_invalid=args.debug_invalid,
        debug_invalid_dir=args.debug_invalid_dir,
        sleep=args.sleep,
        output=args.output,
    )

if __name__ == "__main__":
    main()
