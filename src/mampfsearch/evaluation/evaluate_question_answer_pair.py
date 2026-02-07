from mampfsearch.utils import prompts
from mampfsearch.utils.config import get_llm_client
import pandas as pd
import json
import time
from typing import Dict, Any, Optional
import re
import logging
import os

# --- Configuration ---
# INPUT_CSV = "generated_unstructured_questions_20260205_093011.csv"
# generated_unstructured_questions_gemma_2qpc.csv
# INPUT_CSV = "Results/Lecture21/gpt-oss-5-feb/generated_multi_questions_21.csv"
# OUTPUT_CSV = "Results/Lecture21/gpt-oss-5-feb/evaluated_multi_questions_gemma.csv"

INPUT_CSV = "generated_multi_questions.csv"
OUTPUT_CSV = "evaluated_multi_questions.csv"

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are a careful assistant. Follow the instructions exactly and respond with only the requested JSON."

EVALUATION_CRITERIA = (
    "clarity",
    "conciseness",
    "relevance",
    "consistency",
    "answerability",
    "answer_consistency",
    "educational_complexity",
    "independence",
    "overall_review",
)


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


def evaluate_dataset(input_file: str, output_file: str):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("CSV file not found.")
        return

    print(f"Starting evaluation of {len(df)} rows with QG-Eval criteria...")

    # Precompute a stable output schema so we can append in batches.
    output_columns = list(df.columns)
    for criteria_name in EVALUATION_CRITERIA:
        output_columns.append(f"{criteria_name}_score")
        output_columns.append(f"{criteria_name}_reason")
    output_columns.append("eval_error")

    # Start fresh by removing an existing output file.
    if os.path.exists(output_file):
        os.remove(output_file)

    buffer = []
    wrote_header = False

    def flush_buffer():
        nonlocal buffer, wrote_header
        if not buffer:
            return
        batch_df = pd.DataFrame(buffer, columns=output_columns)
        batch_df.to_csv(
            output_file,
            mode="a",
            index=False,
            header=not wrote_header,
        )
        wrote_header = True
        buffer = []

    for index, row in df.iterrows():
        prompt = prompts.EVALUATION_PROMPT.format(
            context=row.get("context", ""),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
        )

        llm_client = get_llm_client()

        try:
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
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
            eval_data = parse_llm_response(
                response=content,
                keys=EVALUATION_CRITERIA,
            )

            row_result = row.to_dict()
            if eval_data:
                # Flatten the nested JSON structure into CSV columns
                for criteria_name, details in eval_data.items():
                    row_result[f"{criteria_name}_score"] = details.get("score")
                    row_result[f"{criteria_name}_reason"] = details.get("reasoning")
            else:
                row_result["eval_error"] = "JSON Parsing Failed"

            buffer.append(row_result)
        except Exception as e:
            print(f"Error evaluating row {index}: {e}")
            row_result = row.to_dict()
            row_result["eval_error"] = str(e)
            buffer.append(row_result)

        if len(buffer) >= 5:
            flush_buffer()

        if (index + 1) % 5 == 0:
            print(f"Processed {index + 1}/{len(df)}...")

    flush_buffer()
    print(f"Evaluation complete. Saved to {output_file}")


if __name__ == "__main__":
    evaluate_dataset(INPUT_CSV, OUTPUT_CSV)
