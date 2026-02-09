import os
import sys
import logging

from mampfsearch.utils import config
from mampfsearch.utils.schema import nodeLabels, relationships

# TODO:Improve nested classifications. For example a Definition can be inside a Proof and both should be considered.

logger = logging.getLogger(__name__)

# Define the classification labels
TYPES = ["Definition", "Theorem", "Proof", "Example", "Other"]
PREFIXES = ["B", "I"]
# Generate all possible labels like B-Definition, I-Definition, etc.
ALL_LABELS = {f"{p}-{t}" for p in PREFIXES for t in TYPES}


def classify_segments():
    graph = config.get_graph_storage()
    driver = graph.driver

    # 2. Setup LLM Client
    llm_client = config.get_llm_client()

    logger.info("Starting classification script...")

    with driver.session(database=config.NEO4J_DATABASE_NAME) as session:
        # Get all lectures
        logger.info("Fetching all lectures...")
        result = session.run(
            f"MATCH (l:{nodeLabels['lecture']}) RETURN l.id as id ORDER BY l.id"
        )
        lecture_ids = [record["id"] for record in result]

        logger.info(f"Found {len(lecture_ids)} lectures.")

        for lecture_id in lecture_ids:
            logger.info(f"Processing Lecture: {lecture_id}")

            # Retrieve segments for the lecture ordered by position
            # Retrieving explicit labels to check for existence
            query_segments = f"""
            MATCH (l:{nodeLabels["lecture"]} {{id: $lecture_id}})-[:{relationships["has_segment"]}]->(s:{nodeLabels["segment"]})
            RETURN s.id as id, s.text as text, labels(s) as labels
            ORDER BY s.position ASC
            """

            segments = list(session.run(query_segments, lecture_id=lecture_id))
            logger.info(f"  Found {len(segments)} segments.")

            prev_segment_text = None
            prev_segment_label = None

            for segment in segments:
                seg_id = segment["id"]
                text = segment["text"]
                current_labels = set(segment["labels"])

                # Check if already classified with any of our relevant labels
                existing_label = None
                for label in current_labels:
                    # Check if label matches B-Type or I-Type
                    if label in ALL_LABELS:
                        existing_label = label
                        break

                if existing_label:
                    logger.debug(
                        f"  Skipping {seg_id}: Already has label {existing_label}"
                    )
                    # Update context for next segment
                    prev_segment_text = text
                    prev_segment_label = existing_label
                    continue

                # Prepare prompt
                system_prompt = (
                    "You are an expert mathematician and classifier of mathematical lecture notes. "
                    "Your task is to classify the current text segment into one of the following categories: "
                    f"{', '.join(TYPES)}. "
                    "You must also assign a BIO tag: 'B' if it is the beginning of a new section of that type, "
                    "or 'I' if it is a continuation of the previous section of the SAME type.\n"
                    "Output ONLY the combined label (e.g., 'B-Definition', 'I-Proof')."
                )

                user_content = f"Current Segment Text:\n{text}\n\n"

                if prev_segment_text:
                    user_content += f"Previous Segment Text:\n{prev_segment_text}\n"
                    if prev_segment_label:
                        user_content += (
                            f"Previous Segment Label: {prev_segment_label}\n"
                        )
                else:
                    user_content += "Previous Segment: None (Start of Lecture)\n"

                user_content += "\nClassify the Current Segment:"

                try:
                    response = llm_client.chat.completions.create(
                        model=config.LLM_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.0,
                    )

                    predicted_label = response.choices[0].message.content.strip()

                    # Validate and clean label
                    final_label = None
                    if predicted_label in ALL_LABELS:
                        final_label = predicted_label
                    else:
                        # Attempt to find valid label in output if it was chatty
                        for l in ALL_LABELS:
                            if l in predicted_label:
                                final_label = l
                                break

                    if not final_label:
                        logger.warning(
                            f"  LLM returned invalid label '{predicted_label}' for segment {seg_id}. Defaulting to B-Other."
                        )
                        final_label = "B-Other"

                    logger.info(f"  Classified {seg_id} -> {final_label}")

                    # Update Neo4j
                    session.run(
                        f"MATCH (s:{nodeLabels['segment']} {{id: $id}}) SET s:`{final_label}`",
                        id=seg_id,
                    )

                    prev_segment_label = final_label

                except Exception as e:
                    logger.error(f"  Error classifying segment {seg_id}: {e}")
                    # On error, don't update DB. Assume context is lost or default to Other for context?
                    # Safer to just keep text but no label for context.
                    prev_segment_label = None

                prev_segment_text = text

    driver.close()
    logger.info("Classification complete.")


if __name__ == "__main__":
    classify_segments()
