import logging
import csv
import argparse
import sys
import os
from datetime import datetime
from typing import List

from mampfsearch.utils.config import get_graph_storage
from mampfsearch.utils.schema import nodeLabels
from mampfsearch.core.entity_selection import find_relevant_entities_in_lecture
from mampfsearch.core.question_generation import (
    create_multiple_segment_spanning_question,
    generate_unstructured_questions,
)

from mampfsearch.utils.models import MathEntityNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_spanning_questions_for_lecture(lecture_name: str, writer, graph_storage):
    logger.info(f"Processing lecture: {lecture_name}")
    try:
        lecture_node = graph_storage.get_lecture_node(name=lecture_name)
        if not lecture_node:
            logger.error(f"Lecture with name '{lecture_name}' not found.")
            return
        lecture_id = lecture_node.graph_id
    except Exception as e:
        logger.error(f"Error fetching lecture node: {e}")
        return

    # Get all segments to preserve spans
    logger.info("Fetching segments...")
    try:
        all_segments = graph_storage.get_segments_of_lecture(lecture_id=lecture_id)
        logger.info(f"Fetched {len(all_segments)} segments for lecture.")
    except Exception as e:
        logger.error(f"Error fetching segments: {e}")
        return

    logger.info("Finding relevant entities...")
    try:
        entities: List[MathEntityNode] = find_relevant_entities_in_lecture(
            lecture_id=lecture_id, top_k=10
        )
        logger.info(f"Found {len(entities)} entities.")
    except Exception as e:
        logger.error(f"Error finding entities: {e}")
        return

    for entity in entities:
        # Skip entities that look like transcription errors (too long)
        if len(entity.name) > 80:
            logger.warning(
                f"Skipping entity with name length {len(entity.name)} (likely transcription error): {entity.name[:50]}..."
            )
            continue

        logger.info(f"Generating questions for entity: {entity.name}")

        try:
            # Pass all segments; function finds spans about entity
            questions = create_multiple_segment_spanning_question(
                all_segments, entity.name
            )

            for q in questions:
                evaluation = q.get("evaluation", {})
                verdict = evaluation.get("verdict") if evaluation else None

                logger.info(f"Question generated. Verdict: {verdict}")

                if verdict == "DISCARD":
                    logger.info("Discarding question due to 'DISCARD' verdict.")
                    continue

                writer.writerow(
                    {
                        "lecture_name": lecture_name,
                        "entity_name": entity.name,
                        "context": q.get("context", ""),
                        "question": q.get("question", ""),
                        "answer": q.get("answer", ""),
                    }
                )
        except Exception as e:
            logger.error(f"Error generating questions for entity {entity.name}: {e}")
            continue


def generate_unstructured_questions_for_lecture(
    lecture_name: str, writer, graph_storage
):
    logger.info(f"Processing lecture (unstructured): {lecture_name}")
    try:
        lecture_node = graph_storage.get_lecture_node(name=lecture_name)
        if not lecture_node:
            logger.error(f"Lecture with name '{lecture_name}' not found.")
            return
        lecture_id = lecture_node.graph_id
    except Exception as e:
        logger.error(f"Error fetching lecture node: {e}")
        return

    # Get all segments to join for full transcript
    logger.info("Fetching segments for transcript...")
    try:
        all_segments_nodes = graph_storage.get_segments_of_lecture(
            lecture_id=lecture_id
        )
        # Sort segments by position
        all_segments_nodes.sort(key=lambda s: s.segment.position)
        full_transcript = "\n".join([s.segment.text for s in all_segments_nodes])
        logger.info(f"Constructed transcript of length {len(full_transcript)}.")
    except Exception as e:
        logger.error(f"Error fetching/constructing transcript: {e}")
        return

    logger.info("Finding relevant entities...")
    try:
        entities: List[MathEntityNode] = find_relevant_entities_in_lecture(
            lecture_id=lecture_id, top_k=10
        )
        logger.info(f"Found {len(entities)} entities.")
    except Exception as e:
        logger.error(f"Error finding entities: {e}")
        return

    for entity in entities:
        if len(entity.name) > 80:
            continue

        logger.info(f"Generating unstructured questions for entity: {entity.name}")
        try:
            questions = generate_unstructured_questions(
                context=full_transcript,
                n_questions=5,
                entity_name=entity.name,
            )

            for q in questions:
                writer.writerow(
                    {
                        "lecture_name": lecture_name,
                        "entity_name": entity.name,
                        "context": "Full Lecture Transcript",
                        "question": q.get("question", ""),
                        "answer": q.get("answer", ""),
                    }
                )
        except Exception as e:
            logger.error(
                f"Error generating unstructured questions for {entity.name}: {e}"
            )
            continue


def main(output_file: str, lecture_name: str = None, mode: str = "spanning"):
    graph_storage = get_graph_storage()

    lectures = []
    if lecture_name:
        lectures = [lecture_name]
    else:
        logger.info("Fetching all lectures from the graph...")
        # Query for all lectures
        cypher = (
            f"MATCH (l:{nodeLabels['lecture']}) RETURN l.name as name ORDER BY l.name"
        )
        try:
            result, _, _ = graph_storage.driver.execute_query(
                cypher, database_=graph_storage.database_name
            )
            lectures = [record["name"] for record in result]
            logger.info(f"Found {len(lectures)} lectures in the database.")
        except Exception as e:
            logger.error(f"Failed to fetch lectures from database: {e}")
            return

    csv_header = ["lecture_name", "entity_name", "context", "question", "answer"]

    # Check if file exists, if so append timestamp to avoid overwriting
    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(output_file)
        output_file = f"{root}_{timestamp}{ext}"

    # Open CSV in append mode if we wanted, but here we overwrite for a clean run
    # If looping, we open once and write many times
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_header, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        total_lectures = len(lectures)
        for i, lec in enumerate(lectures, 1):
            logger.info(f"Processing lecture {i}/{total_lectures}: {lec}")
            if mode == "unstructured":
                generate_unstructured_questions_for_lecture(lec, writer, graph_storage)
            else:
                generate_spanning_questions_for_lecture(lec, writer, graph_storage)

    logger.info(f"Finished generating questions. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate questions for lectures and save to CSV."
    )
    parser.add_argument(
        "--lecture",
        type=str,
        required=False,
        help="Name of a specific lecture to process. If omitted, processes all lectures.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_questions.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="spanning",
        choices=["spanning", "unstructured"],
        help="Question generation mode: 'spanning' (default) or 'unstructured'.",
    )

    args = parser.parse_args()

    main(args.output, args.lecture, args.mode)
