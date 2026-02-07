import argparse
import logging
import os
import csv
from datetime import datetime
from mampfsearch.utils.config import get_graph_storage

# Import from sibling scripts
from mampfsearch.evaluation.generate_questions_csv import (
    generate_unstructured_questions_for_lecture,
    generate_multi_entity_spanning_questions_for_lecture,
)
from mampfsearch.evaluation.evaluate_question_answer_pair import evaluate_dataset
from mampfsearch.evaluation.visualize_evaluated_data import generate_evaluation_plots
from mampfsearch.evaluation.pairwise_evaluation import run_pairwise_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvaluationPipeline")


def run_pipeline(
    lecture_name: str,
    model_name: str,
    sentences_per_chunk: int = 10,
    questions_per_chunk: int = 5,
    max_definition_words: int = 500,
    max_comention_words: int = 1500,
    max_comention_entities: int = 5,
    base_results_dir: str = "Results",
):
    # 1. Setup Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize names for path safety
    safe_lec_name = lecture_name.replace(" ", "_").replace("/", "_")
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")

    run_dir = os.path.join(
        base_results_dir, safe_lec_name, f"{safe_model_name}_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    logger.info(
        f"Starting evaluation pipeline for lecture '{lecture_name}' with model '{model_name}'"
    )
    logger.info(f"Results will be stored in: {run_dir}")

    # 2. Write Run Info
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Lecture Name: {lecture_name}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Sentences Per Chunk: {sentences_per_chunk}\n")
        f.write(f"Questions Per Chunk: {questions_per_chunk}\n")
        f.write(f"Max Definition Words: {max_definition_words}\n")
        f.write(f"Max Comention Words: {max_comention_words}\n")
        f.write(f"Max Comention Entities: {max_comention_entities}\n")

    # 3. Initialize Graph Storage
    graph_storage = get_graph_storage()

    # 4. Generate Questions

    # Define file paths
    gen_unstructured_csv = os.path.join(run_dir, "generated_unstructured.csv")
    gen_multi_csv = os.path.join(run_dir, "generated_multi.csv")

    csv_header = ["lecture_name", "entity_name", "context", "question", "answer"]

    # 4a. Multi-Entity Generation
    logger.info("Step 1/5: Generating Multi-Entity Questions...")
    with open(gen_multi_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_header, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        stats_qg_multi = generate_multi_entity_spanning_questions_for_lecture(
            lecture_name=lecture_name,
            writer=writer,
            graph_storage=graph_storage,
            max_definition_words=max_definition_words,
            max_comention_words=max_comention_words,
            max_comention_entities=max_comention_entities,
        )

    # 4b. Unstructured Generation
    logger.info("Step 2/5: Generating Unstructured Questions...")
    with open(gen_unstructured_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_header, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        stats_qg_unstructured = generate_unstructured_questions_for_lecture(
            lecture_name=lecture_name,
            writer=writer,
            graph_storage=graph_storage,
            sentences_per_chunk=sentences_per_chunk,
            questions_per_chunk=questions_per_chunk,
        )

    # 5. Evaluate Individually

    eval_unstructured_csv = os.path.join(run_dir, "evaluated_unstructured.csv")
    eval_multi_csv = os.path.join(run_dir, "evaluated_multi.csv")

    # 5a. Evaluate Multi-Entity
    logger.info("Step 3/5: Evaluating Multi-Entity Questions...")
    stats_eval_multi = evaluate_dataset(gen_multi_csv, eval_multi_csv)

    # 5b. Evaluate Unstructured
    logger.info("Step 4/5: Evaluating Unstructured Questions...")
    stats_eval_unstructured = evaluate_dataset(gen_unstructured_csv, eval_unstructured_csv)

    # 6. Visualize Results
    logger.info("Step 5/5: Generating Visualization Plots...")
    # Using defaults or passed model name for labels
    generate_evaluation_plots(
        baseline_csv=eval_unstructured_csv,
        proposed_csv=eval_multi_csv,
        output_dir=run_dir,
        qg_model=model_name,
        eval_model="gpt-oss-20b",  # Assuming evaluation is done by the default model in evaluate_dataset
    )

    # 7. Pairwise Evaluation
    logger.info("Step 6/6: Running Pairwise Evaluation...")
    pairwise_csv = os.path.join(run_dir, "pairwise_results.csv")

    # We compare Unstructured (A) vs Multi-Entity (B)
    # We compare Unstructured (A) vs Multi-Entity (B)
    stats_pairwise = run_pairwise_evaluation(
        csv_a=gen_unstructured_csv,  # Evaluating Q/A pairs quality directly, usually on generated files before distinct eval or after?
        # Pairwise usually takes the raw Q/A, which are in the generated files.
        csv_b=gen_multi_csv,
        name_a="Unstructured",
        name_b="Multi-Entity",
        output=pairwise_csv,
        n_pairs=150,  # Defaulting to 50 for speed, user can adjust if we added arg
        model="openai/gpt-oss-20b",  # Default judge model
    )

    logger.info("Pipeline Completed Successfully!")
    
    # Append stats to run_info.txt
    try:
        with open(info_path, "a") as f:
            f.write("\n--- LLM Usage Statistics ---\n")
            f.write(f"QG Multi-Entity: Input={stats_qg_multi.get('input_words', 0)}, Output={stats_qg_multi.get('output_words', 0)}\n")
            f.write(f"QG Unstructured: Input={stats_qg_unstructured.get('input_words', 0)}, Output={stats_qg_unstructured.get('output_words', 0)}\n")
            f.write(f"Eval Multi-Entity: Input={stats_eval_multi.get('input_words', 0)}, Output={stats_eval_multi.get('output_words', 0)}\n")
            f.write(f"Eval Unstructured: Input={stats_eval_unstructured.get('input_words', 0)}, Output={stats_eval_unstructured.get('output_words', 0)}\n")
            f.write(f"Pairwise Eval: Input={stats_pairwise.get('input_words', 0)}, Output={stats_pairwise.get('output_words', 0)}\n")
            
            # Total
            total_input = (
                stats_qg_multi.get('input_words', 0) + 
                stats_qg_unstructured.get('input_words', 0) + 
                stats_eval_multi.get('input_words', 0) + 
                stats_eval_unstructured.get('input_words', 0) + 
                stats_pairwise.get('input_words', 0)
            )
            total_output = (
                stats_qg_multi.get('output_words', 0) + 
                stats_qg_unstructured.get('output_words', 0) + 
                stats_eval_multi.get('output_words', 0) + 
                stats_eval_unstructured.get('output_words', 0) + 
                stats_pairwise.get('output_words', 0)
            )
            f.write(f"TOTAL: Input={total_input}, Output={total_output}\n")
            
        logger.info(f"Appended LLM usage stats to {info_path}")
    except Exception as e:
        logger.error(f"Failed to append stats to run_info.txt: {e}")

    logger.info("Pipeline Completed Successfully!")
    logger.info(f"All results are available in: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for a lecture."
    )
    parser.add_argument(
        "--lecture_name", required=True, help="Name of the lecture to evaluate."
    )
    parser.add_argument(
        "--model_name", required=True, help="Name of the model being tested."
    )
    parser.add_argument(
        "--sentences_per_chunk",
        type=int,
        default=10,
        help="Sentences per chunk for unstructured gen.",
    )
    parser.add_argument(
        "--questions_per_chunk",
        type=int,
        default=5,
        help="Questions per chunk for unstructured gen.",
    )
    parser.add_argument(
        "--max_definition_words",
        type=int,
        default=500,
        help="Max words for definition blocks.",
    )
    parser.add_argument(
        "--max_comention_words",
        type=int,
        default=1500,
        help="Max words for co-mention blocks.",
    )
    parser.add_argument(
        "--max_comention_entities",
        type=int,
        default=5,
        help="Max number of co-mentioned entities to include.",
    )

    args = parser.parse_args()

    run_pipeline(
        lecture_name=args.lecture_name,
        model_name=args.model_name,
        sentences_per_chunk=args.sentences_per_chunk,
        questions_per_chunk=args.questions_per_chunk,
        max_definition_words=args.max_definition_words,
        max_comention_words=args.max_comention_words,
        max_comention_entities=args.max_comention_entities,
    )
