import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Import visualization tools
from mampfsearch.evaluation.visualize_evaluated_data import (
    plot_radar_chart,
    plot_box_chart,
    plot_problem_criteria,
    LABELS_MAP,
    SCORE_COLUMNS,
)

# --- Configuration ---
# Directory containing subdirectories of runs
RESULTS_DIR = "/home/fbuerckel/Mampfsearch/Mampfsearch/src/mampfsearch/evaluation/Aggregated_results_Gemma"

# Output directory for aggregated results (will be created if not exists)
# Defaults to a timestamped folder inside RESULTS_DIR
OUTPUT_DIR = os.path.join(
    RESULTS_DIR, f"aggregated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AggregateEvaluation")


def load_run_data(run_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads the CSV files from a single run directory.
    Returns a dictionary of DataFrames.
    """
    files = {
        "evaluated_multi": "evaluated_multi.csv",
        "evaluated_unstructured": "evaluated_unstructured.csv",
        "pairwise_results": "pairwise_results.csv",
        "generated_multi": "generated_multi.csv",
        "generated_unstructured": "generated_unstructured.csv",
    }

    data = {}
    for key, filename in files.items():
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                data[key] = df
            except Exception as e:
                logger.warning(f"Failed to read {filename} in {run_dir}: {e}")
        else:
            # logger.debug(f"File {filename} not found in {run_dir}")
            pass

    return data


def aggregate_data(
    results_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans results_dir for run subdirectories and aggregates data.
    Returns aggregated DataFrames for:
    - evaluated (combined multi + unstructured with Method col)
    - pairwise
    - generated_multi
    - generated_unstructured
    """
    evaluated_list = []
    pairwise_list = []
    gen_multi_list = []
    gen_unstructured_list = []

    # Walk through immediate subdirectories
    # Note: Structure seems to be results/LectureName/RunTimestamp/
    # The user request said "The folder consists of multiple folder, one for each run".
    # But ls output showed Results/LectureName/RunTimestamp...
    # If RESULTS_DIR is the root "Results", we need to look recursively or
    # the user might point RESULTS_DIR to a specific lecture folder (e.g. Results/Lecture21).
    # I will assume we want to find ALL runs under the given RESULTS_DIR, recursively.

    logger.info(f"Scanning {results_dir} for run data...")

    run_count = 0

    for root, dirs, files in os.walk(results_dir):
        # Check if this folder looks like a run folder (has the CSVs)
        required_files = ["evaluated_multi.csv", "evaluated_unstructured.csv"]
        if all(f in files for f in required_files):
            run_id = os.path.basename(root)
            parent_dir = os.path.basename(os.path.dirname(root))
            # Construct a readable run name, e.g. "Lecture21/RunTimestamp"
            run_name = f"{parent_dir}/{run_id}"

            logger.info(f"Found run: {run_name}")
            data = load_run_data(root)

            if "evaluated_multi" in data:
                df = data["evaluated_multi"].copy()
                df["Run"] = run_name
                df["Method"] = "Proposed (Hybrid)"
                evaluated_list.append(df)

            if "evaluated_unstructured" in data:
                df = data["evaluated_unstructured"].copy()
                df["Run"] = run_name
                df["Method"] = "Baseline (Unstructured)"
                evaluated_list.append(df)

            if "pairwise_results" in data:
                df = data["pairwise_results"].copy()
                df["Run"] = run_name
                pairwise_list.append(df)

            if "generated_multi" in data:
                df = data["generated_multi"].copy()
                df["Run"] = run_name
                gen_multi_list.append(df)

            if "generated_unstructured" in data:
                df = data["generated_unstructured"].copy()
                df["Run"] = run_name
                gen_unstructured_list.append(df)

            run_count += 1

    logger.info(f"aggregated data from {run_count} runs.")

    df_evaluated = (
        pd.concat(evaluated_list, ignore_index=True)
        if evaluated_list
        else pd.DataFrame()
    )
    df_pairwise = (
        pd.concat(pairwise_list, ignore_index=True) if pairwise_list else pd.DataFrame()
    )
    df_gen_multi = (
        pd.concat(gen_multi_list, ignore_index=True)
        if gen_multi_list
        else pd.DataFrame()
    )
    df_gen_unstructured = (
        pd.concat(gen_unstructured_list, ignore_index=True)
        if gen_unstructured_list
        else pd.DataFrame()
    )

    return df_evaluated, df_pairwise, df_gen_multi, df_gen_unstructured


def calculate_statistics(df_evaluated: pd.DataFrame, df_pairwise: pd.DataFrame) -> str:
    stats_output = []

    stats_output.append("=== Aggregation Statistics ===")
    stats_output.append(f"Total Evaluated Questions: {len(df_evaluated)}")

    if not df_evaluated.empty:
        # 1. Counts per method
        counts = df_evaluated["Method"].value_counts()
        stats_output.append("\n-- Distribution by Method --")
        for method, count in counts.items():
            stats_output.append(f"{method}: {count}")

        # 2. Average Scores
        stats_output.append("\n-- Average Scores --")
        # Ensure we only aggregate numeric columns
        numeric_cols = [c for c in SCORE_COLUMNS if c in df_evaluated.columns]
        means = df_evaluated.groupby("Method")[numeric_cols].mean()
        stats_output.append(means.to_string())

        # 3. Outlier Detection (Lectures where one method performed very differently)
        # We assume 'lecture_name' exists in the CSVs (based on header description)
        if "lecture_name" in df_evaluated.columns:
            stats_output.append("\n\n=== Outlier Detection (by Lecture) ===")

            # Pivot to get scores by lecture and method
            # We take the mean score for each lecture/method combination
            lecture_means = (
                df_evaluated.groupby(["lecture_name", "Method"])[numeric_cols]
                .mean()
                .reset_index()
            )

            # We want to find lectures where the deviance from global mean is high
            global_means = df_evaluated.groupby("Method")[numeric_cols].mean()

            # Or perhaps better: Find lectures where 'Overall' score gap between methods is unusual?
            # User said: "Lectures where one method performed very different to the average"
            # This could mean: Lecture X's Baseline score is far from Global Baseline Average.

            for method in df_evaluated["Method"].unique():
                method_global_mean = global_means.loc[method]
                method_lecture_means = lecture_means[lecture_means["Method"] == method]

                stats_output.append(f"\nScanning outliers for {method}...")

                # Check for significant deviation in 'overall_review_score' (if exists) or just average of all scores
                target_col = (
                    "overall_review_score"
                    if "overall_review_score" in numeric_cols
                    else numeric_cols[0]
                )

                global_val = method_global_mean[target_col]
                stats_output.append(f"Global Mean ({target_col}): {global_val:.2f}")

                # Define outlier threshold (e.g., > 1.0 difference)
                for _, row in method_lecture_means.iterrows():
                    lec = row["lecture_name"]
                    val = row[target_col]
                    diff = val - global_val

                    if abs(diff) > 1.0:  # Arbitrary threshold for "very different"
                        stats_output.append(
                            f"  [OUTLIER] {lec}: Score {val:.2f} (Diff: {diff:+.2f})"
                        )

    else:
        stats_output.append("No evaluated data available.")

    # Pairwise Stats
    if not df_pairwise.empty and "winner" in df_pairwise.columns:
        stats_output.append("\n\n=== Pairwise Statistics ===")

        # Clean winners
        df_pairwise["winner_clean"] = df_pairwise["winner"].astype(str).str.strip()
        # Map to standard names if needed, but assuming they are already Multi-Entity/Unstructured/TIE

        winner_counts = df_pairwise["winner_clean"].value_counts()
        stats_output.append(winner_counts.to_string())

        # Win Rate per Lecture Outliers
        if "Run" in df_pairwise.columns:
            stats_output.append("\n\n-- Win Rate Outliers (by Run) --")
            # Calculate win rates per run
            run_groups = df_pairwise.groupby("Run")

            # Global Win Rates
            total_pairs = len(df_pairwise)
            global_multi_win_rate = (
                len(df_pairwise[df_pairwise["winner_clean"] == "Multi-Entity"])
                / total_pairs
                if total_pairs > 0
                else 0
            )
            stats_output.append(
                f"Global Multi-Entity Win Rate: {global_multi_win_rate:.2%}"
            )

            for run_name, group in run_groups:
                n = len(group)
                if n < 5:
                    continue  # Skip small samples

                multi_wins = len(group[group["winner_clean"] == "Multi-Entity"])
                win_rate = multi_wins / n

                # Check for significant deviation (e.g. > 20% diff)
                diff = win_rate - global_multi_win_rate
                if abs(diff) > 0.20:
                    stats_output.append(
                        f"  [OUTLIER] {run_name}: Multi-Entity Win Rate {win_rate:.2%} (Diff: {diff:+.2%})"
                    )

    return "\n".join(stats_output)


def generate_pairwise_win_rate_plot(df: pd.DataFrame, output_dir: str):
    """
    Generates plots for pairwise results, including per-lecture win rates.
    """
    if df.empty or "winner" not in df.columns:
        return

    logger.info("Generating Pairwise Plots...")

    # Clean winner column
    df["winner_clean"] = df["winner"].astype(str).str.strip()

    # 1. Simple Outcomes Bar Chart
    plt.figure(figsize=(8, 6))
    counts = df["winner_clean"].value_counts()
    ax = counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Aggregated Pairwise Outcomes")
    plt.xlabel("Winner")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_pairwise_outcomes.svg"))
    plt.close()

    # 2. Outcomes per Run (Lecture)
    if "Run" in df.columns:
        # Create crosstab
        ct = pd.crosstab(df["Run"], df["winner_clean"], normalize="index") * 100

        # Sort by Multi-Entity win rate if present
        if "Multi-Entity" in ct.columns:
            ct = ct.sort_values("Multi-Entity", ascending=True)

        plt.figure(figsize=(10, max(6, len(ct) * 0.4)))
        ct.plot(
            kind="barh",
            stacked=True,
            colormap="coolwarm",
            edgecolor="black",
            figsize=(10, max(6, len(ct) * 0.4)),
        )
        plt.title("Pairwise Winner Distribution per Run")
        plt.xlabel("Percentage")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pairwise_outcomes_per_run.svg"))
        plt.close()


def main():
    if not os.path.exists(RESULTS_DIR):
        logger.error(f"Results directory not found: {RESULTS_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # 1. Aggregate
    df_eval, df_pair, df_gen_multi, df_gen_unstructured = aggregate_data(RESULTS_DIR)

    if df_eval.empty:
        logger.warning("No evaluation data found. Exiting.")
        return

    # 2. Save Aggregated CSVs
    logger.info("Saving aggregated CSVs...")
    df_eval.to_csv(os.path.join(OUTPUT_DIR, "aggregated_evaluated.csv"), index=False)

    # New: Save Aggregated by Lecture
    if not df_eval.empty and "lecture_name" in df_eval.columns:
        numeric_cols = [c for c in SCORE_COLUMNS if c in df_eval.columns]
        if numeric_cols:
            df_lecture = (
                df_eval.groupby(["lecture_name", "Method"])[numeric_cols]
                .mean()
                .reset_index()
            )
            df_lecture.to_csv(
                os.path.join(OUTPUT_DIR, "aggregated_by_lecture.csv"), index=False
            )
            logger.info("Saved aggregated_by_lecture.csv")

    if not df_pair.empty:
        df_pair.to_csv(os.path.join(OUTPUT_DIR, "aggregated_pairwise.csv"), index=False)
    if not df_gen_multi.empty:
        df_gen_multi.to_csv(
            os.path.join(OUTPUT_DIR, "aggregated_generated_multi.csv"), index=False
        )
    if not df_gen_unstructured.empty:
        df_gen_unstructured.to_csv(
            os.path.join(OUTPUT_DIR, "aggregated_generated_unstructured.csv"),
            index=False,
        )

    # 3. Statistics
    logger.info("Calculating statistics...")
    stats_text = calculate_statistics(df_eval, df_pair)
    with open(os.path.join(OUTPUT_DIR, "statistics.txt"), "w") as f:
        f.write(stats_text)
    print(stats_text)

    # 4. Plots
    logger.info("Generating plots...")
    # Using existing visualization logic
    try:
        # The existing functions expect a dataframe with "Method" and score columns.
        # We have exactly that in df_eval.
        plot_radar_chart(df_eval, OUTPUT_DIR, "Gemma 3", "Gemma 3")
        plot_box_chart(df_eval, OUTPUT_DIR, "Gemma 3", "Gemma 3")
        plot_problem_criteria(df_eval, OUTPUT_DIR, "Gemma 3", "Gemma 3")

        # Custom pairwise plots
        generate_pairwise_win_rate_plot(df_pair, OUTPUT_DIR)

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
