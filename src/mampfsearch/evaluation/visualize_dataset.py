import logging
import matplotlib.pyplot as plt
import collections
import numpy as np
from mampfsearch.utils import config
from mampfsearch.utils.schema import nodeLabels, relationships

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from classification
TYPES = ["Definition", "Theorem", "Proof", "Example", "Other"]
PREFIXES = ["B", "I"]
ALL_CLASSIFICATION_LABELS = {f"{p}-{t}" for p in PREFIXES for t in TYPES}


def fetch_segment_data():
    """Fetches segment data from Neo4j."""
    graph = config.get_graph_storage()
    driver = graph.driver

    # Order by lecture and position to detect blocks
    query = f"""
    MATCH (l:{nodeLabels["lecture"]})-[:{relationships["has_segment"]}]->(s:{nodeLabels["segment"]})
    RETURN l.id as lecture_id, s.position as position, s.id as id, labels(s) as labels, s.text as text
    ORDER BY l.id, s.position
    """

    data = []
    with driver.session(database=config.NEO4J_DATABASE_NAME) as session:
        result = session.run(query)
        for record in result:
            data.append(
                {
                    "lecture_id": record["lecture_id"],
                    "position": record["position"],
                    "id": record["id"],
                    "labels": record["labels"],
                    "text": record["text"],
                }
            )

    driver.close()
    return data


def analyze_and_visualize(data):
    """Analyzes data and produces visualizations."""

    # Increase font sizes for better readability in thesis
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    # storage for stats
    label_counts = collections.Counter()
    segment_type_counts = (
        collections.Counter()
    )  # Counts occurrences on segments (individual)

    # Block storage: Maps type -> list of lengths for each distinct block
    block_lengths = collections.defaultdict(list)

    # State tracking: Maps base_type -> current_accumulated_length
    active_blocks = {}

    previous_lecture_id = None
    all_lengths = []
    unclassified_count = 0

    for item in data:
        lecture_id = item["lecture_id"]
        labels = set(item["labels"])
        text = item["text"] if item["text"] else ""
        text_len = len(text.split())
        all_lengths.append(text_len)

        # 1. Handle Lecture Change
        if lecture_id != previous_lecture_id:
            # End all active blocks from previous lecture
            for b_type, length in active_blocks.items():
                block_lengths[b_type].append(length)
            active_blocks = {}
            previous_lecture_id = lecture_id

        # 2. Identify classification labels on THIS segment
        classification_labels_on_segment = [
            l for l in labels if l in ALL_CLASSIFICATION_LABELS
        ]

        # Helper to track what we processed this turn (to identify what ends)
        processed_types_this_segment = set()

        if classification_labels_on_segment:
            for label in classification_labels_on_segment:
                label_counts[label] += 1
                prefix, base_type = label.split("-")
                segment_type_counts[base_type] += 1
                processed_types_this_segment.add(base_type)

                if prefix == "B":
                    # Explicit start of a block.
                    # If this type was already active, the previous block ENDS here.
                    if base_type in active_blocks:
                        block_lengths[base_type].append(active_blocks[base_type])

                    # Start new block
                    active_blocks[base_type] = text_len

                elif prefix == "I":
                    if base_type in active_blocks:
                        # Continue block
                        active_blocks[base_type] += text_len
                    else:
                        # "I" without preceding block -> Treat as new block start
                        active_blocks[base_type] = text_len
        else:
            unclassified_count += 1

        # 3. Check for blocks that ended naturally (were active, but NOT present in this segment)
        types_to_end = [
            t for t in active_blocks if t not in processed_types_this_segment
        ]
        for b_type in types_to_end:
            block_lengths[b_type].append(active_blocks[b_type])
            del active_blocks[b_type]

    # End any remaining blocks after loop
    for b_type, length in active_blocks.items():
        block_lengths[b_type].append(length)

    # Calculate block counts from lengths
    block_counts = {k: len(v) for k, v in block_lengths.items()}

    # Statistics Reporting
    total_segments = len(data)
    avg_segment_len = np.mean(all_lengths) if all_lengths else 0

    logger.info("=" * 30)
    logger.info("GENERAL STATISTICS")
    logger.info(f"Total Segments: {total_segments}")
    logger.info(f"Average Segment Length: {avg_segment_len:.2f} words")
    logger.info("=" * 30)

    logger.info("BLOCK STATISTICS (Consecutive Sequences)")
    for t, lengths in block_lengths.items():
        avg_len = np.mean(lengths) if lengths else 0
        median_len = np.median(lengths) if lengths else 0
        min_len = np.min(lengths) if lengths else 0
        max_len = np.max(lengths) if lengths else 0
        count = len(lengths)
        logger.info(
            f"  {t}: {count} blocks. Avg: {avg_len:.1f}. Median: {median_len:.1f}. Range: [{min_len}, {max_len}]"
        )
    logger.info("=" * 30)

    logger.info("CLASSIFICATION STATISTICS (Per Segment)")
    logger.info(f"Total Classified Occurrences: {sum(label_counts.values())}")
    logger.info(f"Unclassified Segments: {unclassified_count}")
    for t, c in segment_type_counts.most_common():
        logger.info(f"  {t}: {c}")

    logger.info("=" * 30)

    # Visualizations

    # 1. Plot Distribution of Specific Labels (BIO)
    plt.figure(figsize=(12, 6))
    sorted_labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in sorted_labels]

    plt.bar(sorted_labels, counts, color="skyblue")
    plt.title("Distribution of Specific Segment Labels (BIO)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("segment_labels_distribution.png")
    plt.savefig("segment_labels_distribution.svg")
    logger.info("Saved segment_labels_distribution.png and .svg")

    # 2. Plot Distribution of Base Types (Individual Segments)
    plt.figure(figsize=(10, 6))
    sorted_types = sorted(segment_type_counts.keys())
    t_counts = [segment_type_counts[t] for t in sorted_types]

    # Handle Unclassified in type plot if significant?
    if unclassified_count > 0:
        sorted_types.append("Unclassified")
        t_counts.append(unclassified_count)

    plt.bar(sorted_types, t_counts, color="lightgreen")
    plt.title("Distribution of Segment Types (Individual Segments)")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("segment_types_distribution.png")
    plt.savefig("segment_types_distribution.svg")
    logger.info("Saved segment_types_distribution.png and .svg")

    # 2b. Plot Distribution of BLOCKS
    plt.figure(figsize=(10, 6))
    sorted_block_types = sorted(block_counts.keys())
    b_counts = [block_counts[t] for t in sorted_block_types]

    plt.bar(sorted_block_types, b_counts, color="salmon")
    plt.title("Distribution of Segment Blocks")
    plt.xlabel("Type")
    plt.ylabel("Count (Blocks)")
    plt.tight_layout()
    plt.savefig("segment_blocks_distribution.png")
    plt.savefig("segment_blocks_distribution.svg")
    logger.info("Saved segment_blocks_distribution.png and .svg")

    # 3. Boxplot of Block Word Counts
    plt.figure(figsize=(12, 8))

    plot_data = []
    plot_labels = []

    sorted_keys = sorted(block_lengths.keys())
    for t in sorted_keys:
        if block_lengths[t]:
            plot_data.append(block_lengths[t])
            # Label with name and count (n)
            plot_labels.append(f"{t}\n(n={len(block_lengths[t])})")

    if plot_data:
        # Create boxplot
        plt.boxplot(plot_data, labels=plot_labels)
        plt.title("Word Count Distribution per Block Type")
        plt.ylabel("Block Word Count (Log Scale)")
        plt.xlabel("Type")
        plt.yscale("log")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("segment_lengths_boxplot.png")
        plt.savefig("segment_lengths_boxplot.svg")
        logger.info("Saved segment_lengths_boxplot.png and .svg")
    else:
        logger.warning("No block data to plot for boxplot")

    # 4. Pie Chart for Types
    plt.figure(figsize=(8, 8))
    plt.pie(t_counts, labels=sorted_types, autopct="%1.1f%%", startangle=140)
    plt.title("Segment Type Proportions")
    plt.tight_layout()
    plt.savefig("segment_types_pie.png")
    plt.savefig("segment_types_pie.svg")
    logger.info("Saved segment_types_pie.png and .svg")


def main():
    try:
        import matplotlib

        logger.info(f"Using matplotlib version {matplotlib.__version__}")
    except ImportError:
        logger.error(
            "Matplotlib is not installed. Please install it with 'pip install matplotlib'."
        )
        return

    logger.info("Fetching data...")
    data = fetch_segment_data()

    if not data:
        logger.warning("No segments found in the database.")
        return

    logger.info("Analyzing and visualizing...")
    analyze_and_visualize(data)


if __name__ == "__main__":
    main()
