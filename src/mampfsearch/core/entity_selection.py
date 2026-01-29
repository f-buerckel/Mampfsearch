from typing import List, Dict, Optional, Any

from mampfsearch.utils.models import SegmentNode, MathEntityNode
from mampfsearch.utils.config import get_graph_storage, SELECTION_WEIGHTS

import logging

logger = logging.getLogger(__name__)


def calculate_entity_relevance(stats: Dict[str, Any]) -> float:
    density_ratio = stats["local_density"] / stats["global_density"]
    tf = stats["term_frequency"]
    pr = stats["pagerank_score"]

    return (
        SELECTION_WEIGHTS["density_ratio"] * density_ratio
        + SELECTION_WEIGHTS["term_frequency"] * tf
        + SELECTION_WEIGHTS["pagerank_score"] * pr
    )


def find_relevant_entities_in_lecture(
    lecture_id: Optional[str] = None,
    lecture_name: Optional[str] = None,
    top_k: Optional[int] = 10,
) -> List[MathEntityNode]:
    graph_storage = get_graph_storage()

    segmentNodes = graph_storage.get_segments_of_lecture(
        lecture_id=lecture_id, lecture_name=lecture_name
    )

    relevancy: Dict[str, float] = {}

    statistics = graph_storage.get_statistics(segmentNodes)
    for entity_name, stats in statistics.items():
        relevance_score = calculate_entity_relevance(stats)
        relevancy[entity_name] = relevance_score

    sorted_entities: List[MathEntityNode] = []
    for name, relevance in sorted(
        relevancy.items(), key=lambda item: item[1], reverse=True
    ):
        entity_stats = statistics.get(name, {})
        entity_id = entity_stats.get("entity_id")
        if not entity_id:
            logger.debug(
                "Skipping entity '%s' because statistics are missing 'entity_id': %s",
                name,
                entity_stats,
            )
            continue

        # ugly but works.
        node = graph_storage.get_entity_node(id=entity_id)
        sorted_entities.append(node)
        logger.debug("Entity: %s, Relevance Score: %.4f", name, relevance)
        logger.debug(entity_stats)

    return sorted_entities[:top_k]
