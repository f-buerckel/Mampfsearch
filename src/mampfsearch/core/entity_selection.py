from typing import List, Dict, Optional

from mampfsearch.utils.models import SegmentNode, MathEntityNode
from mampfsearch.utils.config import get_graph_storage, SELECTION_WEIGHTS


def find_relevant_entities_in_lecture(
    lecture_id: Optional[str] = None, lecture_name: Optional[str] = None
) -> List[MathEntityNode]:
    graph_storage = get_graph_storage()

    segmentNodes = graph_storage.get_segments_of_lecture(
        lecture_id=lecture_id, lecture_name=lecture_name
    )

    relevancy: Dict[str, float] = {}

    statistics = graph_storage.get_statistics(segmentNodes)
    for entity_name, stats in statistics.items():
        density_ratio = stats["local_density"] / stats["global_density"]
        tf = stats["term_frequency"]
        pr = stats["pagerank_score"]
        relevance_score = (
            SELECTION_WEIGHTS["density_ratio"] * density_ratio
            + SELECTION_WEIGHTS["term_frequency"] * tf
            + SELECTION_WEIGHTS["pagerank_score"] * pr
        )
        relevancy[entity_name] = relevance_score

    for name, relevance in dict(
        sorted(relevancy.items(), key=lambda item: item[1], reverse=True)
    ).items():
        print(f"Entity: {name}, Relevance Score: {relevance:.4f}")
        print(statistics[name])
