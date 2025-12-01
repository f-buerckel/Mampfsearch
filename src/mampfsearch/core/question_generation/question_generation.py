from mampfsearch.utils.models import LectureNode, MathEntityNode, SegmentNode
from mampfsearch.utils import config

from typing import List
import logging

logger = logging.getLogger(__name__)


class QuestionGenerator:
    def __init__(self):
        self.graph_storage = config.get_graph_storage()

    def find_relevant_entities(
        self, segmentNodes: List[SegmentNode]
    ) -> list[MathEntityNode]:
        local_and_global_density = self.graph_storage.get_local_and_global_density(
            segmentNodes
        )
        for entity_id, densities in local_and_global_density.items():
            density_ratio = densities["local_density"] / densities["global_density"]
            logger.info(
                f"Entity ID: {entity_id}, Local Density: {densities['local_density']}, Global Density: {densities['global_density']}, Density Ratio: {density_ratio}"
            )
        return []


if __name__ == "__main__":
    graph_storage = config.get_graph_storage()
    qg = QuestionGenerator()
    segmentNodes = graph_storage.get_segments_of_lecture(lecture_name="Lecture3")
    qg.find_relevant_entities(segmentNodes)
