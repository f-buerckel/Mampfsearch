from mampfsearch.utils.config import get_graph_storage
from mampfsearch.core.entity_selection import find_relevant_entities_in_lecture

import logging

logger = logging.getLogger(__name__)


def generate_questions_about_single_entity(entity_id, lecture_id):
    graph_storage = get_graph_storage()

    segments = graph_storage.get_segments_containing_entity(
        entity_id=entity_id, lecture_id=lecture_id
    )
