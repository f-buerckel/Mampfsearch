import logging

from mampfsearch.utils import config
from mampfsearch.utils import config

logger = logging.getLogger(__name__)


def init():
    """Initialize the collection for lectures"""
    lectures_info = create_lectures_collection()
    entities_info = create_entities_collection()
    logger.info("Collection initialization completed")
    return {
        "collections": [lectures_info, entities_info],
    }


def create_lectures_collection():
    client = config.get_vector_storage()
    name = config.LECTURE_COLLECTION_NAME
    dimension = config.EMBEDDING_DIMENSION
    return client.create_lecture_collection(name, dimension)


def create_entities_collection():
    client = config.get_vector_storage()
    name = config.ENTITIES_COLLECTION_NAME
    dimension = config.EMBEDDING_DIMENSION
    return client.create_entity_collection(name, dimension)
