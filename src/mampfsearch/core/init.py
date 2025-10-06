import logging

from mampfsearch.utils import config
from qdrant_client import models

logger = logging.getLogger(__name__)

def init():
    """Initialize the collection for lectures"""
    lectures_info = create_lectures_collection()
    entities_info = create_entities_collection()
    logger.info("Collection initialization completed")
    return {
        "collections": [lectures_info, entities_info]
    }

def create_lectures_collection():
    client = config.get_qdrant_client()

    name = config.LECTURE_COLLECTION_NAME
    exists = client.collection_exists(name)

    info = {
        "collection_name": name,
        "exists": exists,
    }

    if exists:
        logger.info(f"Collection {name} already exists")
        return info
    
    dimension=config.EMBEDDING_DIMENSION
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": 
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                ),
            "colbert":
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams()
            )
        }
    )
    
    logger.info(f"Created collection {name} (vector dimension={dimension})")

    info.update({
        "status": "exists",
        "vector_dimension": dimension,
    })
    return info

def create_entities_collection():
    client = config.get_qdrant_client()

    name = config.ENTITIES_COLLECTION_NAME
    exists = client.collection_exists(name)

    info = {
        "collection_name": name,
        "exists": exists,
    }


    if exists:
        logger.info(f"Collection {name} already exists")
        return info
    
    dimension=config.EMBEDDING_DIMENSION
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": 
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                ),
            "colbert":
                models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams()
            )
        }
    )
    
    logger.info(f"Created collection {name} (vector dimension={dimension})")

    info.update({
        "status": "Created",
        "vector_dimension": dimension,
    })
    return info
   