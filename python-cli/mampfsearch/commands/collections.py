from mampfsearch.utils.config import get_qdrant_client
import logging

logger = logging.getLogger(__name__)

def delete(name):
    client = get_qdrant_client()
    if not client.collection_exists(name):
        logger.warning(f"Collection {name} does not exist")
        return
    
    client.delete_collection(name)
    logger.info(f"Deleted collection {name}")

def list():
    client = get_qdrant_client()
    collections = client.get_collections().collections

    logger.info(f"Found {len(collections)} collections:")

    for collection in collections:
        logger.info(collection.name)
    
    return

def get(name):
    client = get_qdrant_client()
    if not client.collection_exists(name):
        logger.warning(f"Collection {name} does not exist")
        return
    
    collection_info = client.get_collection(name)
    model_info = collection_info.config.params.vectors

    logger.info(f"Status: {collection_info.status}")
    logger.info(f"Indexed vector count: {collection_info.indexed_vectors_count}")
    logger.info(f"Embedding dimension: {model_info.size}")
    logger.info(f"Distance metric: {model_info.distance}")

    return
