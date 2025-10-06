import logging

from enum import Enum
from fastapi import APIRouter, HTTPException
from mampfsearch.core.init import init, create_lectures_collection
from mampfsearch.utils import config

router = APIRouter(
    prefix="/maintenance",
    tags=["Maintenance"],
)

logger = logging.getLogger(__name__)

@router.post("/init", status_code=201)
async def initialize_collections():
    """Initialize the default lectures collection"""

    result = init()
    return {
        "message": "Collections initialization completed",
        "collections": result["collections"],
    }

@router.get("/status")
async def get_collection_info():
    """Return status for both lecture and entity collections."""
    client = config.get_qdrant_client()

    def collect_info(name: str):
        exists = client.collection_exists(name)
        info = {"collection_name": name, "exists": exists}
        if exists:
            col = client.get_collection(name)
            info.update({
                "status": col.status,
                "indexed_vectors_count": col.indexed_vectors_count,
            })
        return info

    return {
        "collections": [
            collect_info(config.LECTURE_COLLECTION_NAME),
            collect_info(config.ENTITIES_COLLECTION_NAME),
        ]
    }

class Collections(str, Enum):
    lectures = "lectures"
    entities = "entities"

@router.delete("/delete/{collection}", status_code=204)
async def delete_collection(collection: Collections):
    """Delete either the lectures or entities collection."""
    client = config.get_qdrant_client()
    collection_name = (
        config.LECTURE_COLLECTION_NAME if collection == Collections.lectures
        else config.ENTITIES_COLLECTION_NAME
    )

    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist"
        )

    client.delete_collection(collection_name)
    logger.info(f"Deleted collection '{collection_name}'")