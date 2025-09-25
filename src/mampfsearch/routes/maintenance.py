import logging
from fastapi import APIRouter, HTTPException
from mampfsearch.core.lectures.init import init, create_lectures_collection
from mampfsearch.utils import config

router = APIRouter(
    prefix="/maintenance",
    tags=["Maintenance"],
)

logger = logging.getLogger(__name__)

@router.post("/init", status_code=201)
async def initialize_collections():
    """Initialize the default lectures collection"""
    init()
    return {
        "message": "Collection initialization completed successfully",
        "collection_name": config.LECTURE_COLLECTION_NAME
    }


@router.get("/status")
async def get_collection_info():
    """Get detailed information about a specific collection"""
    client = config.get_qdrant_client()
    collection_name = config.LECTURE_COLLECTION_NAME
    
    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist"
        )
    
    collection_info = client.get_collection(collection_name)
    
    return {
        "collection_name": collection_name,
        "status": collection_info.status,
        "indexed_vectors_count": collection_info.indexed_vectors_count,
    }

@router.delete("/delete", status_code=204)
async def delete_collection():
    """Delete a collection"""
    client = config.get_qdrant_client()
    collection_name = config.LECTURE_COLLECTION_NAME
    
    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist"
        )
    
    client.delete_collection(collection_name)
    logger.info(f"Deleted collection '{collection_name}'")