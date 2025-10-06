from fastapi import APIRouter, HTTPException
from pathlib import Path

from mampfsearch.utils import config, models
from mampfsearch.core.entity_extraction import extract_entities

router = APIRouter(
    prefix="/graph",
    tags=["Graph"],
)

@router.post("/extract")
async def extract_entities_endpoint(
    file: Path
) -> models.ExtractionInfo:

    if not file.exists() or not file.is_file():
        raise HTTPException(status_code=400, detail="File does not exist or is not a file.")

    client = config.get_qdrant_client()
    if not client.collection_exists(config.ENTITIES_COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail=f"Entity collection '{config.ENTITIES_COLLECTION_NAME}' does not exist. Initialize it via POST /maintenance/init"
        )

    info = extract_entities(
        file_path=Path(file),
        max_sentences_per_chunk=3,
        print_chunks=True
    )

    return info
