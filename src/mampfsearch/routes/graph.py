from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from collections import Counter
from pathlib import Path
from typing import Optional

from mampfsearch.utils import config, models
from mampfsearch.core.extraction_pipeline import extract
from mampfsearch.retrievers import EntityRetriever


router = APIRouter(
    prefix="/graph",
    tags=["Graph"],
)


@router.post("/extract")
async def extract_entities_endpoint(
    file: Path,
    course_name: str,
    lecture_name: str,
    lecture_position: int,
    lecture_description: Optional[str] = None,
    background_task: BackgroundTasks = None,
):
    if not file.exists() or not file.is_file():
        raise HTTPException(
            status_code=400, detail="File does not exist or is not a file."
        )

    client = config.get_vector_storage()
    if not client.collection_exists(config.ENTITIES_COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail=f"Entity collection '{config.ENTITIES_COLLECTION_NAME}' does not exist. Initialize it via POST /maintenance/init",
        )

    background_task.add_task(
        extract,
        file_path=Path(file),
        course_name=course_name,
        lecture_name=lecture_name,
        lecture_position=lecture_position,
        lecture_description=lecture_description,
    )

    return {"message": "Entity extraction started in background"}


@router.get("/search")
async def search_entities(
    query: str,
    limit: int,
) -> list[models.EntityRetrievalItem]:
    """Search entities with semantic search"""

    retriever = EntityRetriever()
    responses = retriever.retrieve(query, limit)

    return responses


@router.get("/entities")
async def get_all_entities(
    label: Optional[str] = Query(
        None, description="Filter by entity label (e.g., ALGORITHM, THEOREM_RULE)"
    ),
    limit: int = Query(100, ge=1, le=100000),
    include_aliases: bool = Query(
        False,
        description="Include list of all text variations (aliases) for each entity",
    ),
) -> dict:
    client = config.get_vector_storage()

    if not client.collection_exists(config.ENTITIES_COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail=f"Entity collection '{config.ENTITIES_COLLECTION_NAME}' does not exist.",
        )

    points = client.list_entities(
        collection_name=config.ENTITIES_COLLECTION_NAME,
        label=label,
        limit=limit,
    )

    entities_summary = []
    for point in points:
        entity = models.Entity(**point.payload)

        entity_data = {
            "name": entity.name,
            "label": entity.label,
            "num_instances": len(entity.entity_instances)
            if entity.entity_instances
            else 0,
        }

        if include_aliases and entity.entity_instances:
            aliases = list(set(instance.text for instance in entity.entity_instances))
            aliases.sort()
            entity_data["aliases"] = aliases

        entities_summary.append(entity_data)

    entities_summary.sort(key=lambda e: (e["label"], -e["num_instances"], e["name"]))
    label_counts = Counter(e["label"] for e in entities_summary)

    return {
        "total": len(entities_summary),
        "limit": limit,
        "include_aliases": include_aliases,
        "label_distribution": dict(label_counts),
        "entities": entities_summary,
    }
