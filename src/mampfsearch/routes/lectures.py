from fastapi import APIRouter, HTTPException
from mampfsearch.commands.lectures.search import search_lectures
from mampfsearch.utils import config, models

router = APIRouter(
    prefix="/lectures",
    tags=["Lectures"],
)

@router.post("/search")
async def search_lectures_endpoint(
    request: models.SearchRequest
) -> list[models.RetrievalItem]:

    retrieval_items = search_lectures(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        retriever_type=request.retriever_type,
        reranking=request.reranking,
    )

    return retrieval_items
