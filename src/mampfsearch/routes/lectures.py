from fastapi import APIRouter, HTTPException
from mampfsearch.commands.lectures.search import search_lectures
from mampfsearch.commands.lectures.ask import ask
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

@router.post("/ask")
async def ask_lectures_endpoint(
    request: models.AskRequest
) -> models.Response:

    response = await ask(
        question=request.question,
        retriever=request.retriever_type,
        limit=request.limit,
        collection_name=request.collection_name,
    )

    return response