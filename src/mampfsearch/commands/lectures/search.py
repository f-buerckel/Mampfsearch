import urllib3
from mampfsearch.utils import config, helpers, models
from mampfsearch import retrievers
from rerankers import Reranker
import logging

logger = logging.getLogger(__name__)

urllib3.disable_warnings()

def search_lectures(
        query: str,
        collection_name: str,
        limit: int,
        retriever_type: models.RetrieverTypeEnum,
        reranking: bool =False
        ) -> list[models.RetrievalItem]:

    """Search lectures with keyword or semantic search"""

    retriever = retrievers.HybridRetriever()
    if retriever_type == models.RetrieverTypeEnum.dense:
        retriever = retrievers.DenseRetriever()
    elif retriever_type == models.RetrieverTypeEnum.hybrid:
        retriever = retrievers.HybridRetriever()
    elif retriever_type == models.RetrieverTypeEnum.hybrid_colbert:
        retriever = retrievers.HybridColbertRerankingRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
 
    if reranking:
        reranker = Reranker('BAAI/bge-reranker-v2-m3', verbose=False)
        retriever = retrievers.RerankerRetriever(base_retriever=retriever, reranker=reranker)

    responses = retriever.retrieve(query, collection_name, limit)

    return responses

def search_lectures_command(query, limit, retriever, reranking):

    """Retrieve relevant lecture parts for a given query"""

    responses = search_lectures(query, config.LECTURE_COLLECTION_NAME, limit, retriever, reranking)
    for response in responses:
        logger.info(f"Score: {response.score}")
        logger.info(f"Text: {response.text}")
        logger.info(f"Lecture: {response.lecture} ({response.start_time} - {response.end_time})")
        logger.info(f"Lecture position: {response.lecture_position}")
        logger.info(f"Position: {response.position}")
        logger.info("-" * 50)