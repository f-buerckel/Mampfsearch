import click
import urllib3
from mampfsearch.utils import config
from mampfsearch.utils import helpers
from mampfsearch import retrievers
import logging

logger = logging.getLogger(__name__)

urllib3.disable_warnings()

def search_lectures(term, collection, limit, retriever_type, reranking=False, expand_first_answer=False):
    from rerankers import Reranker

    """Search lectures with keyword or semantic search"""

    retriever = retrievers.HybridRetriever()
    if retriever_type == "dense":
        retriever = retrievers.DenseRetriever()
    elif retriever_type == "hybrid":
        retriever = retrievers.HybridRetriever()
    elif retriever_type == "hybrid+colbert":
        retriever = retrievers.HybridColbertRerankingRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
 
    if reranking:
        reranker = Reranker('BAAI/bge-reranker-v2-m3', verbose=False)
        retriever = retrievers.RerankerRetriever(base_retriever=retriever, reranker=reranker)

    responses = retriever.retrieve(term, collection, limit)

    return responses

@click.command(name="search")
@click.argument("query")
@click.option("--limit", "-l", default=3, help="Number of results to return")
@click.option(
    "--retriever", "-r",
    type=click.Choice(["dense", "hybrid", "hybrid+colbert"], case_sensitive=False),
    default="hybrid+colbert", 
    help="Type of retriever to use for search"
)
@click.option(
    "--reranking/--no-reranking",
    default=False,
    help="Enable or disable reranking"
)
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