import click
from .search import search_lectures
from mampfsearch.utils.prompts import QA_PROMPT, RAG_PROMPT_JSON
from mampfsearch.utils import config
import logging

logger = logging.getLogger(__name__)

@click.command(name="ask")
@click.argument("question", required=True)
@click.option(
    "--retriever", "-r",
    type=click.Choice(["dense", "hybrid", "hybrid+colbert"], case_sensitive=False),
    default="hybrid+colbert", 
    help="Type of retriever to use for search"
)
def ask(question, retriever):
    """Ask a question and get the answer from the lectures"""

    ollama_client = config.get_ollama_client()

    response = search_lectures(term=question, collection=config.LECTURE_COLLECTION_NAME, limit=5, retriever_type=retriever, expand_first_answer=True)
    if len(response) == 0:
        logger.info("No results found.")
        return

    contexts = [hit.text for hit in response]
    context_str = "\n\n".join(f"{i+1}: {passage}" for i, passage in enumerate(contexts))

    prompt = RAG_PROMPT_JSON.format(question=question, context=context_str)

    logger.info(prompt)
    ollama_client.pull("gemma3n:e4b")
    answer = ollama_client.generate(model="gemma3n:e4b", prompt=prompt, options={"temperature": 0.5}).response
    logger.info("Answer:")
    logger.info(answer)

