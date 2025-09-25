import logging
import json

from openai import AsyncOpenAI

from mampfsearch.core.lectures.search import search_lectures

from mampfsearch.utils.prompts import QA_PROMPT, RAG_PROMPT_JSON
from mampfsearch.utils.models import Response, RetrieverTypeEnum
from mampfsearch.utils import config

logger = logging.getLogger(__name__)

async def ask(question: str,
              retriever: RetrieverTypeEnum = RetrieverTypeEnum.hybrid,
              limit: int = 5,
              ) -> Response:
    """Ask a question and get the answer from the lectures"""

    client = config.get_llm_client()

    response = search_lectures(
        query=question,
        collection_name=config.LECTURE_COLLECTION_NAME,
        limit=limit,
        retriever_type=retriever,
        reranking=False
    )
    if len(response) == 0:
        logger.info("No results found.")
        return '{"answer": "I could not find any relevant information to answer this question.", "confidence_score": 0.0, "source_snippets": {}}'

    contexts = [hit.text for hit in response]
    context_str = "\n\n".join(f"{i+1}: {passage}" for i, passage in enumerate(contexts))

    prompt = RAG_PROMPT_JSON.format(question=question, context=context_str)

    logger.info("Generating answer...")

    answer = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": prompt},
        ],
    )

    response = answer.choices[0].message.content

    try:
        response_dic = json.loads(response)
    except json.JSONDecodeError:
        logger.error("Failed to parse answer as JSON.")
        logger.info(f"Raw answer: {response}")
        response_dic = {
            "answer": "I could not generate a valid answer.",
            "confidence_score": 0.0,
            "source_snippets": {}
        }
    
    logger.info(f"Answer: {response_dic['answer']}")

    response = Response(**response_dic)
    return response