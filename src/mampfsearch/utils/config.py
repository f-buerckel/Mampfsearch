from mampfsearch.core.graph_storage import Neo4jGraphStorage
from dotenv import load_dotenv
import logging
import os

load_dotenv()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

VLLM_HOST = "localhost"
VLLM_PORT = 8001
# LLM_MODEL_NAME = "leon-se/gemma-3-27b-it-qat-W4A16-G128"
# LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_MODEL_NAME = "gpt-5.2"
LOCAL_LLM = False
API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

GRAPH_STORAGE = "neo4j"

LECTURE_COLLECTION_NAME = "Lectures"
ENTITIES_COLLECTION_NAME = "Entities"
NEO4J_DATABASE_NAME = "neo4j"
NEO4J_IN_MEMORY_GRAPH_NAME = "kg"

PREFETCH_LIMIT = 50

# If there is an entity embedding with cosine similarity above this threshold, we consider it the same entity.
ENTITY_EMBED_SIM_THRESHOLD = 0.83


# Maximum words allowed between entities to consider a possible relationship between them
MAX_WORDS_BETWEEN_ENTITIES_FOR_RELATIONSHIP = 15


# Weights for reranking entities for question generation
SELECTION_WEIGHTS = {
    "density_ratio": 0.7,
    "term_frequency": 0.1,
    "pagerank_score": 0.2,
}


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from FlagEmbedding import BGEM3FlagModel

        _embedding_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    return _embedding_model


_graph_storage = None


def get_graph_storage():
    global _graph_storage
    if _graph_storage is None:
        if GRAPH_STORAGE == "neo4j":
            _graph_storage = _get_neo4j_graph_storage()
        else:
            raise ValueError(f"Unknown GRAPH_STORAGE: {GRAPH_STORAGE}")

    return _graph_storage


def _get_neo4j_graph_storage():
    storage = Neo4jGraphStorage(
        url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        database_name=NEO4J_DATABASE_NAME,
    )
    return storage


from mampfsearch.core.vector_storage import QdrantVectorStorage

_vector_storage = None


def get_vector_storage():
    global _vector_storage
    if _vector_storage is None:
        _vector_storage = QdrantVectorStorage(host=QDRANT_HOST, port=QDRANT_PORT)

    return _vector_storage


_async_llm_client = None


def get_async_llm_client():
    global _async_llm_client
    if _async_llm_client is None:
        from openai import AsyncOpenAI

        if LOCAL_LLM:
            _async_llm_client = AsyncOpenAI(
                base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="dummy"
            )
        else:
            _async_llm_client = AsyncOpenAI(api_key=API_KEY)

    return _async_llm_client


_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        if LOCAL_LLM:
            _llm_client = OpenAI(
                base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="dummy"
            )
        else:
            _llm_client = OpenAI(api_key=API_KEY)

    return _llm_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("../mampfsearch.log"), logging.StreamHandler()],
)

logging.getLogger("mampfsearch").setLevel(logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.INFO)
logging.getLogger("spacy_llm").setLevel(logging.INFO)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.INFO)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)
logging.getLogger("runner").setLevel(logging.WARNING)
