from pathlib import Path 
import logging

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

VLLM_HOST = "localhost"
VLLM_PORT = 8001

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

LECTURE_COLLECTION_NAME = "Lectures"
ENTITIES_COLLECTION_NAME = "Entities"

PREFETCH_LIMIT = 50

# If there is an entity embedding with cosine similarity above this threshold, we consider it the same entity.
ENTITY_EMBED_SIM_THRESHOLD = 0.83


_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _embedding_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    return _embedding_model


_qdrant_client = None
def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )

    return _qdrant_client


_llm_client = None
def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI
        _llm_client = AsyncOpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="dummy")

    return _llm_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../mampfsearch.log'),
        logging.StreamHandler() 
    ]
)

logging.getLogger(__name__).setLevel(logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.INFO)
logging.getLogger("spacy_llm").setLevel(logging.INFO)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.INFO)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)
logging.getLogger("runner").setLevel(logging.WARNING)
