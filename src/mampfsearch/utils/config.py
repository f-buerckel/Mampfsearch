from pathlib import Path 
import logging
import atexit

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_HOST = "http://localhost:11434"



EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

LECTURE_COLLECTION_NAME = "Lectures"

PREFETCH_LIMIT = 50


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


_ollama_client = None
def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        from ollama import Client
        _ollama_client = Client(host = OLLAMA_HOST)

    return _ollama_client

#Paths
def get_root_path():
    return Path(__file__).parent.parent.parent.parent

def get_lectures_path():
    return get_root_path() / "docker" / "lectures"

def get_benchmark_path():
    return get_root_path() / "benchmarks"

logging.basicConfig(
  logging  level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../mampfsearch.log'),
        logging.StreamHandler() 
    ]
)

logger = .getLogger(__name__)