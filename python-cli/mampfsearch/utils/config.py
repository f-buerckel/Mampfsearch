from pathlib import Path 
import atexit

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_HOST = "http://localhost:11434"



EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


_bge_embedding_model = None
def get_bge_embedding_model():
    global _bge_embedding_model
    if _bge_embedding_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _bge_embedding_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    return _bge_embedding_model

LECTURE_COLLECTION_NAME = "Lectures"

PREFETCH_LIMIT = 50

def get_qdrant_client():
    from qdrant_client import QdrantClient
    """Return configured qdrant client"""
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )

def get_ollama_client():
    from ollama import Client
    """Return configured ollama client"""
    ollama_client = Client(host = OLLAMA_HOST)
    return ollama_client

#Paths
def get_root_path():
    return Path(__file__).parent.parent.parent.parent

def get_lectures_path():
    return get_root_path() / "docker" / "lectures"

def get_benchmark_path():
    return get_root_path() / "benchmarks"