from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStorage(ABC):
    """
    Abstract base class for vector storage implementations.
    """

    @abstractmethod
    def upsert(self, collection_name: str, points: List[Any]):
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5, with_payload: bool = True) -> List[Any]:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_lecture_collection(self, collection_name: str, dimension: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_entity_collection(self, collection_name: str, dimension: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def upload_chunks(self, collection_name: str, vectors: List[Dict[str, Any]], payloads: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def list_entities(self, collection_name: str, label: Optional[str] = None, limit: int = 100) -> List[Any]:
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        pass
