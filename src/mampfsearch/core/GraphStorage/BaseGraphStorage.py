from abc import ABC, abstractmethod
from mampfsearch.utils.models import EntityCandidate

class BaseGraphStorage(ABC):
    """
    Abstract base class for graph storage implementations.
    """

    @abstractmethod
    def insert_entity(entity_id: str, entity_candidate: EntityCandidate):
        pass