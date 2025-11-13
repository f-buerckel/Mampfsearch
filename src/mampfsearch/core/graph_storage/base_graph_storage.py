from abc import ABC, abstractmethod
from mampfsearch.utils.models import EntityCandidate
from typing import Optional

class BaseGraphStorage(ABC):
    """
    Abstract base class for graph storage implementations.
    """

    @abstractmethod
    def insert_entity(entity_id: str, entity_candidate: EntityCandidate):
        pass

    def get_relationship_id(
        self,
        entity_1_id: str,
        relationship: str,
        entity_2_id: str,
    ) -> Optional[str]:
        pass