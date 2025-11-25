from abc import ABC, abstractmethod
from mampfsearch.utils.models import MathEntityCandidate
from typing import Optional


class BaseGraphStorage(ABC):
    """
    Abstract base class for graph storage implementations.
    """

    @abstractmethod
    def add_entity(self, entity_id: str, entity_candidate: MathEntityCandidate):
        pass

    @abstractmethod
    def merge_entity(
        self, entity_id: str, entity_alias: str, entity_candidate: MathEntityCandidate
    ):
        pass

    @abstractmethod
    def insert_relationship(
        self,
        relationship_id: str,
        entity_1_id: str,
        entity_2_id: str,
        relationship: str,
        reasoning: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def get_relationship_id(
        self,
        entity_1_id: str,
        relationship: str,
        entity_2_id: str,
    ) -> Optional[str]:
        pass
