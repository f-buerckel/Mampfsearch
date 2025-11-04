from abc import ABC, abstractmethod
from mampfsearch.utils.models import EntityCandidate

class BaseEntityLinker(ABC):
    
    @abstractmethod
    def link(self, entity_candidate: EntityCandidate) -> tuple[bool, bool]:
        """
        Link an entity candidate to the knowledge base.
        
        Args:
            entity_candidate: The entity to link
            
        Returns:
            Tuple of (is_new, is_merged) indicating if entity was newly inserted or merged
        """
        pass