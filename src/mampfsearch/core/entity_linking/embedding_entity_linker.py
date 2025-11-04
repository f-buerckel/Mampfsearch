import logging
import uuid
from collections import Counter
from qdrant_client.models import PointStruct

from mampfsearch.utils.models import EntityCandidate, Entity, EntityRetrievalItem
from mampfsearch.utils import config
from mampfsearch.retrievers import EntityRetriever
from . import BaseEntityLinker

logger = logging.getLogger(__name__)


class EmbeddingEntityLinker(BaseEntityLinker):
    """Entity linking using embedding similarity"""
    
    def __init__(self):
        self.similarity_threshold = config.ENTITY_EMBED_SIM_THRESHOLD
        self.retriever = EntityRetriever()
    
    def link(self, entity_candidate: EntityCandidate) -> tuple[bool, bool]:
        """Link entity using embedding similarity"""
        results = self.retriever.retrieve(entity_candidate.text, limit=1)
        
        if not results or results[0].score < self.similarity_threshold:
            # Insert as new entity
            logger.info(
                f"Inserting new entity '{entity_candidate.text}' "
                f"with label '{entity_candidate.label}'"
            )
            self._insert_new_entity(entity_candidate)
            return (True, False)
        else:
            # Merge with existing entity
            logger.info(
                f"Entity '{entity_candidate.text}' already in knowledge base "
                f"with name {results[0].entity.name} (score: {results[0].score})"
            )
            self._merge_entities(entity_candidate, results[0])
            return (False, True)
    
    def _insert_new_entity(self, entity_candidate: EntityCandidate):
        """Insert a new entity into both Qdrant and graph storage"""
        entity_id = str(uuid.uuid4())
        
        # Insert into Qdrant
        self._insert_entity_qdrant(entity_id, entity_candidate)
        
        # Insert into graph storage
        graph_storage = config.get_graph_storage()
        graph_storage.insert_entity(
            entity_id=entity_id,
            entity_candidate=entity_candidate
        )
    
    def _insert_entity_qdrant(self, entity_id: str, entity_candidate: EntityCandidate):
        """Insert entity into Qdrant"""
        model = config.get_embedding_model()
        entity_text = entity_candidate.text
        
        embedding = model.encode(entity_text, return_dense=True)
        payload = Entity.from_entity_candidate(entity_candidate).model_dump()
        
        logger.debug(f"Inserting entity '{entity_candidate.text}' into Qdrant")
        
        qdrant_client = config.get_qdrant_client()
        qdrant_client.upsert(
            collection_name=config.ENTITIES_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=entity_id,
                    payload=payload,
                    vector={
                        "dense": embedding["dense_vecs"],
                    }
                )
            ]
        )
    
    def _merge_entities(self, entity_candidate: EntityCandidate, entity_kb: EntityRetrievalItem):
        """Merge entity candidate with existing entity"""
        entity = entity_kb.entity
        entity_instances = entity.entity_instances or []
        
        # Determine most common label and name
        labels = [ent.label for ent in entity_instances] + [entity_candidate.label]
        most_common_label = Counter(labels).most_common(1)[0][0]
        
        aliases = [ent.text for ent in entity_instances] + [entity_candidate.text]
        most_common_name = Counter(aliases).most_common(1)[0][0]
        
        # Update if changed
        if most_common_label != entity.label:
            logger.info(
                f"Updating entity '{entity.name}' label "
                f"from '{entity.label}' to '{most_common_label}'"
            )
            entity.label = most_common_label
        
        if most_common_name != entity.name:
            logger.info(
                f"Updating entity '{entity.name}' name "
                f"from '{entity.name}' to '{most_common_name}'"
            )
            entity.name = most_common_name
        
        entity_instances.append(entity_candidate)
        
        # Update Qdrant
        self._merge_entities_qdrant(entity_kb.id, entity, entity_instances)
    
    def _merge_entities_qdrant(
        self, 
        entity_id: str, 
        entity: Entity, 
        entity_instances: list[EntityCandidate]
    ):
        """Update entity in Qdrant with merged data"""
        client = config.get_qdrant_client()
        client.set_payload(
            collection_name=config.ENTITIES_COLLECTION_NAME,
            payload={
                "name": entity.name,
                "entity_instances": [ei.model_dump() for ei in entity_instances],
                "label": entity.label,
            },
            points=[entity_id],
        )
        
        logger.debug(f"Updated entity in Qdrant: {entity.name}")