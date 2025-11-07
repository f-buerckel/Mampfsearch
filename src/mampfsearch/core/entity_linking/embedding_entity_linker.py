import logging
import uuid

from spacy import Language
from spacy.tokens import Span, Doc

from mampfsearch.utils import config
from mampfsearch.retrievers import EntityRetriever
from mampfsearch.utils.models import EntityCandidate, Entity
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

if not Span.has_extension("is_new_entity"):
    Span.set_extension("is_new_entity", default=False)

@Language.factory("embedding_entity_linker")
def create_embedding_entity_linker(nlp: Language, name: str):
    return EmbeddingEntityLinker()

# TODO: Determine if this should really be a spaCy component?
# I dont know what the spaCy philosophy is regarding component that take a document and interact with an external database.
# Because now this component step is really state dependent does not contribute to other steps.
# On the other hand in theory it takes a document, processes it and returns a document with optionally some enriched annotations.
# Practical Concern: As a spaCy component I cant make it async which would be nice for the calls to the graph storage.
class EmbeddingEntityLinker():

    def __init__(self):
        self.similarity_threshold = config.ENTITY_EMBED_SIM_THRESHOLD
        self.retriever = EntityRetriever()

    def __call__(self, doc):
        for ent in doc.ents:
            results = self.retriever.retrieve(ent.text, limit=1)

            if results and results[0].score >= self.similarity_threshold:
                # match found
                logger.debug(f"Entity '{ent.text}' matched with {results[0].id}")
                entity_id = results[0].id
                ent._.is_new_entity = False
            else:
                # New entity - insert immediately
                logger.info(f"No match found for entity '{ent.text}', inserting now")
                entity_id = str(uuid.uuid4())
                ent._.is_new_entity = True
                
                # Create entity candidate and insert
                entity_candidate = EntityCandidate(
                    text=ent.text,
                    label=ent.label_,
                    Location=doc._.location
                )
                
                self._insert_entity(entity_id, entity_candidate)
            
            for token in ent:
                token.ent_kb_id_ = entity_id

        return doc
    
    # Sadly the linker also has to do the insertion. Otherwise it will only link to entities
    # that were already present before the extraction run. This fails if the same entities in the same document 
    # which is quite common.
    def _insert_entity(self, entity_id: str, entity_candidate: EntityCandidate):
        """Insert entity into both Qdrant and graph storage immediately"""
        # Insert into Qdrant
        model = config.get_embedding_model()
        embedding = model.encode(entity_candidate.text, return_dense=True)
        payload = Entity.from_entity_candidate(entity_candidate).model_dump()
        
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
        
        # Insert into graph storage
        graph_storage = config.get_graph_storage()
        graph_storage.insert_entity(
            entity_id=entity_id,
            entity_candidate=entity_candidate
        )
        
        logger.debug(f"Inserted entity '{entity_candidate.text}' with id '{entity_id}'")