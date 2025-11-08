import uuid
import logging
from spacy import Language
from spacy.tokens import Doc
from mampfsearch.utils import config
from mampfsearch.utils.models import Relationship

logger = logging.getLogger(__name__)


if not Doc.has_extension("rel_kb_id"):
    Doc.set_extension("rel_kb_id", default={})


@Language.factory(
    "simple_relationship_linker",
    requires=["doc.ents", "doc._.rel", "token.ent_kb_id"],
    assigns=["doc._.rel_kb_id"],
)
class SimpleRelationshipLinker:
    def __init__(self, nlp, name):
        pass

    def __call__(self, doc):
        for offset, rel in doc._.rel.items():
            ent1_id = rel.entity_1[0].ent_kb_id_
            ent2_id = rel.entity_2[0].ent_kb_id_

            if ent1_id and ent2_id:
                relationship_id = str(uuid.uuid4())
                logger.info(f"Linking relationship '{rel.relationship}' between entities {ent1_id} and {ent2_id} with id {relationship_id}")
                doc._.rel_kb_id[offset] = relationship_id
                self.insert_relationship_into_storage(rel, relationship_id)
            else:
                logger.warning(f"Could not link relationship '{rel.relationship}' due to missing entity IDs: ent1_id={ent1_id}, ent2_id={ent2_id}")

        return doc
    
    @staticmethod
    def insert_relationship_into_storage(relationship: Relationship, relationship_id: str):
        """Insert relationship into graph storage immediately"""
        graph_db = config.get_graph_storage()
        graph_db.insert_relationship(
            relationship_id=relationship_id,
            entity_1_id=relationship.entity_1[0].ent_kb_id_,
            entity_2_id=relationship.entity_2[0].ent_kb_id_,
            relationship=relationship.relationship,
            reasoning=relationship.reasoning
        )