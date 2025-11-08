import logging

from spacy import Language
from spacy.tokens import Doc
from mampfsearch.utils import config
from mampfsearch.utils.models import Relationship
from mampfsearch.utils.prompts import RELATIONSHIP_VALIDATION_PROMPT


logger = logging.getLogger(__name__)


@Language.factory(
    "llm_relationship_validation",
    requires=["doc._.rel"],
    assigns=["doc._.rel"],
)
class RelationshipValidationLLM:
    def __init__(self, nlp, name):
        self.llm_client = config.get_llm_client()
    
    def __call__(self, doc):
        if not doc._.rel:
            return doc
        
        validated_relationships = []
        
        for offset, rel in doc._.rel.items():
            if self.validate_relationship(rel):
                validated_relationships.append((offset, rel))
        
        doc._.rel.clear()
        for offset, rel in validated_relationships:
            doc._.rel[offset] = rel
        
        return doc
    
    def validate_relationship(self, rel: Relationship) -> bool:
        prompt = RELATIONSHIP_VALIDATION_PROMPT.format(
            context=rel.context.text,
            sentence=rel.context.text, 
            entity1=rel.entity_1.text,
            entity2=rel.entity_2.text,
            relationship=rel.relationship,
        )
        try:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": prompt},
                ],
            )
            content = response.choices[0].message.content.lower()
            if "yes" in content:
                logger.debug(f"Relationship '{rel.entity_1.text}' --[{rel.relationship}]--> '{rel.entity_2.text}' validated by LLM.")
                return True
            else:
                logger.info(f"Relationship '{rel.entity_1.text}' --[{rel.relationship}]--> '{rel.entity_2.text}' rejected by LLM as context-specific.")
                return False
        except Exception as e:
            logger.error(f"Error during relationship validation LLM call: {e}")
            return True