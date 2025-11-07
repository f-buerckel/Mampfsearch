import logging
import re
import json

from spacy.lang.en import English
from spacy.lang.de import German
from mampfsearch.utils import config
from mampfsearch.utils.models import Relationship, RelationshipCandidate
from mampfsearch.utils.prompts import RELATIONSHIP_EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


class RelationshipExtractionLLM:
    def __init__(self, language: str = "en"):

        self.llm_client = config.get_llm_client()
        self.language = language
        self.max_words_between_entities = config.MAX_WORDS_BETWEEN_ENTITIES_FOR_RELATIONSHIP
    
    def __call__(self, doc):

        candidates = self.extract_relationship_candidates(doc)
        relationships = self.extract_relationships(candidates)
        doc = set_annotatins(doc, relationships)
        
        return doc
    
    def extract_relationships(self, candidates: list[RelationshipCandidate]) -> list[Relationship]:
        relationships = []
        for candidate in candidates:
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                entity1=candidate.entity_1.text,
                entity2=candidate.entity_2.text,
                sentence=candidate.sentence.text,
                context=candidate.context.text,
            )
            try:
                response = self.llm_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                relationship = self.parse_llm_response(content, candidate)
                if relationship:
                    logger.info(f"Extracted relationship: {relationship.entity_1} {relationship.relationship} {relationship.entity_2}")
                    relationships.append(relationship)
            except Exception as e:
                logger.error(f"Error during relationship extraction LLM call: {e}")
        
        return relationships

    
    @staticmethod
    def parse_llm_response(response_content: str, candidate: RelationshipCandidate) -> Relationship | None:
        try:

            # Escape backslashes that aren't part of valid JSON escape sequences
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            response_content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', response_content)

            data = json.loads(response_content)
            reasoning = data.get("reasoning", "")
            relationship = data.get("relationship", "")
            
            if relationship and relationship != "NO_RELATIONSHIP":
                return Relationship(
                    entity_1=candidate.entity_1,
                    entity_2=candidate.entity_2,
                    relationship=relationship,
                    reasoning=reasoning,
                    context=candidate.context,
                )
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nContent: {response_content}")
            return None

    def extract_relationship_candidates(self, doc) -> list[RelationshipCandidate]:
        candidates = []
        for sent in doc.sents:
            if len(sent.ents) < 2:
                continue
            for i, ent1 in enumerate(sent.ents):
                for ent2 in sent.ents[i+1:]:
                    words_between_entities = doc[ent1.start:ent2.end].text
                    word_distance = len(words_between_entities.split(" "))
                    if word_distance <= self.max_words_between_entities:
                        candidate = RelationshipCandidate(
                            entity_1=ent1,
                            entity_2=ent2,
                            sentence=sent,
                            context=doc,
                        )
                        candidates.append(candidate)
        return candidates
    
    @staticmethod
    def set_annotatins(doc, relationships: list[Relationship]):
        for relationship in relationship:
            ent1 = relationship.entity_1
            ent2 = relationship.entity_2
            offset = (ent1.start, ent2.start)
            if offset not in doc._.rel:
                doc._.rel[offset] = relationship.relationship
        return doc

    
@English.factory("llm_relationship_extraction")
def create_english_relationship_extraction_llm(nlp, name):
    return RelationshipExtractionLLM(language="en")

@German.factory("llm_relationship_extraction")
def create_german_relationship_extraction_llm(nlp, name):
    return RelationshipExtractionLLM(language="de")
