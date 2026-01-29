import logging
import re
import json

from spacy import Language
from spacy.tokens import Doc
from mampfsearch.utils import config
from mampfsearch.utils.models import Relationship, RelationshipCandidate
from mampfsearch.utils.prompts import (
    RELATIONSHIP_EXTRACTION_PROMPT,
)


logger = logging.getLogger(__name__)

if not Doc.has_extension("rel"):
    Doc.set_extension("rel", default={})


@Language.factory(
    "llm_relationship_extraction",
    requires=["doc.ents", "token.ent_kb_id"],
    assigns=["doc._.rel"],
)
class RelationshipExtractionLLM:
    def __init__(self, nlp, name):
        self.llm_client = config.get_llm_client()
        self.language = nlp.lang
        self.max_words_between_entities = (
            config.MAX_WORDS_BETWEEN_ENTITIES_FOR_RELATIONSHIP
        )

    def __call__(self, doc):
        logger.info(
            f"Starting relationship extraction for doc with text length: {len(doc.text)}"
        )
        candidates = self.extract_relationship_candidates(doc)
        logger.info(
            f"Identified {len(candidates)} candidate pairs for relationship extraction."
        )
        relationships = self.extract_relationships(candidates)
        logger.info(f"Finalized extraction: Found {len(relationships)} relationships.")
        doc = self.set_annotatins(doc, relationships)

        return doc

    def extract_relationships(
        self, candidates: list[RelationshipCandidate]
    ) -> list[Relationship]:
        relationships = []
        for candidate in candidates:
            # Check if token level KB IDs are equal. If so they are the same entity and we probably want to skip this relationship.
            if candidate.entity_1[0].ent_kb_id_ == candidate.entity_2[0].ent_kb_id_:
                logger.debug(
                    f"Skipping candidate {candidate.entity_1.text} <-> {candidate.entity_2.text} as they refer to the same entity."
                )
                continue  # Skip if both entities link to the same KB ID
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                entity1=candidate.entity_1.text,
                entity2=candidate.entity_2.text,
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
                    logger.info(
                        f"Extracted relationship: {relationship.entity_1} {relationship.relationship} {relationship.entity_2}"
                    )
                    relationships.append(relationship)
            except Exception as e:
                logger.error(f"Error during relationship extraction LLM call: {e}")

        return relationships

    @staticmethod
    def parse_llm_response(
        response_content: str, candidate: RelationshipCandidate
    ) -> Relationship | None:
        try:
            # Escape backslashes that aren't part of valid JSON escape sequences
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            response_content = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", response_content)

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
            logger.error(
                f"Failed to parse JSON response: {e}\nContent: {response_content}"
            )
            return None

    def extract_relationship_candidates(self, doc) -> list[RelationshipCandidate]:
        candidates = []
        if len(doc.ents) < 2:
            return candidates
        for i, ent1 in enumerate(doc.ents):
            for ent2 in doc.ents[i + 1 :]:
                words_between_entities = doc[ent1.start : ent2.end].text
                word_distance = len(words_between_entities.split(" "))
                if word_distance <= self.max_words_between_entities:
                    candidate = RelationshipCandidate(
                        entity_1=ent1,
                        entity_2=ent2,
                        context=doc,
                    )
                    candidates.append(candidate)
        return candidates

    @staticmethod
    def set_annotatins(doc, relationships: list[Relationship]):
        for relationship in relationships:
            ent1 = relationship.entity_1
            ent2 = relationship.entity_2
            offset = (ent1.start, ent2.start)
            if offset not in doc._.rel:
                doc._.rel[offset] = relationship
        return doc
