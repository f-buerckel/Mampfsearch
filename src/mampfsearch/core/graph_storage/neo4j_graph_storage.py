import logging
import re
import json
from . import BaseGraphStorage
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from mampfsearch.utils.models import EntityCandidate

from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Neo4jGraphStorage(BaseGraphStorage):
    def __init__(self, url: str, user: str, password: str, database_name: str):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.database_name = database_name

    def insert_entity(self, entity_id: str, entity_candidate: EntityCandidate):
        location = entity_candidate.Location

        try:
            location_json = json.dumps(location.model_dump()) if location else None

            self.driver.execute_query(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    label: $label,
                    text: $text,
                    locations: $locations,
                    aliases: $aliases,
                    created_at: datetime()
                })
                """,
                id=entity_id,
                name=entity_candidate.text.lower(),
                label=entity_candidate.label,
                text=entity_candidate.text,
                locations=[location_json],
                aliases=[entity_candidate.text.lower()],
                database_=self.database_name,
            )

            logger.debug(f"Inserted entity into Neo4j: {entity_candidate.text}")

        except Neo4jError as e:
            logger.error(f"Failed to insert entity into Neo4j: {e.message}")

    def merge_entity(
        self, entity_id: str, entity_alias: str, entity_candidate: EntityCandidate
    ):
        location_json = json.dumps(entity_candidate.Location.model_dump())
        try:
            self.driver.execute_query(
                """
                MATCH (e:Entity {id: $id})
                SET e.aliases = e.aliases + $alias,
                    e.locations = e.locations + $location,
                    e.updated_at = datetime()
                """,
                id=entity_id,
                alias=[entity_alias],
                location=[location_json],
                database_=self.database_name,
            )
            logger.debug(f"Merged alias '{entity_alias}' into entity '{entity_id}'")
        except Neo4jError as e:
            logger.error(f"Failed to merge entity alias into Neo4j: {e.message}")

    def insert_relationship(
        self,
        relationship_id: str,
        entity_1_id: str,
        entity_2_id: str,
        relationship: str,
        reasoning: Optional[str] = None,
    ):
        sanitized_relationship = re.sub(r"\W+", "_", relationship or "").strip("_")

        cypher = f"""
        MATCH (e1:Entity {{id: $entity_1_id}})
        MATCH (e2:Entity {{id: $entity_2_id}})
        MERGE (e1)-[r:{sanitized_relationship} {{id: $relationship_id}}]->(e2)
        ON CREATE SET
            r.reasoning = $reasoning,
            r.created_at = datetime()
        RETURN r
        """

        try:
            result, _, _ = self.driver.execute_query(
                cypher,
                relationship_id=relationship_id,
                entity_1_id=entity_1_id,
                entity_2_id=entity_2_id,
                reasoning=reasoning,
                database_=self.database_name,
            )
            if result and len(result) > 0:
                logger.debug(
                    f"Inserted/Merged relationship {sanitized_relationship} "
                    f"({relationship_id}) between {entity_1_id} -> {entity_2_id}"
                )
                return True
            logger.warning(
                f"No relationship created: entity ids may be invalid "
                f"({entity_1_id}, {entity_2_id})"
            )
            return False
        except Neo4jError as e:
            logger.error(f"Failed to insert relationship: {e}")
            return False

    def get_relationship_id(
        self,
        entity_1_id: str,
        relationship: str,
        entity_2_id: str,
    ) -> Optional[str]:
        """Return the relationship id if it exists, else None."""
        sanitized_relationship = re.sub(r"\W+", "_", relationship or "").strip("_")
        cypher = f"""
        MATCH (:Entity {{id: $entity_1_id}})-[r:{sanitized_relationship}]->(:Entity {{id: $entity_2_id}})
        RETURN r.id AS id
        LIMIT 1
        """
        try:
            result, _, _ = self.driver.execute_query(
                cypher,
                entity_1_id=entity_1_id,
                entity_2_id=entity_2_id,
                database_=self.database_name,
            )
            if result and len(result) > 0:
                return result[0].get("id")
            return None
        except Neo4jError as e:
            logger.error(f"Failed to fetch relationship id: {e}")
            return None

    def batch_insert_wikidata_concepts(
        self, concepts: List[Dict[str, Any]], category_label: str
    ):
        """
        Inserts a batch of Wikidata concepts into Neo4j.

        Args:
            concepts: List of dictionaries. Each dict must contain:
                      - 'uri': The Wikidata ID (e.g., http://www.wikidata.org/entity/Q123)
                      - 'name': The English label
                      Optional: 'formula', 'description'
            category_label: The specific type from Wikidata (e.g., "Theorem", "Mathematical Structure")
        """
        if not concepts:
            return

        # Sanitize the label (e.g., "mathematical structure" -> "MathematicalStructure")
        # This prevents Cypher injection.
        safe_label = "".join(x.title() for x in category_label.split())

        # 2. The Query
        # We use MERGE on 'id' to ensure we don't duplicate nodes if we run this twice.
        # We add MULTIPLE labels:
        #   :Entity (for your app compatibility),
        #   :Wikidata (for filtering source),
        #   :<SafeLabel> (specific type)

        cypher = f"""
        UNWIND $batch AS row
        MERGE (e:Entity {{id: row.uri}})
        // Only set created_at if the node is effectively new
        ON CREATE SET e.created_at = datetime()
        SET e:Wikidata,
            e:{safe_label},
            e.name = row.name,
            e.text = row.name,
            e.formula = row.formula,
            e.description = row.description,
            e.source = 'wikidata',
            e.updated_at = datetime()
        """

        try:
            # Execute the query
            self.driver.execute_query(
                cypher, batch=concepts, database_=self.database_name
            )
            logger.info(
                f"Wikidata Import: Inserted/Updated {len(concepts)} nodes of type ':{safe_label}'"
            )
            return True

        except Neo4jError as e:
            logger.error(f"Failed to batch insert Wikidata entities: {e.message}")
            return False

    def insert_grouped_relationships(
        self, rel_type: str, batch: List[Dict[str, str]]
    ) -> bool:
        """
        Inserts a batch of relationships between existing entities.
        If a source or target entity is missing in Neo4j, the relationship is skipped.

        Args:
            rel_type: The relationship type string (e.g., "SUBCLASS_OF").
            batch: List of dicts, where each dict has 'source' and 'target' URIs.
        """
        if not batch:
            return True

        # Sanitize relationship type (allow uppercase, numbers, underscores)
        # e.g. "SUBCLASS_OF" remains "SUBCLASS_OF"
        sanitized_rel = re.sub(r"\W+", "_", rel_type).strip("_").upper()

        cypher = f"""
        UNWIND $batch AS row
        MATCH (source:Entity {{id: row.source}})
        MATCH (target:Entity {{id: row.target}})
        MERGE (source)-[r:{sanitized_rel}]->(target)
        ON CREATE SET 
            r.source = 'wikidata',
            r.created_at = datetime()
        """

        try:
            self.driver.execute_query(cypher, batch=batch, database_=self.database_name)
            logger.info(
                f"Wikidata Import: Processed {len(batch)} relationships of type '{sanitized_rel}'"
            )
            return True
        except Neo4jError as e:
            logger.error(
                f"Failed to batch insert Wikidata relationships ({rel_type}): {e.message}"
            )
            return False
