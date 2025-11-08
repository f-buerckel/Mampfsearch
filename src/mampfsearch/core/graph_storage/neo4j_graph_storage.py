import logging 
import re
import json
from . import BaseGraphStorage
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from mampfsearch.utils.models import EntityCandidate

from typing import Optional

logger = logging.getLogger(__name__)

class Neo4jGraphStorage(BaseGraphStorage):

    def __init__(self,
     url: str,
     user: str,
     password: str,
     database_name: str
    ):
        
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
    
    def merge_entity(self, entity_id: str, entity_alias: str, entity_candidate: EntityCandidate):
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