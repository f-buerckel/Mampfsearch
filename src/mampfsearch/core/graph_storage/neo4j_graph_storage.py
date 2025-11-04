import logging 
from . import BaseGraphStorage
from neo4j import GraphDatabase

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


    def insert_entity(self, entity_id: str, entity_candidate):
        location = entity_candidate.Location
        
        try:
            location_data = location.model_dump()
            
            self.driver.execute_query(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    label: $label,
                    text: $text,
                    created_at: datetime()
                })
                """,
                id=entity_id,
                name=entity_candidate.text.lower(),
                label=entity_candidate.label,
                text=entity_candidate.text,
                database_=self.database_name,
            )
            
            logger.debug(f"Inserted entity into Neo4j: {entity_candidate.text}")
            
        except Neo4jError as e:
            logger.error(f"Failed to insert entity into Neo4j: {e.message}")