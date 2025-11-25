import logging
import re
import json
import uuid

from . import BaseGraphStorage
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from mampfsearch.utils.models import (
    MathEntityCandidate,
    CourseNode,
    Course,
    LectureNode,
    Lecture,
    SegmentNode,
    Segment,
    PdfFileNode,
    PdfFile,
    PassageNode,
    Passage,
    BaseNode,
)

from typing import Optional, List, Dict, Any, Type

logger = logging.getLogger(__name__)


class Neo4jGraphStorage(BaseGraphStorage):
    def __init__(self, url: str, user: str, password: str, database_name: str):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.database_name = database_name

    def _get_node_properties(
        self,
        node_cls: Optional[Type[BaseNode]],
        id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Generic helper: fetch a single node (Course, Lecture, ...) by id or name.
        Uses node_cls.get_identifying_label() as the Neo4j label if provided.
        Returns raw properties + labels dict, or None.
        """
        if not id and not name:
            raise ValueError("Either 'id' or 'name' must be provided")

        params: Dict[str, Any] = {}
        if id:
            where = "n.id = $id"
            params["id"] = id
        else:
            where = "n.name = $name"
            params["name"] = name

        if node_cls:
            label = node_cls.get_identifying_label()
            cypher = f"""
            MATCH (n:{label})
            WHERE {where}
            RETURN n, labels(n) AS labels
            LIMIT 1
            """
        else:
            cypher = f"""
            MATCH (n)
            WHERE {where}
            RETURN n, labels(n) AS labels
            LIMIT 1
            """

        try:
            result, _, _ = self.driver.execute_query(
                cypher,
                **params,
                database_=self.database_name,
            )
            if not result:
                return None

            record = result[0]
            props = dict(record["n"])
            props["labels"] = record["labels"]
            return props

        except Neo4jError as e:
            logger.error(f"Failed to fetch node: {e}")
            return None

    def get_course_node(
        self, id: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[CourseNode]:
        if not id and not name:
            raise ValueError("Either 'id' or 'name' must be provided")

        result = self._get_node_properties(node_cls=CourseNode, id=id, name=name)
        if not result:
            return None

        return CourseNode(
            graph_id=result["id"],
            name=result["name"],
            labels=set(result.get("labels", [])),
            course=Course(
                name=result.get("name"),
                description=result.get("description"),
                instructor=result.get("instructor"),
            ),
        )

    def get_lecture_node(
        self, id: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[LectureNode]:
        if not id and not name:
            raise ValueError("Either 'id' or 'name' must be provided")

        result = self._get_node_properties(node_cls=LectureNode, id=id, name=name)
        if not result:
            return None

        return LectureNode(
            graph_id=result["id"],
            name=result["name"],
            labels=set(result.get("labels", [])),
            lecture=Lecture(
                name=result.get("name"),
                position=result.get("position"),
                description=result.get("description"),
                upload_date=result.get("upload_date"),
            ),
        )

    def add_course_node(self, course: Course) -> CourseNode:
        try:
            course_values = course.model_dump()
            course_id = str(uuid.uuid4())

            self.driver.execute_query(
                """
                MERGE (c:Course {id: $id})
                SET c.name = $params.name,
                    c.description = $params.description,
                    c.instructor = $params.instructor
                """,
                params=course_values,
                id=course_id,
                database_=self.database_name,
            )
            logger.debug(f"Inserted course node into Neo4j: {course_values['name']}")

            courseNode = CourseNode(
                graph_id=course_id,
                name=course.name,
                labels={CourseNode.get_identifying_label()},
                course=course,
            )
            return courseNode

        except Neo4jError as e:
            logger.error(f"Failed to insert course node into Neo4j: {e.message}")
            return None

    def add_lecture_node(self, lecture: Lecture, courseNode: CourseNode) -> LectureNode:
        try:
            lecture_values = lecture.model_dump()
            lecture_id = str(uuid.uuid4())
            self.driver.execute_query(
                """
                MATCH (c:Course {id: $course_id})
                MERGE (l:Lecture {id: $id})
                SET l.name = $params.name,
                    l.position = $params.position,
                    l.description = $params.description,
                    l.upload_date = $params.upload_date
                MERGE (c)-[:HAS_LECTURE]->(l)
                
                // create IS_SUCCESSOR relationship
                WITH c, l
                MATCH (prev:Lecture)<-[:HAS_LECTURE]-(c)
                WHERE prev.position = l.position - 1
                MERGE (prev)-[:IS_SUCCESSOR]->(l)
                """,
                course_id=courseNode.graph_id,
                id=lecture_id,
                params=lecture_values,
                database_=self.database_name,
            )
            logger.debug(f"Inserted lecture node into Neo4j: {lecture_values['name']}")
            lectureNode = LectureNode(
                graph_id=lecture_id,
                name=lecture.name,
                labels={LectureNode.get_identifying_label()},
                lecture=lecture,
            )
            return lectureNode

        except Neo4jError as e:
            logger.error(f"Failed to insert lecture node into Neo4j: {e.message}")

    def add_pdf_file_node(self, pdf_file: PdfFile, course: CourseNode) -> PdfFileNode:
        try:
            pdf_values = pdf_file.model_dump()
            pdf_id = str(uuid.uuid4())

            self.driver.execute_query(
                """
                MATCH (c:Course {id: $course_id})
                MERGE (p:PdfFile {id: $id})
                SET p.name = $params.name,
                    p.upload_date = $params.upload_date,
                    p.description = $params.description
                MERGE (c)-[:HAS_PDF_FILE]->(p)
                """,
                course_id=course.graph_id,
                id=pdf_id,
                params=pdf_values,
                database_=self.database_name,
            )
            logger.debug(
                f"Inserted/merged PDF file node into Neo4j: {pdf_values['name']}"
            )

            pdfFileNode = PdfFileNode(
                graph_id=pdf_id,
                name=pdf_file.name,
                labels={PdfFileNode.get_identifying_label()},
                pdf_file=pdf_file,
            )
            return pdfFileNode

        except Neo4jError as e:
            logger.error(f"Failed to insert PDF file node into Neo4j: {e.message}")
            return None

    def add_passage_node(self, passage: Passage, pdf_file: PdfFileNode) -> PassageNode:
        try:
            passage_values = passage.model_dump()
            passage_id = str(uuid.uuid4())

            self.driver.execute_query(
                """
                MATCH (p:PdfFile {id: $pdf_file_id})
                MERGE (pa:Passage {id: $id})
                SET pa.name = $params.text,
                    pa.text = $params.text,
                    pa.location = $location
                MERGE (p)-[:HAS_PASSAGE]->(pa)
                """,
                pdf_file_id=pdf_file.graph_id,
                id=passage_id,
                params=passage_values,
                location=json.dumps(passage.location.model_dump())
                if passage.location
                else None,
                database_=self.database_name,
            )
            logger.debug(
                f"Inserted/merged passage node into Neo4j: {passage_values['text'][0:10]}"
            )

            passageNode = PassageNode(
                graph_id=passage_id,
                name=passage.text,
                labels={PassageNode.get_identifying_label()},
                passage=passage,
            )
            return passageNode
        except Neo4jError as e:
            logger.error(f"Failed to insert passage node into Neo4j: {e.message}")
            return None

    def add_segment_node(self, segment: Segment, lecture: LectureNode) -> SegmentNode:
        try:
            segment_values = segment.model_dump()
            segment_id = str(uuid.uuid4())
            self.driver.execute_query(
                """
                MATCH (l:Lecture {id: $lecture_id})
                MERGE (s:Segment {id: $id})
                SET s.name = $params.text,
                    s.text = $params.text,
                    s.location = $location
                MERGE (l)-[:HAS_SEGMENT]->(s)
                """,
                lecture_id=lecture.graph_id,
                id=segment_id,
                params=segment_values,
                location=json.dumps(segment.location.model_dump())
                if segment.location
                else None,
                database_=self.database_name,
            )
            logger.debug(
                f"Inserted segment node into Neo4j: {segment_values['text'][0:10]}"
            )
            segmentNode = SegmentNode(
                graph_id=segment_id,
                name=segment.text,
                labels={SegmentNode.get_identifying_label()},
                segment=segment,
            )
            return segmentNode
        except Neo4jError as e:
            logger.error(f"Failed to insert segment node into Neo4j: {e.message}")

    def add_entity(
        self,
        entity_id: str,
        entity_candidate: MathEntityCandidate,
        segmentNode: SegmentNode,
    ):
        try:
            self.driver.execute_query(
                """
                CREATE (e:LectureEntity {
                    id: $id,
                    name: $name,
                    label: $label,
                    text: $text,
                    aliases: $aliases,
                    created_at: datetime()
                })
                WITH e
                Match(s:Segment {id: $segment_id})
                MERGE (s)-[:MENTIONS_ENTITY]->(e)
                """,
                id=entity_id,
                name=entity_candidate.text.lower(),
                label=entity_candidate.label,
                text=entity_candidate.text,
                aliases=[entity_candidate.text.lower()],
                segment_id=segmentNode.graph_id,
                database_=self.database_name,
            )

            logger.debug(f"Inserted entity into Neo4j: {entity_candidate.text}")

        except Neo4jError as e:
            logger.error(f"Failed to insert entity into Neo4j: {e.message}")

    def merge_entity(self, entity_id: str, entity_alias: str, segmentNode: SegmentNode):
        try:
            self.driver.execute_query(
                """
                MATCH (e:LectureEntity {id: $id})
                MATCH (s:Segment {id: $segment_id})
                SET e:LectureEntity,
                    e.aliases = coalesce(e.aliases, []) + $alias,
                    e.updated_at = datetime()
                MERGE (s)-[:MENTIONS_ENTITY]->(e)
                """,
                id=entity_id,
                alias=[entity_alias],
                segment_id=segmentNode.graph_id,
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
        MATCH (e1:LectureEntity {{id: $entity_1_id}})
        MATCH (e2:LectureEntity {{id: $entity_2_id}})
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
        MATCH (:LectureEntity {{id: $entity_1_id}})-[r:{sanitized_relationship}]->(:LectureEntity {{id: $entity_2_id}})
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

    def insert_wikidata_concepts(
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
            e.wikipedia_url = row.wikipedia_url,
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
