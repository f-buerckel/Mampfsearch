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
    Topic,
    TopicNode,
    VideoLocation,
)
from mampfsearch.utils.schema import nodeLabels, relationships

from typing import Optional, List, Dict, Any, Type

logger = logging.getLogger(__name__)


class Neo4jGraphStorage(BaseGraphStorage):
    def __init__(self, url: str, user: str, password: str, database_name: str):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.database_name = database_name

    def get_segments_of_lecture(self, lectureNode: LectureNode) -> List[SegmentNode]:
        cypher = f"""
        MATCH (l:{nodeLabels["segment"]} {{id: $lecture_id}})-[:{relationships["has_segment"]}]->(s:{nodeLabels["segment"]})
        RETURN s, labels(s) AS labels
        ORDER BY s.position ASC
        """
        result, _, _ = self.driver.execute_query(
            cypher,
            lecture_id=lectureNode.graph_id,
            database_=self.database_name,
        )
        segmentNodes = []
        for record in result:
            props = dict(record["s"])
            node = SegmentNode(
                graph_id=props["id"],
                name=props["name"],
                labels=set(record.get("labels", [])),
                segment=Segment(
                    text=props.get("text"),
                    location=VideoLocation(
                        start_time=0, end_time=0
                    ),  # TODO: Actually fix this
                    position=props.get("position", 0),
                ),
            )
            segmentNodes.append(node)

        return segmentNodes

    def update_global_density(self):
        cypher = f"""
        // 1. Calculate total number of entity mentions across the ENTIRE database
        MATCH ()-[r:{relationships["mentions_entity"]}]->(:{nodeLabels["lecture_entity"]})
        WITH count(r) as total_global_mentions

        // 2. For each entity, calculate its specific share of that total
        MATCH (:{nodeLabels["segment"]})-[r:{relationships["mentions_entity"]}]->(e:{nodeLabels["lecture_entity"]})
        WITH e, count(r) as entity_global_count, total_global_mentions

        // 3. Set the "Background Probability" on the node
        // e.g., If "Vector" is 1% of all mentions, global_mention_ratio = 0.01
        SET e.global_density = toFloat(entity_global_count) / total_global_mentions
        """
        self.driver.execute_query(
            cypher,
            database_=self.database_name,
        )
        logger.info("Updated global mention ratios for all entities.")

    def get_local_and_global_density(
        self, segmentNodes: List[SegmentNode]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns:
            Dict[entity_id, {
                "local_density": float,
                "global_density": float
            }]
        """

        segment_ids = [node.graph_id for node in segmentNodes]

        cypher = f"""
        MATCH (s:{nodeLabels["segment"]})-[r:{relationships["mentions_entity"]}]->(e:{nodeLabels["lecture_entity"]})
        WHERE s.id IN $segment_ids
        
        // 1. Count mentions per entity in these segments
        WITH e, count(r) as local_count
        
        // 2. Calculate the TOTAL mentions in these segments
        // We collect the stats to not lose the information after the aggregation and later unwind to recover
        WITH collect({{entity: e, count: local_count}}) as stats, sum(local_count) as total_mentions
        
        // 3. Unwind and calculate Local Density
        UNWIND stats as item
        RETURN item.entity.id AS entity_id, 
               
               // Calculate Local Density: (This Entity Count / Total Counts)
               toFloat(item.count) / total_mentions AS local_density,
               
               // Retrieve pre-calculated Global Background Probability
               coalesce(item.entity.global_mention_ratio, 0.0) AS global_density
        """

        result, _, _ = self.driver.execute_query(
            cypher,
            segment_ids=segment_ids,
            database_=self.database_name,
        )

        entity_metrics = {}

        for record in result:
            entity_id = record["entity_id"]

            entity_metrics[entity_id] = {
                "local_density": record["local_density"],
                "global_density": record["global_density"],
            }

        return entity_metrics

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
                f"""
                MERGE (c:{nodeLabels["course"]} {{id: $id}})
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
                f"""
                MATCH (c:{nodeLabels["course"]} {{id: $course_id}})
                MERGE (l:{nodeLabels["lecture"]} {{id: $id}})
                SET l.name = $params.name,
                    l.position = $params.position,
                    l.description = $params.description,
                    l.upload_date = $params.upload_date
                MERGE (c)-[:{relationships["has_lecture"]}]->(l)
                
                // create IS_SUCCESSOR relationship
                WITH c, l
                MATCH (prev:{nodeLabels["lecture"]})<-[:{relationships["has_lecture"]}]-(c)
                WHERE prev.position = l.position - 1
                MERGE (prev)-[:{relationships["is_successor"]}]->(l)
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
                f"""
                MATCH (c:{nodeLabels["course"]} {{id: $course_id}})
                MERGE (p:{nodeLabels["pdf_file"]} {{id: $id}})
                SET p.name = $params.name,
                    p.upload_date = $params.upload_date,
                    p.description = $params.description
                MERGE (c)-[:{relationships["has_pfd_file"]}]->(p)
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
                f"""
                MATCH (p:{nodeLabels["pdf_file"]} {{id: $pdf_file_id}})
                MERGE (pa:{nodeLabels["passage"]} {{id: $id}})
                SET pa.name = $params.text,
                    pa.text = $params.text,
                    pa.location = $location
                MERGE (p)-[:{relationships["has_passage"]}]->(pa)
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

    def add_segment_node(
        self, segment: Segment, lectureNode: LectureNode
    ) -> SegmentNode:
        try:
            segment_values = segment.model_dump()
            segment_id = str(uuid.uuid4())
            self.driver.execute_query(
                f"""
                MATCH (l:{nodeLabels["lecture"]} {{id: $lecture_id}})
                MERGE (s:{nodeLabels["segment"]} {{id: $id}})
                SET s.name = $params.text,
                    s.text = $params.text,
                    s.location = $location,
                    s.position = $params.position
                MERGE (l)-[:{relationships["has_segment"]}]->(s)
                
                // create IS_SUCCESSOR relationship
                WITH l, s
                MATCH (prev:{nodeLabels["segment"]})<-[:{relationships["has_segment"]}]-(l)
                WHERE prev.position = s.position - 1
                MERGE (prev)-[:{relationships["is_successor"]}]->(s)
                """,
                lecture_id=lectureNode.graph_id,
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

    def add_topic_node(self, topic: Topic) -> TopicNode:
        try:
            topic_values = topic.model_dump()
            topic_id = str(topic_values.get("uri", str(uuid.uuid4())))

            self.driver.execute_query(
                f"""
                MERGE (t:{nodeLabels["topic"]} {{id: $id}})
                SET t.name = $params.name,
                    t.uri = $params.uri,
                    t.description = $params.description,
                    t.wikipedia_url = $params.wikipedia_url
                """,
                params=topic_values,
                id=topic_id,
                database_=self.database_name,
            )
            logger.debug(f"Inserted topic node into Neo4j: {topic_values['name']}")

            topicNode = TopicNode(
                graph_id=topic_id,
                name=topic.name,
                labels={TopicNode.get_identifying_label()},
                topic=topic,
            )
            return topicNode

        except Neo4jError as e:
            logger.error(f"Failed to insert topic node into Neo4j: {e.message}")
            return None

    def add_entity(
        self,
        entity_id: str,
        entity_candidate: MathEntityCandidate,
        segmentNode: SegmentNode,
    ):
        try:
            self.driver.execute_query(
                f"""
                CREATE (e:{nodeLabels["lecture_entity"]} {{
                    id: $id,
                    name: $name,
                    label: $label,
                    text: $text,
                    aliases: $aliases,
                    created_at: datetime()
                }})
                WITH e
                Match(s:{nodeLabels["segment"]} {{id: $segment_id}})
                MERGE (s)-[:{relationships["mentions_entity"]}]->(e)
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
                f"""
                MATCH (e {{id: $id}})
                MATCH (s:{nodeLabels["segment"]} {{id: $segment_id}})
                SET e:{nodeLabels["lecture_entity"]},
                    e.aliases = coalesce(e.aliases, []) + $alias,
                    e.updated_at = datetime()
                MERGE (s)-[:{relationships["mentions_entity"]}]->(e)
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
        if relationship not in relationships:
            logger.warning(f"Relationship '{relationship}' not recognized.")
            raise ValueError(f"Relationship '{relationship}' not recognized.")

        cypher = f"""
        MATCH (e1:{nodeLabels["lecture_entity"]} {{id: $entity_1_id}})
        MATCH (e2:{nodeLabels["lecture_entity"]} {{id: $entity_2_id}})
        MERGE (e1)-[r:{relationship} {{id: $relationship_id}}]->(e2)
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
                    f"Inserted/Merged relationship {relationship} "
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
        MATCH (:{nodeLabels["lecture_entity"]} {{id: $entity_1_id}})-[r:{sanitized_relationship}]->(:{nodeLabels["lecture_entity"]} {{id: $entity_2_id}})
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
        #   :{nodeLabels['wikidata_entity']} (for filtering source),
        #   :<SafeLabel> (specific type)

        cypher = f"""
        UNWIND $batch AS row
        MERGE (e:Entity {{id: row.uri}})
        // Only set created_at if the node is effectively new
        ON CREATE SET e.created_at = datetime()
        SET e:{nodeLabels["wikidata_entity"]},
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
        MATCH (source {{id: row.source}})
        MATCH (target {{id: row.target}})
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
                f"Failed to batch insert Wikidata relationships (:{rel_type}): {e.message}"
            )
            return False
