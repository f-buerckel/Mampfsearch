import time
import logging
from collections import defaultdict
from typing import Dict, List, Iterable, Set
import uuid

from SPARQLWrapper import SPARQLWrapper, JSON, POST

from mampfsearch.utils import config
from mampfsearch.utils.models import MathEntity, Topic, TopicNode
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 4000
REL_CHUNK_SIZE = 500  # Smaller batch size for relationships to avoid URL length limits
USER_AGENT = "MathKnowledgeGraphBot/1.0 (fbuerckel@mathi.uni-heidelberg.de)"


class WikidataMathExtractor:
    """
    Extracts math-related entities from Wikidata and loads them into the graph. Then, it matches relationships strictly between the extracted entities.
    """

    # Mapping Wikidata Properties (P-Codes) to Neo4j Relationship Types
    REL_MAPPING = {
        "http://www.wikidata.org/prop/direct/P279": "SUBCLASS_OF",
        "http://www.wikidata.org/prop/direct/P361": "PART_OF",
        "http://www.wikidata.org/prop/direct/P527": "HAS_PART",
        "http://www.wikidata.org/prop/direct/P2579": "STUDIED_BY",
        "http://www.wikidata.org/prop/direct/P101": "IN_FIELD",
        "http://www.wikidata.org/prop/direct/P1343": "DESCRIBED_BY",
        "http://www.wikidata.org/prop/direct/P31": "INSTANCE_OF",
    }

    ROOT_CLASSES = [
        "Q24034552",
        "Q246672",
        "Q114425676",
        "Q65943",
        "Q319141",
        "Q748349",
        "Q20026918",
        "Q1936384",
        "Q11348",
        "Q4516355",
        "Q200726",
        "Q122488935",
        "Q1166618",
        "Q12482",
        "Q217413",
        "Q467606",
        "Q1056428",
        "Q82571",
        "Q874429",
        "Q1208658",
        "Q903820",
        "Q727659",
        "Q13220368",
        "Q579978",
        "Q12479",
        "Q613048",
        "Q10843274",
        "Q7754",
        "Q854531",
        "Q193756",
        "Q190549",
        "Q876215",
        "Q15614122",
        "Q5275326",
        "Q8087",
        "Q15210169",
        "Q180969",
        "Q42989",
        "Q212803",
        "Q76592",
        "Q24175351",
        "Q8789",
        "Q5862903",
        "Q865811",
        "Q745328",
        "Q44455",
        "Q11216",
        "Q141495",
        "Q638328",
        "Q131222",
    ]

    def __init__(
        self,
        endpoint: str = WIKIDATA_ENDPOINT,
        batch_size: int = BATCH_SIZE,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.user_agent = user_agent

        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", self.user_agent)
        self.sparql.setMethod(POST)

        self.graph_storage = config.get_graph_storage()

        # all extracted entitiy URIs
        self.seen_uris: Set[str] = set()

    def extract_entities(self) -> int:
        """
        Extract wikidata entities bsaed and insert them into the graph database.
        """
        offset = 0
        total_processed = 0
        logger.info("Starting Wikidata math entity import...")

        while True:
            logger.info("Fetching batch at offset %d...", offset)
            wikidata_result = self._fetch_batch(offset)

            if not wikidata_result:
                logger.info("No more data returned from Wikidata.")
                break

            count = self._process_and_load_entities(wikidata_result)
            total_processed += count

            offset += self.batch_size
            time.sleep(1)

        logger.info(
            "Entity import complete. Total entities processed: %d", total_processed
        )
        return total_processed

    def _get_topics_query(self) -> str:
        values = " ".join(f"wd:{rc}" for rc in self.ROOT_CLASSES)
        return f"""
        SELECT ?rootClass ?rootLabel ?rootDesc ?article_en WHERE {{
          VALUES ?rootClass {{ {values} }}

          ?rootClass rdfs:label ?rootLabel .
          FILTER(LANG(?rootLabel) = "en")

          OPTIONAL {{
            ?rootClass schema:description ?rootDesc .
            FILTER(LANG(?rootDesc) = "en")
          }}

          OPTIONAL {{
            ?article_en schema:about ?rootClass ;
                        schema:isPartOf <https://en.wikipedia.org/> .
          }}
        }}
        """

    def extract_topics(self) -> None:
        query = self._get_topics_query()
        self.sparql.setQuery(query)
        results = self.sparql.query().convert()
        topics = results["results"]["bindings"]

        for t in topics:
            uri = t["rootClass"]["value"]
            name = t["rootLabel"]["value"]
            description = t.get("rootDesc", {}).get("value")
            wikipedia_url = t.get("article_en", {}).get("value")

            topic = Topic(
                name=name,
                uri=uri,
                description=description,
                wikipedia_url=wikipedia_url,
            )
            self.graph_storage.add_topic_node(topic)
            self.seen_uris.add(uri)

    def extract_and_insert(self) -> None:
        """
        1) Extract and insert all topics
        2) Extract and insert all entities that are somehow related to topics (rootClassees)
        3) Partition extracted entities into smaller chunks, then retrieve all relationships of nodes in a chunk and insert them.
        """

        self.extract_topics()
        self.extract_entities()

        if not self.seen_uris:
            logger.info("No URIs collected skipping extraction")
            return

        logger.info("Starting relationship extraction for %d URIs", len(self.seen_uris))

        # 3. Get the Relationships (Chunked to not overwhelm wikidata api)
        chunk_count = 0
        total_rels = 0

        chunks = [
            self.seen_uris[i : i + REL_CHUNK_SIZE]
            for i in range(0, len(self.seen_uris), REL_CHUNK_SIZE)
        ]
        for chunk in chunks:
            chunk_count += 1

            # Fetch raw connections from Wikidata for this chunk
            rel_bindings = self._fetch_relationships_for_chunk(chunk)

            if rel_bindings:
                inserted = self._process_and_load_relationships(rel_bindings)
                total_rels += inserted

            if chunk_count % 10 == 0:
                logger.info(f"Processed {chunk_count} chunks of URIs...")

            time.sleep(0.5)  # just to be polite

        logger.info(
            f"Relationship extraction complete. Total relationships created: {total_rels}"
        )

    def _get_sparql_query(self, limit: int, offset: int) -> str:
        """Build the entity SPARQL query for a single page."""
        rootClasses = " ".join(f"wd:{rc}" for rc in self.ROOT_CLASSES)
        return f"""
        SELECT 
          ?item 
          ?itemLabel 
          (SAMPLE(?rootLabel) AS ?rootLabel) 
          (SAMPLE(?formula) AS ?formula) 
          (SAMPLE(?desc) AS ?desc) 
          (SAMPLE(?article_en) AS ?wikipedia_url)
          WHERE {{
        VALUES ?rootClass {{
        {rootClasses}
        }}
          
        ?item (wdt:P31|wdt:P279|wdt:P2579)? / (wdt:P31|wdt:P279|wdt:P2579)? ?rootClass .
          
        # EXCLUSIONS
        MINUS {{ ?class wdt:P279* wd:Q11563}}
        MINUS {{ ?item wdt:P31 wd:Q28920044 }}
        MINUS {{ ?item wdt:P31 wd:Q133250 }}
        MINUS {{ ?item wdt:P31 wd:Q29431432 }}
        
        # WIKIPEDIA FILTER
        ?article_en schema:about ?item ;
                    schema:isPartOf <https://en.wikipedia.org/> .

        # LABELS
        ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en")
        ?rootClass rdfs:label ?rootLabel . FILTER(LANG(?rootLabel) = "en")
        OPTIONAL {{ ?item wdt:P2534 ?formula . }}
        OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }}
        }}
        GROUP BY ?item ?itemLabel
        ORDER BY ?item 
        LIMIT {limit}
        OFFSET {offset}
        """

    def _fetch_batch(self, offset: int) -> List[Dict]:
        query = self._get_sparql_query(self.batch_size, offset)
        self.sparql.setQuery(query)

        for attempt in range(3):
            try:
                results = self.sparql.query().convert()
                return results["results"]["bindings"]
            except Exception as e:
                logger.warning(
                    "Attempt %d failed at offset %d: %s", attempt + 1, offset, e
                )
                time.sleep(2)
        return []

    def _process_and_load_entities(self, wikidata_result: List[Dict]) -> int:
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        for b in wikidata_result:
            root_label = b["rootLabel"]["value"]
            uri = b["item"]["value"]
            wiki_url = b.get("wikipedia_url", {}).get("value")

            entity = {
                "uri": uri,
                "name": b["itemLabel"]["value"],
                "formula": b.get("formula", {}).get("value"),
                "description": b.get("desc", {}).get("value"),
                "wikipedia_url": wiki_url,
            }

            grouped[root_label].append(entity)
            self.seen_uris.add(uri)

        total_inserted = 0
        for label, batch in grouped.items():
            success = self.graph_storage.insert_wikidata_concepts(batch, label)
            if success:
                total_inserted += len(batch)
                self._insert_entities_into_vector_store(label, batch)
        return total_inserted

    def _insert_entities_into_vector_store(
        self, label: str, entities: List[Dict]
    ) -> None:
        """
        Inserts entities into the vector store (Qdrant) with embeddings.
        """
        model = config.get_embedding_model()
        vector_storage = config.get_vector_storage()

        points = []

        for entity in entities:
            embedding = model.encode(entity["name"], return_dense=True)
            entity_model = MathEntity(
                name=entity["name"],
                label=label,
                uri=entity["uri"],
                formula=entity.get("formula"),
                description=entity.get("description"),
                wikipedia_url=entity.get("wikipedia_url"),
            )
            payload = entity_model.model_dump()
            point = PointStruct(
                id=str(uuid.uuid4()),
                payload=payload,
                vector={"dense": embedding["dense_vecs"]},
            )
            points.append(point)

        vector_storage.upsert(
            collection_name=config.ENTITIES_COLLECTION_NAME,
            points=points,
        )

    def _get_relationship_query_for_chunk(self, uris: List[str]) -> str:
        """
        Sparsql query to fetch relationships of all specified URIs for the mapped relationships.
        """
        values_str = " ".join(f"<{u}>" for u in uris)

        valid_props = " ".join(
            f"wdt:{p.split('/')[-1]}" for p in self.REL_MAPPING.keys()
        )  # all properties we care about

        return f"""
    SELECT ?source ?p ?target WHERE {{
      VALUES ?source {{ {values_str} }}
      VALUES ?p {{ {valid_props} }}
      ?source ?p ?target .
    }}
        """

    def _fetch_relationships_for_chunk(self, chunk: List[str]) -> List[Dict]:
        query = self._get_relationship_query_for_chunk(chunk)
        self.sparql.setQuery(query)

        try:
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(f"Failed to fetch relationships for chunk: {e}")
            return []

    def _process_and_load_relationships(self, wikidata_result: List[Dict]) -> int:
        """
        Filters relationships to ensure the TARGET is also in our database,
        then inserts them.
        """
        grouped_rels = defaultdict(list)

        for b in wikidata_result:
            target_uri = b["target"]["value"]

            # Only import if the target is a node we have actually extracted.
            if target_uri not in self.seen_uris:
                continue

            p_code = b["p"]["value"]

            neo4j_rel_type = self.REL_MAPPING.get(p_code)

            if neo4j_rel_type:
                rel_data = {"source": b["source"]["value"], "target": target_uri}
                grouped_rels[neo4j_rel_type].append(rel_data)

        total_inserted = 0
        for rel_type, batch in grouped_rels.items():
            success = self.graph_storage.insert_grouped_relationships(rel_type, batch)
            if success:
                total_inserted += len(batch)

        return total_inserted


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    extractor = WikidataMathExtractor()
    extractor.extract_and_insert()


if __name__ == "__main__":
    main()
