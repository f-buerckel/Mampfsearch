import time
import logging
from collections import defaultdict
from typing import Dict, List, Iterable, Set

from SPARQLWrapper import SPARQLWrapper, JSON, POST

from mampfsearch.utils import config

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 15000
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
        self._seen_uris: Set[str] = set()

    @property
    def seen_uris(self) -> Set[str]:
        return self._seen_uris

    def extract_entities(self) -> int:
        """
        Run the entity extraction + loading, return number of inserted/updated entities.
        """
        offset = 0
        total_processed = 0
        logger.info("Starting Wikidata math entity import...")

        while True:
            logger.info("Fetching batch at offset %d...", offset)
            raw_bindings = self._fetch_batch(offset)

            if not raw_bindings:
                logger.info("No more data returned from Wikidata.")
                break

            count = self._process_and_load_entities(raw_bindings)
            total_processed += count

            offset += self.batch_size
            time.sleep(1)

        logger.info(
            "Entity import complete. Total entities processed: %d", total_processed
        )
        return total_processed

    def run_with_relationships(self) -> None:
        """
        1) Extract + load all entities.
        2) Use collected URIs to query and load relationships using the Closed World assumption.
        """
        # 1. Get the Nodes
        self.extract_entities()

        if not self._seen_uris:
            logger.info("No URIs collected; skipping relationship extraction.")
            return

        logger.info(
            "Starting relationship extraction for %d URIs...", len(self._seen_uris)
        )

        # 2. Get the Relationships (Chunked to not overwhelm wikidata api)
        chunk_count = 0
        total_rels = 0

        for chunk in self._chunk_uris(self._seen_uris, size=REL_CHUNK_SIZE):
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
        return f"""
        SELECT 
          ?item 
          ?itemLabel 
          (SAMPLE(?rootLabel) AS ?rootLabel) 
          (SAMPLE(?formula) AS ?formula) 
          (SAMPLE(?desc) AS ?desc) 
        WHERE {{
          VALUES ?rootClass {{
            wd:Q24034552 wd:Q246672 wd:Q114425676 wd:Q65943 wd:Q319141 
            wd:Q748349 wd:Q20026918 wd:Q1936384 wd:Q11348 wd:Q4516355 
            wd:Q200726 wd:Q122488935
          }}
          
          # PATHS
          {{ ?item wdt:P31|wdt:P279|wdt:P2579 ?rootClass . }}
          UNION
          {{ ?item (wdt:P31|wdt:P279|wdt:P2579) / (wdt:P31|wdt:P279|wdt:P2579) ?rootClass . }}
          
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
                time.sleep(2**attempt)
        return []

    def _process_and_load_entities(self, raw_bindings: List[Dict]) -> int:
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        for b in raw_bindings:
            root_label = b["rootLabel"]["value"]
            uri = b["item"]["value"]

            entity = {
                "uri": uri,
                "name": b["itemLabel"]["value"],
                "formula": b.get("formula", {}).get("value"),
                "description": b.get("desc", {}).get("value"),
            }

            grouped[root_label].append(entity)
            self._seen_uris.add(uri)

        total_inserted = 0
        for label, batch in grouped.items():
            success = self.graph_storage.batch_insert_wikidata_concepts(batch, label)
            if success:
                total_inserted += len(batch)
        return total_inserted

    @staticmethod
    def _chunk_uris(uris: Iterable[str], size: int) -> Iterable[List[str]]:
        chunk: List[str] = []
        for u in uris:
            chunk.append(u)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

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

    def _process_and_load_relationships(self, raw_bindings: List[Dict]) -> int:
        """
        Filters relationships to ensure the TARGET is also in our database,
        then inserts them.
        """
        grouped_rels = defaultdict(list)

        for b in raw_bindings:
            target_uri = b["target"]["value"]

            # Only import if the target is a node we have actually extracted.
            if target_uri not in self._seen_uris:
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
    extractor.run_with_relationships()


if __name__ == "__main__":
    main()
