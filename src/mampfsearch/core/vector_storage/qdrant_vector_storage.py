from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from .base_vector_storage import BaseVectorStorage
import logging
import uuid

logger = logging.getLogger(__name__)

class QdrantVectorStorage(BaseVectorStorage):
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)

    def upsert(self, collection_name: str, points: List[PointStruct]):
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5, with_payload: bool = True) -> List[Any]:
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            with_payload=with_payload,
        )
        return results.points

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)

    def list_collections(self) -> List[str]:
        collections = self.client.get_collections().collections
        return [c.name for c in collections]

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        return self.client.get_collection(collection_name)

    def create_lecture_collection(self, collection_name: str, dimension: int) -> Dict[str, Any]:
        return self._create_collection(collection_name, dimension)

    def create_entity_collection(self, collection_name: str, dimension: int) -> Dict[str, Any]:
        return self._create_collection(collection_name, dimension)

    def _create_collection(self, collection_name: str, dimension: int) -> Dict[str, Any]:
        exists = self.collection_exists(collection_name)
        info = {
            "collection_name": collection_name,
            "exists": exists,
        }

        if exists:
            logger.info(f"Collection {collection_name} already exists")
            return info

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
                "colbert": models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams())
            },
        )

        logger.info(f"Created collection {collection_name} (vector dimension={dimension})")

        info.update(
            {
                "status": "exists",
                "vector_dimension": dimension,
            }
        )
        return info

    def upload_chunks(self, collection_name: str, vectors: List[Dict[str, Any]], payloads: List[Dict[str, Any]]):
        from mampfsearch.utils import helpers
        points = []
        for i, embedding in enumerate(vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload=payloads[i],
                    vector={
                        "dense": embedding["dense_vecs"],
                        "colbert": embedding["colbert_vecs"],
                        "sparse": helpers.convert_sparse_vector(
                            embedding["lexical_weights"]
                        ),
                    },
                )
            )
        
        self.upsert(collection_name, points)
        logger.info(f"Inserted {len(vectors)} vectors into collection {collection_name}")

    def list_entities(self, collection_name: str, label: Optional[str] = None, limit: int = 100) -> List[Any]:
        scroll_filter = None
        if label:
            scroll_filter = Filter(
                must=[FieldCondition(key="label", match=MatchValue(value=label))]
            )

        points, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        return points

    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)
