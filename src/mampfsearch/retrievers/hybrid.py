from .base import BaseRetriever
from mampfsearch.utils.models import LectureRetrievalItem
from typing import List
from mampfsearch.utils import config, helpers

class HybridRetriever(BaseRetriever):
    def retrieve(self, query: str, collection_name: str, limit: int) -> List[LectureRetrievalItem]:
        from qdrant_client import models
        client = config.get_qdrant_client()
        model = config.get_embedding_model()
        
        query_embedding = model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
        )

        prefetch = [
            models.Prefetch(
                query=query_embedding["dense_vecs"][0],
                using="dense",
                limit=config.PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=helpers.convert_sparse_vector(query_embedding["lexical_weights"][0]),
                using="sparse",
                limit=config.PREFETCH_LIMIT,
            )
        ]
        
        points = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        
        return [LectureRetrievalItem.from_qdrant_point(point) for point in points.points]