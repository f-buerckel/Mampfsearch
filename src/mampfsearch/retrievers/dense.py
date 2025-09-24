from .base import BaseRetriever
from mampfsearch.utils.models import RetrievalItem
from typing import List
from mampfsearch.utils import config

class DenseRetriever(BaseRetriever):
    def retrieve(self, query: str, collection_name: str, limit: int) -> List[RetrievalItem]:
        client = config.get_qdrant_client()
        model = config.get_embedding_model()
        
        query_embedding = model.encode(
            [query],
            return_dense=True
        )
        
        points = client.query_points(
            collection_name=collection_name,
            query=query_embedding["dense_vecs"][0],
            using="dense",
            limit=limit,
            with_payload=True
        )
        
        return [RetrievalItem.from_qdrant_point(point) for point in points.points]