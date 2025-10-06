from mampfsearch.utils.models import EntityRetrievalItem
from typing import List
from mampfsearch.utils import config

class EntityRetriever():
    def retrieve(self, query: str, limit: int) -> List[EntityRetrievalItem]:
        client = config.get_qdrant_client()
        model = config.get_embedding_model()
        
        query_embedding = model.encode(
            [query],
            return_dense=True
        )
        
        points = client.query_points(
            collection_name=config.ENTITIES_COLLECTION_NAME,
            query=query_embedding["dense_vecs"][0],
            using="dense",
            limit=limit,
            with_payload=True
        )
        
        return [EntityRetrievalItem.from_qdrant_point(point) for point in points.points]