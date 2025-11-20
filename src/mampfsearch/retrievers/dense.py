from .base import BaseRetriever
from mampfsearch.utils.models import LectureRetrievalItem
from typing import List
from mampfsearch.utils import config


class DenseRetriever(BaseRetriever):
    def retrieve(
        self, query: str, collection_name: str, limit: int
    ) -> List[LectureRetrievalItem]:
        client = config.get_vector_storage()
        model = config.get_embedding_model()

        query_embedding = model.encode([query], return_dense=True)

        points = client.search(
            collection_name=config.LECTURE_COLLECTION_NAME,
            query_vector=query_embedding["dense_vecs"][0],
            limit=limit,
            with_payload=True,
        )

        return [LectureRetrievalItem.from_qdrant_point(point) for point in points]
