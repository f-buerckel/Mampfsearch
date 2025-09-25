from .base import BaseRetriever
from typing import List
from mampfsearch.utils import config
from mampfsearch.utils.models import RetrievalItem

class RerankerRetriever(BaseRetriever):
    
    def __init__(self, base_retriever: BaseRetriever, reranker: 'Reranker'):
        """
        Initialize the RerankerRetriever with a base retriever and a reranker.

        :param base_retriever: Base retriever to get initial results for reranking
        :param reranker: Reranker to rerank results from the base retriever
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def retrieve(self, query: str, collection_name: str, limit: int) -> List[RetrievalItem]:
        initial_points = self.base_retriever.retrieve(query, collection_name, config.PREFETCH_LIMIT)

        documents = [result.text for result in initial_points]
        reranked_documents = self.reranker.rank(query, documents)

        reranked_points = []
        for document in reranked_documents.results[:limit]:
            index = document.doc_id
            point = initial_points[index]

            # Create a new RetrievalItem with the reranked score and the original point's attributes
            new_point = RetrievalItem(
                score=document.score,
                text=point.text,
                lecture=point.lecture,
                lecture_position=point.lecture_position,
                start_time=point.start_time,
                end_time=point.end_time,
                position=point.position
            )
            reranked_points.append(new_point)

        
        return reranked_points

        
