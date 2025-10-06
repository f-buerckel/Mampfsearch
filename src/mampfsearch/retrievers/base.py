from abc import ABC, abstractmethod
from typing import List
from mampfsearch.utils.models import LectureRetrievalItem

class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str, collection_name: str, limit: int = 10) -> List[LectureRetrievalItem]:
        """
        Retrieve a list of LectureRetrievalItems based on the query.

        :param query: The search query.
        :param limit: The maximum number of results to return.
        :return: A list of LectureRetrievalItems.
        """
        pass
