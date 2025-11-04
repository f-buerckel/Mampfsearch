from abc import ABC, abstractmethod
from typing import List
from mampfsearch.utils.models import Chunk, EntityCandidate


class BaseNER(ABC):

   def extract(self, chunk: Chunk) -> List[EntityCandidate]:
       pass