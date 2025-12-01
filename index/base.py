from abc import ABC, abstractmethod
from typing import Union, Set, List

SearchResult = Union[List[int], Set[int]]


class Index(ABC):
    @property
    @abstractmethod
    def search(self, *args, **kwargs) -> SearchResult:
        pass
