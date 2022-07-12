from abc import ABC, abstractmethod
from typing import List


class PointProvider(ABC):
    @abstractmethod
    def from_int(self, argb: int) -> List[float]:
        pass

    @abstractmethod
    def to_int(self, point: List[float]) -> int:
        pass

    @abstractmethod
    def distance(self, a: List[float], b: List[float]) -> float:
        pass
