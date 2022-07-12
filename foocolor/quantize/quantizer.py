import abc
from typing import Dict


class Quantizer(abc.ABC):
    @abc.abstractmethod
    def quantize(self, max_colors: int) -> "QuantizerResult":
        pass


class QuantizerResult:
    def __init__(
        self,
        color_to_count: Dict[int, int],
        input_pixel_to_cluster_pixel: Dict[int, int] = {},
    ):
        self.color_to_count = color_to_count
        self.input_pixel_to_cluster_pixel = input_pixel_to_cluster_pixel
