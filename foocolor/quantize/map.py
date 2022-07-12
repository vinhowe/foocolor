import numpy as np

from .quantizer import Quantizer, QuantizerResult


class QuantizerMap(Quantizer):
    def quantize(self, pixels: np.ndarray, max_colors: int) -> QuantizerResult:
        counts = np.unique(pixels[pixels[:, 3] < 255], return_counts=True)
        count_by_color = dict(zip(counts[0], counts[1]))

        return QuantizerResult(count_by_color)
