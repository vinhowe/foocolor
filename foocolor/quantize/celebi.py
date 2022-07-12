import time

import numpy as np

from ..util import color_util
from .point_provider_lab import PointProviderLab
from .quantizer import Quantizer, QuantizerResult
from .wsmeans import QuantizerWsmeans
from .wu import QuantizerWu


class QuantizerCelebi(Quantizer):
    def __init__(self, pixels: np.ndarray) -> None:
        super().__init__()
        self._pixels = (
            pixels if pixels.shape[-1] == 3 else pixels[pixels[:, 3] == 255][:, :3]
        ).astype(np.int64)
        self._pixels = color_util.argb_from_rgb(
            self._pixels[:, 0], self._pixels[:, 1], self._pixels[:, 2]
        )

    def quantize(
        self,
        max_colors: int,
        return_input_pixel_to_cluster_pixel: bool = False,
    ) -> QuantizerResult:
        unique_pixels, counts = np.unique(self._pixels, return_counts=True, axis=-1)
        wu = QuantizerWu(unique_pixels, counts)
        wu_result = wu.quantize(max_colors)
        wsmeans = QuantizerWsmeans(unique_pixels, counts)
        wsmeans_result = wsmeans.quantize(
            max_colors,
            starting_clusters=list(wu_result.color_to_count.keys()),
            point_provider=PointProviderLab(),
            return_input_pixel_to_cluster_pixel=return_input_pixel_to_cluster_pixel,
        )
        return wsmeans_result
