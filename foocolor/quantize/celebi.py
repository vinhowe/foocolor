import time

import numpy as np

from foocolor.util import color_util

from .point_provider_lab import PointProviderLab
from .quantizer import Quantizer, QuantizerResult
from .wsmeans import QuantizerWsmeans
from .wu import QuantizerWu


class QuantizerCelebi(Quantizer):
    def quantize(
        self,
        pixels: np.ndarray,
        max_colors: int,
        return_input_pixel_to_cluster_pixel: bool = False,
    ) -> QuantizerResult:
        pixels = (
            pixels if pixels.shape[-1] == 3 else pixels[pixels[:, 3] == 255][:, :3]
        ).astype(np.int64)
        argb_pixels = color_util.argb_from_rgb(pixels[:, 0], pixels[:, 1], pixels[:, 2])
        unique_pixels, counts = np.unique(argb_pixels, return_counts=True, axis=-1)
        wu = QuantizerWu()
        wu_result = wu.quantize(unique_pixels, counts, max_colors)
        wsmeans_result = QuantizerWsmeans.quantize(
            unique_pixels,
            counts,
            max_colors,
            starting_clusters=list(wu_result.color_to_count.keys()),
            point_provider=PointProviderLab(),
            return_input_pixel_to_cluster_pixel=return_input_pixel_to_cluster_pixel,
        )
        return wsmeans_result
