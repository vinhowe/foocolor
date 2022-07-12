import numpy as np

from ..util import color_util
from .point_provider import PointProvider


class PointProviderLab(PointProvider):
    def from_int(self, argb):
        return color_util.lab_from_argb(argb)

    def to_int(self, lab):
        return color_util.argb_from_lab(lab[0], lab[1], lab[2])

    def distance(self, one, two):
        # Standard CIE 1976 delta E formula also takes the square root, unneeded
        # here. This method is used by quantization algorithms to compare distance,
        # and the relative ordering is the same, with or without a square root.

        # This relatively minor optimization is helpful because this method is
        # called at least once for each pixel in an image.
        return np.sum((one - two) ** 2)
