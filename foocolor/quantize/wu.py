from enum import Enum
from typing import List, NamedTuple, Optional

import numpy as np

from ..util import color_util
from .quantizer import Quantizer, QuantizerResult


class QuantizerWu(Quantizer):
    def __init__(self, unique_pixels, counts) -> None:
        self._unique_pixels = unique_pixels
        self._counts = counts

        self._weights: Optional[np.ndarray] = None
        self._moments_r: Optional[np.ndarray] = None
        self._moments_g: Optional[np.ndarray] = None
        self._moments_b: Optional[np.ndarray] = None
        self._moments: Optional[np.ndarray] = None
        self._cubes: Optional[List[Box]] = None

    def quantize(self, max_colors: int) -> QuantizerResult:
        self._setup()
        self._construct_histogram()
        self._compute_moments()
        create_boxes_result = self._create_boxes(max_colors)
        results = self._create_result(create_boxes_result.result_count)
        return QuantizerResult({result: 0 for result in results})

    def _setup(self) -> None:
        self._weights = np.zeros(35937)
        self._moments_r = np.zeros(35937)
        self._moments_g = np.zeros(35937)
        self._moments_b = np.zeros(35937)
        self._moments = np.zeros(35937)
        self._cubes = [Box() for _ in range(256)]

    @staticmethod
    def _get_index(r, g, b):
        return (r << (5 * 2)) + (r << (5 + 1)) + (g << 5) + r + g + b

    def _construct_histogram(self):
        r, g, b = rgb = color_util.rgb_from_argb(self._unique_pixels)
        bits_to_remove = 8 - 5
        i_rgb = (rgb >> bits_to_remove) + 1
        indices = self._get_index(i_rgb[0], i_rgb[1], i_rgb[2])
        rgb_times_count = (rgb.T * self._counts[:, None]).T
        np.add.at(self._weights, indices, self._counts)
        np.add.at(self._moments_r, indices, rgb_times_count[0])
        np.add.at(self._moments_g, indices, rgb_times_count[1])
        np.add.at(self._moments_b, indices, rgb_times_count[2])
        np.add.at(self._moments, indices, (r**2 + g**2 + b**2) * self._counts)

    def _compute_moments(self):
        for r in range(1, 33):
            area = [0] * 33
            area_r = [0] * 33
            area_g = [0] * 33
            area_b = [0] * 33
            area2 = [0.0] * 33
            for g in range(1, 33):
                line = 0
                line_r = 0
                line_g = 0
                line_b = 0
                line2 = 0.0
                for b in range(1, 33):
                    index = self._get_index(r, g, b)
                    line += self._weights[index]
                    line_r += self._moments_r[index]
                    line_g += self._moments_g[index]
                    line_b += self._moments_b[index]
                    line2 += self._moments[index]

                    area[b] += line
                    area_r[b] += line_r
                    area_g[b] += line_g
                    area_b[b] += line_b
                    area2[b] += line2

                    previous_index = self._get_index(r - 1, g, b)
                    self._weights[index] = self._weights[previous_index] + area[b]
                    self._moments_r[index] = self._moments_r[previous_index] + area_r[b]
                    self._moments_g[index] = self._moments_g[previous_index] + area_g[b]
                    self._moments_b[index] = self._moments_b[previous_index] + area_b[b]
                    self._moments[index] = self._moments[previous_index] + area2[b]

    def _create_boxes(self, max_colors):
        max_color_count = max_colors
        self._cubes = [Box() for _ in range(max_color_count)]
        self._cubes[0] = Box(r0=0, r1=32, g0=0, g1=32, b0=0, b1=32, vol=0)

        volume_variance = [0.0] * max_color_count
        next = 0
        generated_color_count = max_color_count
        for i in range(1, max_color_count):
            if self._cut(self._cubes[next], self._cubes[i]):
                volume_variance[next] = (
                    self._variance(self._cubes[next])
                    if self._cubes[next].vol > 1
                    else 0.0
                )
                volume_variance[i] = (
                    self._variance(self._cubes[i]) if self._cubes[i].vol > 1 else 0.0
                )
            else:
                volume_variance[next] = 0.0
                i -= 1

            next = 0
            temp = volume_variance[0]
            for j in range(1, i + 1):
                if volume_variance[j] > temp:
                    temp = volume_variance[j]
                    next = j
            if temp <= 0.0:
                generated_color_count = i + 1
                break

        return CreateBoxesResult(
            requested_count=max_color_count, result_count=generated_color_count
        )

    def _create_result(self, color_count):
        colors = []
        for i in range(color_count):
            cube = self._cubes[i]
            weight = self._volume(cube, self._weights)
            if weight > 0:
                r = np.around(self._volume(cube, self._moments_r) / weight).astype(
                    np.uint8
                )
                g = np.around(self._volume(cube, self._moments_g) / weight).astype(
                    np.uint8
                )
                b = np.around(self._volume(cube, self._moments_b) / weight).astype(
                    np.uint8
                )
                color = color_util.argb_from_rgb(r, g, b)
                colors.append(color)
        return colors

    def _variance(self, cube):
        dr = self._volume(cube, self._moments_r)
        dg = self._volume(cube, self._moments_g)
        db = self._volume(cube, self._moments_b)
        xx = (
            self._moments[self._get_index(cube.r1, cube.g1, cube.b1)]
            - self._moments[self._get_index(cube.r1, cube.g1, cube.b0)]
            - self._moments[self._get_index(cube.r1, cube.g0, cube.b1)]
            + self._moments[self._get_index(cube.r1, cube.g0, cube.b0)]
            - self._moments[self._get_index(cube.r0, cube.g1, cube.b1)]
            + self._moments[self._get_index(cube.r0, cube.g1, cube.b0)]
            + self._moments[self._get_index(cube.r0, cube.g0, cube.b1)]
            - self._moments[self._get_index(cube.r0, cube.g0, cube.b0)]
        )

        hypotenuse = dr * dr + dg * dg + db * db
        volume_ = self._volume(cube, self._weights)
        return xx - hypotenuse / volume_

    def _cut(self, one, two):
        whole_r = self._volume(one, self._moments_r)
        whole_g = self._volume(one, self._moments_g)
        whole_b = self._volume(one, self._moments_b)
        whole_w = self._volume(one, self._weights)

        max_r_result = self._maximize(
            one, Direction.red, one.r0 + 1, one.r1, whole_r, whole_g, whole_b, whole_w
        )
        max_g_result = self._maximize(
            one, Direction.green, one.g0 + 1, one.g1, whole_r, whole_g, whole_b, whole_w
        )
        max_b_result = self._maximize(
            one, Direction.blue, one.b0 + 1, one.b1, whole_r, whole_g, whole_b, whole_w
        )

        cut_direction = None
        max_r = max_r_result.maximum
        max_g = max_g_result.maximum
        max_b = max_b_result.maximum
        if max_r >= max_g and max_r >= max_b:
            cut_direction = Direction.red
            if max_r_result.cut_location < 0:
                return False
        elif max_g >= max_r and max_g >= max_b:
            cut_direction = Direction.green
        else:
            cut_direction = Direction.blue

        two.r1 = one.r1
        two.g1 = one.g1
        two.b1 = one.b1

        if cut_direction == Direction.red:
            one.r1 = max_r_result.cut_location
            two.r0 = one.r1
            two.g0 = one.g0
            two.b0 = one.b0
        elif cut_direction == Direction.green:
            one.g1 = max_g_result.cut_location
            two.r0 = one.r0
            two.g0 = one.g1
            two.b0 = one.b0
        elif cut_direction == Direction.blue:
            one.b1 = max_b_result.cut_location
            two.r0 = one.r0
            two.g0 = one.g0
            two.b0 = one.b1
        else:
            raise Exception("unexpected direction {}".format(cut_direction))

        one.vol = (one.r1 - one.r0) * (one.g1 - one.g0) * (one.b1 - one.b0)
        two.vol = (two.r1 - two.r0) * (two.g1 - two.g0) * (two.b1 - two.b0)
        return True

    def _maximize(
        self, cube, direction, first, last, whole_r, whole_g, whole_b, whole_w
    ):
        bottom_r = self._bottom(cube, direction, self._moments_r)
        bottom_g = self._bottom(cube, direction, self._moments_g)
        bottom_b = self._bottom(cube, direction, self._moments_b)
        bottom_w = self._bottom(cube, direction, self._weights)

        max = 0.0
        cut = -1

        for i in range(first, last):
            half_r = bottom_r + self._top(cube, direction, i, self._moments_r)
            half_g = bottom_g + self._top(cube, direction, i, self._moments_g)
            half_b = bottom_b + self._top(cube, direction, i, self._moments_b)
            half_w = bottom_w + self._top(cube, direction, i, self._weights)

            if half_w == 0:
                continue

            temp_numerator = (half_r * half_r) + (half_g * half_g) + (half_b * half_b)
            temp_denominator = half_w
            temp = temp_numerator / temp_denominator

            half_r = whole_r - half_r
            half_g = whole_g - half_g
            half_b = whole_b - half_b
            half_w = whole_w - half_w
            if half_w == 0:
                continue
            temp_numerator = (half_r * half_r) + (half_g * half_g) + (half_b * half_b)
            temp_denominator = half_w
            temp += temp_numerator / temp_denominator

            if temp > max:
                max = temp
                cut = i

        return MaximizeResult(cut_location=cut, maximum=max)

    def _volume(self, cube, moment):
        return (
            moment[self._get_index(cube.r1, cube.g1, cube.b1)]
            - moment[self._get_index(cube.r1, cube.g1, cube.b0)]
            - moment[self._get_index(cube.r1, cube.g0, cube.b1)]
            + moment[self._get_index(cube.r1, cube.g0, cube.b0)]
            - moment[self._get_index(cube.r0, cube.g1, cube.b1)]
            + moment[self._get_index(cube.r0, cube.g1, cube.b0)]
            + moment[self._get_index(cube.r0, cube.g0, cube.b1)]
            - moment[self._get_index(cube.r0, cube.g0, cube.b0)]
        )

    def _bottom(self, cube, direction, moment):
        if direction == Direction.red:
            return (
                -moment[self._get_index(cube.r0, cube.g1, cube.b1)]
                + moment[self._get_index(cube.r0, cube.g1, cube.b0)]
                + moment[self._get_index(cube.r0, cube.g0, cube.b1)]
                - moment[self._get_index(cube.r0, cube.g0, cube.b0)]
            )
        elif direction == Direction.green:
            return (
                -moment[self._get_index(cube.r1, cube.g0, cube.b1)]
                + moment[self._get_index(cube.r1, cube.g0, cube.b0)]
                + moment[self._get_index(cube.r0, cube.g0, cube.b1)]
                - moment[self._get_index(cube.r0, cube.g0, cube.b0)]
            )
        elif direction == Direction.blue:
            return (
                -moment[self._get_index(cube.r1, cube.g1, cube.b0)]
                + moment[self._get_index(cube.r1, cube.g0, cube.b0)]
                + moment[self._get_index(cube.r0, cube.g1, cube.b0)]
                - moment[self._get_index(cube.r0, cube.g0, cube.b0)]
            )
        else:
            raise Exception("unexpected direction {}".format(direction))

    def _top(self, cube, direction, position, moment):
        if direction == Direction.red:
            return (
                moment[self._get_index(position, cube.g1, cube.b1)]
                - moment[self._get_index(position, cube.g1, cube.b0)]
                - moment[self._get_index(position, cube.g0, cube.b1)]
                + moment[self._get_index(position, cube.g0, cube.b0)]
            )
        elif direction == Direction.green:
            return (
                moment[self._get_index(cube.r1, position, cube.b1)]
                - moment[self._get_index(cube.r1, position, cube.b0)]
                - moment[self._get_index(cube.r0, position, cube.b1)]
                + moment[self._get_index(cube.r0, position, cube.b0)]
            )
        elif direction == Direction.blue:
            return (
                moment[self._get_index(cube.r1, cube.g1, position)]
                - moment[self._get_index(cube.r1, cube.g0, position)]
                - moment[self._get_index(cube.r0, cube.g1, position)]
                + moment[self._get_index(cube.r0, cube.g0, position)]
            )
        else:
            raise Exception("unexpected direction {}".format(direction))


class Direction(Enum):
    red = 1
    green = 2
    blue = 3


class MaximizeResult(NamedTuple):
    cut_location: int
    maximum: float


class CreateBoxesResult(NamedTuple):
    requested_count: int
    result_count: int


class Box:
    def __init__(
        self,
        r0: int = 0,
        r1: int = 0,
        g0: int = 0,
        g1: int = 0,
        b0: int = 0,
        b1: int = 0,
        vol: int = 0,
    ):
        self.r0 = r0
        self.r1 = r1
        self.g0 = g0
        self.g1 = g1
        self.b0 = b0
        self.b1 = b1
        self.vol = vol

    def __str__(self):
        return f"Box: R {self.r0} -> {self.r1} G  {self.g0} -> {self.g1} B {self.b0} -> {self.b1} VOL = {self.vol}"
