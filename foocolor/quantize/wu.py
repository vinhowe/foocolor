from enum import Enum
from typing import NamedTuple

import numpy as np

from ..util import color_util
from .quantizer import Quantizer, QuantizerResult


class QuantizerWu(Quantizer):
    def __init__(self):
        self.weights = np.zeros(35937)
        self.momentsR = np.zeros(35937)
        self.momentsG = np.zeros(35937)
        self.momentsB = np.zeros(35937)
        self.moments = np.zeros(35937)
        self.cubes = [Box() for _ in range(256)]

    def quantize(self, unique_pixels, counts, color_count):
        self.construct_histogram(unique_pixels, counts)
        self.compute_moments()
        create_boxes_result = self.create_boxes(color_count)
        results = self.create_result(create_boxes_result.result_count)
        return QuantizerResult({result: 0 for result in results})

    @staticmethod
    def get_index(r, g, b):
        return (r << (5 * 2)) + (r << (5 + 1)) + (g << 5) + r + g + b

    def construct_histogram(self, pixels, counts):
        r, g, b = rgb = color_util.rgb_from_argb(pixels)
        bits_to_remove = 8 - 5
        i_rgb = (rgb >> bits_to_remove) + 1
        indices = self.get_index(i_rgb[0], i_rgb[1], i_rgb[2])
        rgb_times_count = (rgb.T * counts[:, None]).T
        np.add.at(self.weights, indices, counts)
        np.add.at(self.momentsR, indices, rgb_times_count[0])
        np.add.at(self.momentsG, indices, rgb_times_count[1])
        np.add.at(self.momentsB, indices, rgb_times_count[2])
        np.add.at(self.moments, indices, (r ** 2 + g ** 2 + b ** 2) * counts)

    def compute_moments(self):
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
                    index = self.get_index(r, g, b)
                    line += self.weights[index]
                    line_r += self.momentsR[index]
                    line_g += self.momentsG[index]
                    line_b += self.momentsB[index]
                    line2 += self.moments[index]

                    area[b] += line
                    area_r[b] += line_r
                    area_g[b] += line_g
                    area_b[b] += line_b
                    area2[b] += line2

                    previous_index = self.get_index(r - 1, g, b)
                    self.weights[index] = self.weights[previous_index] + area[b]
                    self.momentsR[index] = self.momentsR[previous_index] + area_r[b]
                    self.momentsG[index] = self.momentsG[previous_index] + area_g[b]
                    self.momentsB[index] = self.momentsB[previous_index] + area_b[b]
                    self.moments[index] = self.moments[previous_index] + area2[b]

    def create_boxes(self, max_color_count):
        self.cubes = [Box() for _ in range(max_color_count)]
        self.cubes[0] = Box(r0=0, r1=32, g0=0, g1=32, b0=0, b1=32, vol=0)

        volume_variance = [0.0] * max_color_count
        next = 0
        generated_color_count = max_color_count
        for i in range(1, max_color_count):
            if self.cut(self.cubes[next], self.cubes[i]):
                volume_variance[next] = (
                    self.variance(self.cubes[next]) if self.cubes[next].vol > 1 else 0.0
                )
                volume_variance[i] = (
                    self.variance(self.cubes[i]) if self.cubes[i].vol > 1 else 0.0
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

    def create_result(self, color_count):
        colors = []
        for i in range(color_count):
            cube = self.cubes[i]
            weight = self.volume(cube, self.weights)
            if weight > 0:
                r = np.around(self.volume(cube, self.momentsR) / weight).astype(
                    np.uint8
                )
                g = np.around(self.volume(cube, self.momentsG) / weight).astype(
                    np.uint8
                )
                b = np.around(self.volume(cube, self.momentsB) / weight).astype(
                    np.uint8
                )
                color = color_util.argb_from_rgb(r, g, b)
                colors.append(color)
        return colors

    def variance(self, cube):
        dr = self.volume(cube, self.momentsR)
        dg = self.volume(cube, self.momentsG)
        db = self.volume(cube, self.momentsB)
        xx = (
            self.moments[self.get_index(cube.r1, cube.g1, cube.b1)]
            - self.moments[self.get_index(cube.r1, cube.g1, cube.b0)]
            - self.moments[self.get_index(cube.r1, cube.g0, cube.b1)]
            + self.moments[self.get_index(cube.r1, cube.g0, cube.b0)]
            - self.moments[self.get_index(cube.r0, cube.g1, cube.b1)]
            + self.moments[self.get_index(cube.r0, cube.g1, cube.b0)]
            + self.moments[self.get_index(cube.r0, cube.g0, cube.b1)]
            - self.moments[self.get_index(cube.r0, cube.g0, cube.b0)]
        )

        hypotenuse = dr * dr + dg * dg + db * db
        volume_ = self.volume(cube, self.weights)
        return xx - hypotenuse / volume_

    def cut(self, one, two):
        whole_r = self.volume(one, self.momentsR)
        whole_g = self.volume(one, self.momentsG)
        whole_b = self.volume(one, self.momentsB)
        whole_w = self.volume(one, self.weights)

        max_r_result = self.maximize(
            one, Direction.red, one.r0 + 1, one.r1, whole_r, whole_g, whole_b, whole_w
        )
        max_g_result = self.maximize(
            one, Direction.green, one.g0 + 1, one.g1, whole_r, whole_g, whole_b, whole_w
        )
        max_b_result = self.maximize(
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

    def maximize(
        self, cube, direction, first, last, whole_r, whole_g, whole_b, whole_w
    ):
        bottom_r = self.bottom(cube, direction, self.momentsR)
        bottom_g = self.bottom(cube, direction, self.momentsG)
        bottom_b = self.bottom(cube, direction, self.momentsB)
        bottom_w = self.bottom(cube, direction, self.weights)

        max = 0.0
        cut = -1

        for i in range(first, last):
            half_r = bottom_r + self.top(cube, direction, i, self.momentsR)
            half_g = bottom_g + self.top(cube, direction, i, self.momentsG)
            half_b = bottom_b + self.top(cube, direction, i, self.momentsB)
            half_w = bottom_w + self.top(cube, direction, i, self.weights)

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

    def volume(self, cube, moment):
        return (
            moment[self.get_index(cube.r1, cube.g1, cube.b1)]
            - moment[self.get_index(cube.r1, cube.g1, cube.b0)]
            - moment[self.get_index(cube.r1, cube.g0, cube.b1)]
            + moment[self.get_index(cube.r1, cube.g0, cube.b0)]
            - moment[self.get_index(cube.r0, cube.g1, cube.b1)]
            + moment[self.get_index(cube.r0, cube.g1, cube.b0)]
            + moment[self.get_index(cube.r0, cube.g0, cube.b1)]
            - moment[self.get_index(cube.r0, cube.g0, cube.b0)]
        )

    def bottom(self, cube, direction, moment):
        if direction == Direction.red:
            return (
                -moment[self.get_index(cube.r0, cube.g1, cube.b1)]
                + moment[self.get_index(cube.r0, cube.g1, cube.b0)]
                + moment[self.get_index(cube.r0, cube.g0, cube.b1)]
                - moment[self.get_index(cube.r0, cube.g0, cube.b0)]
            )
        elif direction == Direction.green:
            return (
                -moment[self.get_index(cube.r1, cube.g0, cube.b1)]
                + moment[self.get_index(cube.r1, cube.g0, cube.b0)]
                + moment[self.get_index(cube.r0, cube.g0, cube.b1)]
                - moment[self.get_index(cube.r0, cube.g0, cube.b0)]
            )
        elif direction == Direction.blue:
            return (
                -moment[self.get_index(cube.r1, cube.g1, cube.b0)]
                + moment[self.get_index(cube.r1, cube.g0, cube.b0)]
                + moment[self.get_index(cube.r0, cube.g1, cube.b0)]
                - moment[self.get_index(cube.r0, cube.g0, cube.b0)]
            )
        else:
            raise Exception("unexpected direction {}".format(direction))

    def top(self, cube, direction, position, moment):
        if direction == Direction.red:
            return (
                moment[self.get_index(position, cube.g1, cube.b1)]
                - moment[self.get_index(position, cube.g1, cube.b0)]
                - moment[self.get_index(position, cube.g0, cube.b1)]
                + moment[self.get_index(position, cube.g0, cube.b0)]
            )
        elif direction == Direction.green:
            return (
                moment[self.get_index(cube.r1, position, cube.b1)]
                - moment[self.get_index(cube.r1, position, cube.b0)]
                - moment[self.get_index(cube.r0, position, cube.b1)]
                + moment[self.get_index(cube.r0, position, cube.b0)]
            )
        elif direction == Direction.blue:
            return (
                moment[self.get_index(cube.r1, cube.g1, position)]
                - moment[self.get_index(cube.r1, cube.g0, position)]
                - moment[self.get_index(cube.r0, cube.g1, position)]
                + moment[self.get_index(cube.r0, cube.g0, position)]
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
