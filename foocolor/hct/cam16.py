import numpy as np

from ..util.color_util import argb_from_xyz, xyz_from_argb
from .viewing_conditions import VIEWING_CONDITIONS_SRGB, ViewingConditions


class Cam16:
    def __init__(self, hue, chroma, j, q, m, s, jstar, astar, bstar):
        self.hue = hue
        self.chroma = chroma
        self.j = j
        self.q = q
        self.m = m
        self.s = s
        self.jstar = jstar
        self.astar = astar
        self.bstar = bstar

    @classmethod
    def from_int(cls, argb):
        return cls.from_int_in_viewing_conditions(argb, VIEWING_CONDITIONS_SRGB)

    @classmethod
    def from_int_in_viewing_conditions(
        cls, argb, viewing_conditions: ViewingConditions
    ):
        x, y, z = xyz_from_argb(argb)

        r_c = 0.401288 * x + 0.650173 * y - 0.051461 * z
        g_c = -0.250268 * x + 1.204414 * y + 0.045854 * z
        b_c = -0.002079 * x + 0.048952 * y + 0.953127 * z

        r_d = viewing_conditions.rgb_d[0] * r_c
        g_d = viewing_conditions.rgb_d[1] * g_c
        b_d = viewing_conditions.rgb_d[2] * b_c

        r_af = (viewing_conditions.fl * abs(r_d) / 100.0) ** 0.42
        g_af = (viewing_conditions.fl * abs(g_d) / 100.0) ** 0.42
        b_af = (viewing_conditions.fl * abs(b_d) / 100.0) ** 0.42
        r_a = np.sign(r_d) * 400.0 * r_af / (r_af + 27.13)
        g_a = np.sign(g_d) * 400.0 * g_af / (g_af + 27.13)
        b_a = np.sign(b_d) * 400.0 * b_af / (b_af + 27.13)

        a = (11.0 * r_a - 12.0 * g_a + b_a) / 11.0
        b = (r_a + g_a - 2.0 * b_a) / 9.0

        u = (20.0 * r_a + 20.0 * g_a + 21.0 * b_a) / 20.0
        p2 = (40.0 * r_a + 20.0 * g_a + b_a) / 20.0

        atan2 = np.arctan2(b, a)
        atan_degrees = atan2 * 180.0 / np.pi
        hue = atan_degrees if atan_degrees >= 0 else atan_degrees + 360
        hue = hue if hue < 360 else hue - 360
        hue_radians = hue * np.pi / 180.0

        ac = p2 * viewing_conditions.nbb

        j = 100.0 * (ac / viewing_conditions.aw) ** (
            viewing_conditions.c * viewing_conditions.z
        )
        q = (
            (4.0 / viewing_conditions.c)
            * np.sqrt(j / 100.0)
            * (viewing_conditions.aw + 4.0)
            * (viewing_conditions.f_l_root)
        )

        hue_prime = hue + 360 if hue < 20.14 else hue
        e_hue = (1.0 / 4.0) * (np.cos(hue_prime * np.pi / 180.0 + 2.0) + 3.8)
        p1 = 50000.0 / 13.0 * e_hue * viewing_conditions.nc * viewing_conditions.ncb
        t = p1 * np.sqrt(a * a + b * b) / (u + 0.305)
        alpha = (t**0.9) * (
            (1.64 - (0.29**viewing_conditions.background_y_to_white_point_y)) ** 0.73
        )
        c = alpha * np.sqrt(j / 100.0)
        m = c * viewing_conditions.f_l_root
        s = 50.0 * np.sqrt(
            (alpha * viewing_conditions.c) / (viewing_conditions.aw + 4.0)
        )

        jstar = (1.0 + 100.0 * 0.007) * j / (1.0 + 0.007 * j)
        mstar = np.log(1.0 + 0.0228 * m) / 0.0228
        astar = mstar * np.cos(hue_radians)
        bstar = mstar * np.sin(hue_radians)
        return cls(hue, c, j, q, m, s, jstar, astar, bstar)

    @classmethod
    def from_jch(cls, j, c, h):
        return cls.from_jch_in_viewing_conditions(j, c, h, ViewingConditions.sRGB)

    @classmethod
    def from_jch_in_viewing_conditions(cls, j, c, h, viewing_conditions):
        q = (
            (4.0 / viewing_conditions.c)
            * np.sqrt(j / 100.0)
            * (viewing_conditions.aw + 4.0)
            * (viewing_conditions.f_l_root)
        )
        m = c * viewing_conditions.f_l_root
        alpha = c / np.sqrt(j / 100.0)
        s = 50.0 * np.sqrt(
            (alpha * viewing_conditions.c) / (viewing_conditions.aw + 4.0)
        )

        hue_radians = h * np.pi / 180.0
        jstar = (1.0 + 100.0 * 0.007) * j / (1.0 + 0.007 * j)
        mstar = 1.0 / 0.0228 * np.log(1.0 + 0.0228 * m)
        astar = mstar * np.cos(hue_radians)
        bstar = mstar * np.sin(hue_radians)
        return cls(h, c, j, q, m, s, jstar, astar, bstar)

    @classmethod
    def from_ucs(cls, jstar, astar, bstar):
        return cls.from_ucs_in_viewing_conditions(
            jstar, astar, bstar, ViewingConditions.standard
        )

    @classmethod
    def from_ucs_in_viewing_conditions(cls, jstar, astar, bstar, viewing_conditions):
        a = astar
        b = bstar
        m = np.sqrt(a * a + b * b)
        m = (np.exp(m * 0.0228) - 1.0) / 0.0228
        c = m / viewing_conditions.f_l_root
        h = np.atan2(b, a) * (180.0 / np.pi)
        h = h if h >= 0 else h + 360
        j = jstar / (1 - (jstar - 100) * 0.007)

        return cls.from_jch_in_viewing_conditions(j, c, h, viewing_conditions)

    def distance(self, other):
        d_j = self.jstar - other.jstar
        d_a = self.astar - other.astar
        d_b = self.bstar - other.bstar
        d_e_prime = np.sqrt(d_j * d_j + d_a * d_a + d_b * d_b)
        d_e = 1.41 * np.pow(d_e_prime, 0.63)
        return d_e

    def to_int(self):
        return self.viewed(ViewingConditions.sRGB)

    def viewed(self, viewing_conditions):
        alpha = (
            0.0
            if self.chroma == 0.0 or self.j == 0.0
            else self.chroma / np.sqrt(self.j / 100.0)
        )

        t = np.pow(
            alpha
            / np.pow(
                1.64 - np.pow(0.29, viewing_conditions.background_y_to_white_point_y),
                0.73,
            ),
            1.0 / 0.9,
        )
        h_rad = self.hue * np.pi / 180.0

        e_hue = 0.25 * (np.cos(h_rad + 2.0) + 3.8)
        ac = viewing_conditions.aw * np.pow(
            self.j / 100.0, 1.0 / viewing_conditions.c / viewing_conditions.z
        )
        p1 = e_hue * (50000.0 / 13.0) * viewing_conditions.nc * viewing_conditions.ncb

        p2 = ac / viewing_conditions.nbb

        h_sin = np.sin(h_rad)
        h_cos = np.cos(h_rad)

        gamma = (
            23.0 * (p2 + 0.305) * t / (23.0 * p1 + 11 * t * h_cos + 108.0 * t * h_sin)
        )
        a = gamma * h_cos
        b = gamma * h_sin
        r_a = (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0
        g_a = (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0
        b_a = (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0

        r_c_base = max(0, (27.13 * abs(r_a)) / (400.0 - abs(r_a)))
        r_c = (
            np.sign(r_a)
            * (100.0 / viewing_conditions.fl)
            * np.pow(r_c_base, 1.0 / 0.42)
        )
        g_c_base = max(0, (27.13 * abs(g_a)) / (400.0 - abs(g_a)))
        g_c = (
            np.sign(g_a)
            * (100.0 / viewing_conditions.fl)
            * np.pow(g_c_base, 1.0 / 0.42)
        )
        b_c_base = max(0, (27.13 * abs(b_a)) / (400.0 - abs(b_a)))
        b_c = (
            np.sign(b_a)
            * (100.0 / viewing_conditions.fl)
            * np.pow(b_c_base, 1.0 / 0.42)
        )
        r_f = r_c / viewing_conditions.rgb_d[0]
        g_f = g_c / viewing_conditions.rgb_d[1]
        b_f = b_c / viewing_conditions.rgb_d[2]

        x = 1.86206786 * r_f - 1.01125463 * g_f + 0.14918677 * b_f
        y = 0.38752654 * r_f + 0.62144744 * g_f - 0.00897398 * b_f
        z = -0.01584150 * r_f - 0.03412294 * g_f + 1.04996444 * b_f

        argb = argb_from_xyz(x, y, z)
        return argb
