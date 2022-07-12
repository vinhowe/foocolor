import numpy as np

from ..util.color_util import WHITE_POINT_D65, y_from_lstar


class ViewingConditions:
    def __init__(
        self,
        white_point,
        adapting_luminance,
        background_lstar,
        surround,
        discounting_illuminant,
        background_y_to_white_point_y,
        aw,
        nbb,
        ncb,
        c,
        nc,
        drgb_inverse,
        rgb_d,
        fl,
        f_l_root,
        z,
    ):
        self.white_point = white_point
        self.adapting_luminance = adapting_luminance
        self.background_lstar = background_lstar
        self.surround = surround
        self.discounting_illuminant = discounting_illuminant
        self.background_y_to_white_point_y = background_y_to_white_point_y
        self.aw = aw
        self.nbb = nbb
        self.ncb = ncb
        self.c = c
        self.nc = nc
        self.drgb_inverse = drgb_inverse
        self.rgb_d = rgb_d
        self.fl = fl
        self.f_l_root = f_l_root
        self.z = z

    @classmethod
    def make(
        cls,
        white_point=None,
        adapting_luminance=-1.0,
        background_lstar=50.0,
        surround=2.0,
        discounting_illuminant=False,
    ):
        if white_point is None:
            white_point = WHITE_POINT_D65

        if adapting_luminance <= 0.0:
            adapting_luminance = 200.0 / np.pi * y_from_lstar(50.0) / 100.0
        background_lstar = max(30.0, background_lstar)
        # Transform test illuminant white in XYZ to 'cone'/'rgb' responses
        xyz = white_point
        r_w = xyz[0] * 0.401288 + xyz[1] * 0.650173 + xyz[2] * -0.051461
        g_w = xyz[0] * -0.250268 + xyz[1] * 1.204414 + xyz[2] * 0.045854
        b_w = xyz[0] * -0.002079 + xyz[1] * 0.048952 + xyz[2] * 0.953127

        # Scale input surround, domain (0, 2), to CAM16 surround, domain (0.8, 1.0)
        assert 0.0 <= surround <= 2.0
        f = 0.8 + (surround / 10.0)
        # "Exponential non-linearity"
        c = (
            np.interp(((f - 0.9) * 10.0), [0, 1], [0.59, 0.69])
            if f >= 0.9
            else np.interp(((f - 0.8) * 10.0), [0, 1], [0.525, 0.59])
        )

        # Calculate degree of adaptation to illuminant
        d = (
            1.0
            if discounting_illuminant
            else f * (1.0 - ((1.0 / 3.6) * np.exp((-adapting_luminance - 42.0) / 92.0)))
        )
        # Per Li et al, if D is greater than 1 or less than 0, set it to 1 or 0.
        d = 1.0 if d > 1.0 else 0.0 if d < 0.0 else d
        # chromatic induction factor
        nc = f

        # Cone responses to the whitePoint, r/g/b/W, adjusted for discounting.
        #
        # Why use 100.0 instead of the white point's relative luminance?
        #
        # Some papers and implementations, for both CAM02 and CAM16, use the Y
        # value of the reference white instead of 100. Fairchild's Color Appearance
        # Models (3rd edition) notes that this is in error: it was included in the
        # CIE 2004a report on CIECAM02, but, later parts of the conversion process
        # account for scaling of appearance relative to the white point relative
        # luminance. This part should simply use 100 as luminance.
        rgb_d = [
            d * (100.0 / r_w) + 1.0 - d,
            d * (100.0 / g_w) + 1.0 - d,
            d * (100.0 / b_w) + 1.0 - d,
        ]

        # Factor used in calculating meaningful factors
        k = 1.0 / (5.0 * adapting_luminance + 1.0)
        k4 = k * k * k * k
        k4_f = 1.0 - k4

        # Luminance-level adaptation factor
        fl = (k4 * adapting_luminance) + (
            0.1 * k4_f * k4_f * (5.0 * adapting_luminance) ** (1.0 / 3.0)
        )
        # Intermediate factor, ratio of background relative luminance to white relative luminance
        n = y_from_lstar(background_lstar) / white_point[1]

        # Base exponential nonlinearity
        # note Schlomer 2018 has a typo and uses 1.58, the correct factor is 1.48
        z = 1.48 + (n ** 0.5)

        # Luminance-level induction factors
        nbb = 0.725 / (n ** 0.2)
        ncb = nbb

        # Discounted cone responses to the white point, adjusted for post-saturationtic
        # adaptation perceptual nonlinearities.
        rgb_a_factors = [
            (fl * rgb_d[0] * r_w / 100.0) ** 0.42,
            (fl * rgb_d[1] * g_w / 100.0) ** 0.42,
            (fl * rgb_d[2] * b_w / 100.0) ** 0.42,
        ]

        rgb_a = [
            (400.0 * rgb_a_factors[0]) / (rgb_a_factors[0] + 27.13),
            (400.0 * rgb_a_factors[1]) / (rgb_a_factors[1] + 27.13),
            (400.0 * rgb_a_factors[2]) / (rgb_a_factors[2] + 27.13),
        ]

        aw = (40.0 * rgb_a[0] + 20.0 * rgb_a[1] + rgb_a[2]) / 20.0 * nbb

        return cls(
            white_point=white_point,
            adapting_luminance=adapting_luminance,
            background_lstar=background_lstar,
            surround=surround,
            discounting_illuminant=discounting_illuminant,
            background_y_to_white_point_y=n,
            aw=aw,
            nbb=nbb,
            ncb=ncb,
            c=c,
            nc=nc,
            drgb_inverse=[0.0, 0.0, 0.0],
            rgb_d=rgb_d,
            fl=fl,
            f_l_root=fl ** 0.25,
            z=z,
        )


STANDARD = SRGB = ViewingConditions.make()
