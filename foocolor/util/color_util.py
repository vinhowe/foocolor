import numpy as np

SRGB_TO_XYZ = np.array(
    [
        [0.41233895, 0.35762064, 0.18051042],
        [0.2126, 0.7152, 0.0722],
        [0.01932141, 0.11916382, 0.95034478],
    ]
)
SRGB_TO_XYZ.flags.writeable = False

XYZ_TO_SRGB = np.array(
    [
        [
            3.2413774792388685,
            -1.5376652402851851,
            -0.49885366846268053,
        ],
        [
            -0.9691452513005321,
            1.8758853451067872,
            0.04156585616912061,
        ],
        [
            0.05562093689691305,
            -0.20395524564742123,
            1.0571799111220335,
        ],
    ]
)
XYZ_TO_SRGB.flags.writeable = False

WHITE_POINT_D65 = np.array([95.047, 100.0, 108.883])
WHITE_POINT_D65.flags.writeable = False


def argb_from_rgb(red, green, blue):
    return 255 << 24 | (red & 255) << 16 | (green & 255) << 8 | blue & 255


def argb_from_linrgb(linrgb):
    return argb_from_rgb(*delinearized(linrgb).astype(np.uint8))


def alpha_from_argb(argb):
    return (argb >> 24) & 255


def red_from_argb(argb):
    return (argb >> 16) & 255


def green_from_argb(argb):
    return (argb >> 8) & 255


def blue_from_argb(argb):
    return argb & 255


def rgb_from_argb(argb):
    return np.array([red_from_argb(argb), green_from_argb(argb), blue_from_argb(argb)])


def is_opaque(argb):
    return alpha_from_argb(argb) >= 255


def argb_from_xyz(x, y, z):
    matrix = XYZ_TO_SRGB
    linear_rgb = np.dot(matrix, [x, y, z])
    rgb = delinearized(linear_rgb).astype(np.uint8)
    return argb_from_rgb(rgb[0], rgb[1], rgb[2])


def xyz_from_argb(argb):
    return linearized(rgb_from_argb(argb)) @ SRGB_TO_XYZ.T


def argb_from_lab(l, a, b):
    white_point = WHITE_POINT_D65
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = _lab_invf(np.array([fx, fy, fz])) * white_point
    return argb_from_xyz(*xyz)


def lab_from_argb(argb):
    rgb = linearized(rgb_from_argb(argb))
    matrix = SRGB_TO_XYZ
    xyz = np.dot(matrix, rgb).T
    white_point = WHITE_POINT_D65
    xyz_normalized = xyz / white_point
    fx, fy, fz = _lab_f(xyz_normalized).T
    return np.array([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]).T


def argb_from_lstar(lstar):
    y = y_from_lstar(lstar)
    component = delinearized(y).astype(np.uint8)
    return argb_from_rgb(component, component, component)


def lstar_from_argb(argb):
    y = xyz_from_argb(argb)[1]
    return 116.0 * _lab_f(y / 100.0) - 16.0


def y_from_lstar(lstar):
    return 100.0 * _lab_invf((lstar + 16.0) / 116.0)


def linearized(rgb_component):
    normalized = rgb_component / 255.0
    normalized = np.where(
        normalized <= 0.040449936,
        normalized / 12.92 * 100.0,
        np.power((normalized + 0.055) / 1.055, 2.4) * 100.0,
    )
    return normalized


def delinearized(rgb):
    normalized = rgb / 100.0
    delinearized = np.where(
        normalized <= 0.0031308,
        normalized * 12.92,
        1.055 * np.power(normalized, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(delinearized * 255.0), 0, 255)


def _lab_f(t):
    e = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return np.where(t > e, pow(t, 1.0 / 3.0), (kappa * t + 16) / 116)


def _lab_invf(ft):
    e = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    ft3 = np.power(ft, 3)
    return np.where(ft3 > e, ft3, (116 * ft - 16) / kappa)
