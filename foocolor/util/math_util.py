import numpy as np


def sanitize_degrees(degrees):
    degrees = degrees % 360.0
    if degrees < 0:
        degrees = degrees + 360.0
    return degrees


def difference_degrees(a, b):
    return 180.0 - np.abs(np.abs(a - b) - 180.0)
