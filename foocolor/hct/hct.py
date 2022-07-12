from foocolor.hct.cam16 import Cam16
from foocolor.hct.hct_solver import solve_to_int
from foocolor.util.color_util import lstar_from_argb


class Hct:
    def __init__(self, argb: int):
        self._argb = argb
        cam16 = Cam16.from_int(argb)
        self._hue = cam16.hue
        self._chroma = cam16.chroma
        self._tone = lstar_from_argb(argb)

    @classmethod
    def from_(cls, hue, chroma, tone):
        argb = solve_to_int(hue, chroma, tone)
        return Hct(argb)

    @property
    def hue(self):
        return self._hue

    @property
    def chroma(self):
        return self._chroma

    @property
    def tone(self):
        return self._tone

    @property
    def argb(self):
        return self._argb

    def __eq__(self, other):
        return self._argb == other._argb

    def __repr__(self):
        return f"Hct({self._hue}, {self._chroma}, {self._tone}, {self._argb})"
