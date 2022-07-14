from typing import List, Optional

from ..hct import Hct

# Commonly-used tone values.
common_tones = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
common_size = len(common_tones)


class TonalPalette:
    """
    A convenience class for retrieving colors that are constant in hue and
    chroma, but vary in tone.

    This class can be instantiated in two ways:
    1. [of] From hue and chroma. (preferred)
    2. [from_list] From a fixed-size ([TonalPalette.common_size]) list of ints
    representing ARBG colors. Correctness (constant hue and chroma) of the input
    is not enforced. [get] will only return the input colors, corresponding to
    [common_tones].
    """

    def __init__(
        self,
        hue: Optional[float] = None,
        chroma: Optional[float] = None,
        cache: Optional[List[int]] = None,
    ):
        self._hue = hue
        self._chroma = chroma
        self._cache = cache or {}

    @classmethod
    def of(cls, hue: float, chroma: float) -> "TonalPalette":
        """
        Create colors using [hue] and [chroma].
        """
        return cls(hue=hue, chroma=chroma)

    @classmethod
    def from_list(cls, colors: List[int]) -> "TonalPalette":
        """
        Create colors from a fixed-size list of ARGB color ints.

        Inverse of [TonalPalette.as_list].
        """
        assert len(colors) == common_size
        cache = {tone: color for tone, color in zip(common_tones, colors)}
        return cls(cache=cache)

    @property
    def as_list(self) -> List[int]:
        """
        Returns a fixed-size list of ARGB color ints for common tone values.

        Inverse of [from_list].
        """
        return [self.get(tone) for tone in common_tones]

    def get(self, tone: int) -> int:
        """
        Returns the ARGB representation of an HCT color.

        If the class was instantiated from [_hue] and [_chroma], will return the
        color with corresponding [tone].
        If the class was instantiated from a fixed-size list of color ints, [tone]
        must be in [common_tones].
        """
        if self._hue is None or self._chroma is None:
            if tone not in self._cache:
                raise ValueError(
                    f"When a TonalPalette is created with from_list, tone must be one of "
                    f"{common_tones}"
                )
            return self._cache[tone]
        chroma = min(self._chroma, 40.0) if tone >= 90.0 else self._chroma
        return self._cache.setdefault(tone, Hct.from_(self._hue, chroma, tone).argb)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TonalPalette):
            if self._hue is not None and self._chroma is not None:
                return self._hue == other._hue and self._chroma == other._chroma
            else:
                return set(self._cache.values()).issubset(set(other._cache.values()))
        return False

    def __hash__(self) -> int:
        return hash((self._hue, self._chroma)) ^ hash(tuple(self._cache.values()))

    def __repr__(self) -> str:
        if self._hue is not None and self._chroma is not None:
            return f"TonalPalette.of({self._hue}, {self._chroma})"
        else:
            return f"TonalPalette.from_list({self._cache})"
