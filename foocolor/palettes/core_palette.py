from typing import List

from ..hct import Cam16
from .tonal_palette import TonalPalette


class CorePalette:
    """
    An intermediate concept between the key color for a UI theme, and a full
    color scheme. 5 tonal palettes are generated, all except one use the same
    hue as the key color, and all vary in chroma.
    """

    # The number of generated tonal palettes.
    size = 5

    def __init__(
        self,
        primary: TonalPalette,
        secondary: TonalPalette,
        tertiary: TonalPalette,
        neutral: TonalPalette,
        neutral_variant: TonalPalette,
    ):
        self.primary = primary
        self.secondary = secondary
        self.tertiary = tertiary
        self.neutral = neutral
        self.neutral_variant = neutral_variant
        self.error = TonalPalette.of(25, 84)

    @classmethod
    def of(cls, argb: int) -> "CorePalette":
        """
        Create a [CorePalette] from a source ARGB color.
        """
        cam = Cam16.from_int(argb)
        return cls._(cam.hue, cam.chroma)

    @classmethod
    def _(cls, hue: float, chroma: float) -> "CorePalette":
        return cls(
            primary=TonalPalette.of(hue, max(48, chroma)),
            secondary=TonalPalette.of(hue, 16),
            tertiary=TonalPalette.of(hue + 60, 24),
            neutral=TonalPalette.of(hue, 4),
            neutral_variant=TonalPalette.of(hue, 8),
        )

    @classmethod
    def content_of(cls, argb: int) -> "CorePalette":
        """
        Create a content [CorePalette] from a source ARGB color.
        """
        cam = Cam16.from_int(argb)
        return cls._content_of(cam.hue, cam.chroma)

    @classmethod
    def _content_of(cls, hue: float, chroma: float) -> "CorePalette":
        return cls(
            primary=TonalPalette.of(hue, chroma),
            secondary=TonalPalette.of(hue, chroma / 3),
            tertiary=TonalPalette.of(hue + 60, chroma / 2),
            neutral=TonalPalette.of(hue, min(chroma / 12, 4)),
            neutral_variant=TonalPalette.of(hue, min(chroma / 6, 8)),
        )

    @classmethod
    def from_list(cls, colors: List[int]) -> "CorePalette":
        """
        Create a [CorePalette] from a fixed-size list of ARGB color ints
        representing concatenated tonal palettes.

        Inverse of [as_list].
        """
        assert len(colors) == cls.size * TonalPalette.common_size
        return cls(
            primary=TonalPalette.from_list(
                _get_partition(colors, 0, TonalPalette.common_size)
            ),
            secondary=TonalPalette.from_list(
                _get_partition(colors, 1, TonalPalette.common_size)
            ),
            tertiary=TonalPalette.from_list(
                _get_partition(colors, 2, TonalPalette.common_size)
            ),
            neutral=TonalPalette.from_list(
                _get_partition(colors, 3, TonalPalette.common_size)
            ),
            neutral_variant=TonalPalette.from_list(
                _get_partition(colors, 4, TonalPalette.common_size)
            ),
        )

    def as_list(self) -> List[int]:
        """
        Returns a list of ARGB color [int]s from concatenated tonal palettes.

        Inverse of [CorePalette.from_list].
        """
        return [
            *self.primary.as_list,
            *self.secondary.as_list,
            *self.tertiary.as_list,
            *self.neutral.as_list,
            *self.neutral_variant.as_list,
        ]

    def __eq__(self, other: "CorePalette") -> bool:
        return (
            self.primary == other.primary
            and self.secondary == other.secondary
            and self.tertiary == other.tertiary
            and self.neutral == other.neutral
            and self.neutral_variant == other.neutral_variant
            and self.error == other.error
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.primary,
                self.secondary,
                self.tertiary,
                self.neutral,
                self.neutral_variant,
                self.error,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"primary={self.primary}, "
            f"secondary={self.secondary}, "
            f"tertiary={self.tertiary}, "
            f"neutral={self.neutral}, "
            f"neutral_variant={self.neutral_variant}, "
            f"error={self.error}, "
            f")"
        )


# Returns a partition from a list.
#
# For example, given a list with 2 partitions of size 3.
# range = [1, 2, 3, 4, 5, 6];
#
# range.get_partition(0, 3) # [1, 2, 3]
# range.get_partition(1, 3) # [4, 5, 6]
def _get_partition(
    list_: List[int], partition_number: int, partition_size: int
) -> List[int]:
    return list_[
        partition_number * partition_size : (partition_number + 1) * partition_size
    ]
