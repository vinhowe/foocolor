from typing import NamedTuple

from ..palettes import CorePalette


class Scheme(NamedTuple):
    primary: int
    on_primary: int
    primary_container: int
    on_primary_container: int
    secondary: int
    on_secondary: int
    secondary_container: int
    on_secondary_container: int
    tertiary: int
    on_tertiary: int
    tertiary_container: int
    on_tertiary_container: int
    error: int
    on_error: int
    error_container: int
    on_error_container: int
    background: int
    on_background: int
    surface: int
    on_surface: int
    surface_variant: int
    on_surface_variant: int
    outline: int
    shadow: int
    inverse_surface: int
    inverse_on_surface: int
    inverse_primary: int

    @classmethod
    def light(cls, color: int) -> "Scheme":
        return cls.light_from_core_palette(CorePalette.of(color))

    @classmethod
    def dark(cls, color: int) -> "Scheme":
        return cls.dark_from_core_palette(CorePalette.of(color))

    @classmethod
    def light_content(cls, color: int) -> "Scheme":
        return cls.light_from_core_palette(CorePalette.content_of(color))

    @classmethod
    def dark_content(cls, color: int) -> "Scheme":
        return cls.dark_from_core_palette(CorePalette.content_of(color))

    @classmethod
    def light_from_core_palette(cls, palette: CorePalette) -> "Scheme":
        return cls(
            primary=palette.primary.get(40),
            on_primary=palette.primary.get(100),
            primary_container=palette.primary.get(90),
            on_primary_container=palette.primary.get(10),
            secondary=palette.secondary.get(40),
            on_secondary=palette.secondary.get(100),
            secondary_container=palette.secondary.get(90),
            on_secondary_container=palette.secondary.get(10),
            tertiary=palette.tertiary.get(40),
            on_tertiary=palette.tertiary.get(100),
            tertiary_container=palette.tertiary.get(90),
            on_tertiary_container=palette.tertiary.get(10),
            error=palette.error.get(40),
            on_error=palette.error.get(100),
            error_container=palette.error.get(90),
            on_error_container=palette.error.get(10),
            background=palette.neutral.get(99),
            on_background=palette.neutral.get(10),
            surface=palette.neutral.get(99),
            on_surface=palette.neutral.get(10),
            surface_variant=palette.neutral_variant.get(90),
            on_surface_variant=palette.neutral_variant.get(30),
            outline=palette.neutral_variant.get(50),
            shadow=palette.neutral.get(0),
            inverse_surface=palette.neutral.get(20),
            inverse_on_surface=palette.neutral.get(95),
            inverse_primary=palette.primary.get(80),
        )

    @classmethod
    def dark_from_core_palette(cls, palette: CorePalette) -> "Scheme":
        return cls(
            primary=palette.primary.get(80),
            on_primary=palette.primary.get(20),
            primary_container=palette.primary.get(30),
            on_primary_container=palette.primary.get(90),
            secondary=palette.secondary.get(80),
            on_secondary=palette.secondary.get(20),
            secondary_container=palette.secondary.get(30),
            on_secondary_container=palette.secondary.get(90),
            tertiary=palette.tertiary.get(80),
            on_tertiary=palette.tertiary.get(20),
            tertiary_container=palette.tertiary.get(30),
            on_tertiary_container=palette.tertiary.get(90),
            error=palette.error.get(80),
            on_error=palette.error.get(20),
            error_container=palette.error.get(30),
            on_error_container=palette.error.get(80),
            background=palette.neutral.get(10),
            on_background=palette.neutral.get(90),
            surface=palette.neutral.get(10),
            on_surface=palette.neutral.get(90),
            surface_variant=palette.neutral_variant.get(30),
            on_surface_variant=palette.neutral_variant.get(80),
            outline=palette.neutral_variant.get(60),
            shadow=palette.neutral.get(0),
            inverse_surface=palette.neutral.get(90),
            inverse_on_surface=palette.neutral.get(20),
            inverse_primary=palette.primary.get(40),
        )
