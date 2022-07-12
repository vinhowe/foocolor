from typing import Dict, List

import numpy as np

from foocolor.hct.hct import Hct
from foocolor.util.math_util import difference_degrees, sanitize_degrees


class ArgbAndScore(object):
    def __init__(self, argb: int, score: float):
        self.argb = argb
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


_TARGET_CHROMA = 48.0
_WEIGHT_PROPORTION = 0.7
_WEIGHT_CHROMA_ABOVE = 0.3
_WEIGHT_CHROMA_BELOW = 0.1
_CUTOFF_CHROMA = 5.0
_CUTOFF_EXCITED_PROPORTION = 0.01


def score(
    colors_to_population: Dict[int, int], desired: int = 4, filter: bool = True
) -> List[int]:
    population_sum = 0.0
    for population in colors_to_population.values():
        population_sum += population

    # Turn the count of each color into a proportion by dividing by the total
    # count. Also, fill a cache of CAM16 colors representing each color, and
    # record the proportion of colors for each CAM16 hue.
    argb_to_raw_proportion = {}
    argb_to_hct = {}
    hue_proportions = np.zeros(360)
    for color in colors_to_population.keys():
        population = colors_to_population[color]
        proportion = population / population_sum
        argb_to_raw_proportion[color] = proportion

        hct = Hct(color)
        argb_to_hct[color] = hct

        hue = np.floor(hct.hue).astype(int)
        hue_proportions[hue] += proportion

    # Determine the proportion of the colors around each color, by summing the
    # proportions around each color's hue.
    argb_to_hue_proportion = {}
    for color, cam in argb_to_hct.items():
        hue = np.round(cam.hue).astype(int)

        excited_proportion = 0.0
        for i in range(hue - 15, hue + 15):
            neighbor_hue = int(sanitize_degrees(i))
            excited_proportion += hue_proportions[neighbor_hue]
        argb_to_hue_proportion[color] = excited_proportion

    # Remove colors that are unsuitable, ex. very dark or unchromatic colors.
    # Also, remove colors that are very similar in hue.
    filtered_colors = (
        _filter(argb_to_hue_proportion, argb_to_hct)
        if filter
        else list(argb_to_hue_proportion.keys())
    )

    # Score the colors by their proportion, as well as how chromatic they are.
    argb_to_score = {}
    for color in filtered_colors:
        cam = argb_to_hct[color]
        proportion = argb_to_hue_proportion[color]

        proportion_score = proportion * 100.0 * _WEIGHT_PROPORTION

        chroma_weight = (
            _WEIGHT_CHROMA_BELOW
            if cam.chroma < _TARGET_CHROMA
            else _WEIGHT_CHROMA_ABOVE
        )
        chroma_score = (cam.chroma - _TARGET_CHROMA) * chroma_weight

        score = proportion_score + chroma_score
        argb_to_score[color] = score

    argb_and_score_sorted = sorted(
        argb_to_score.items(), key=lambda x: x[1], reverse=True
    )
    argbs_score_sorted = [x[0] for x in argb_and_score_sorted]
    final_colors_to_score = {}
    for difference in range(90, 15, -1):
        final_colors_to_score.clear()
        for color in argbs_score_sorted:
            duplicate_hue = False
            cam = argb_to_hct[color]
            for already_chosen_color in final_colors_to_score.keys():
                already_chosen_cam = argb_to_hct[already_chosen_color]
                if difference_degrees(cam.hue, already_chosen_cam.hue) < difference:
                    duplicate_hue = True
                    break
            if not duplicate_hue:
                final_colors_to_score[color] = argb_to_score[color]
        if len(final_colors_to_score) >= desired:
            break

    # Ensure the list of colors returned is sorted such that the first in the
    # list is the most suitable, and the last is the least suitable.
    colors_by_score_descending = [
        ArgbAndScore(color, score) for color, score in final_colors_to_score.items()
    ]
    colors_by_score_descending.sort(reverse=True)

    # Ensure that at least one color is returned.
    if len(colors_by_score_descending) == 0:
        return [0xFF4285F4]  # Google Blue
    return [x.argb for x in colors_by_score_descending]


def _filter(
    colors_to_excited_proportion: Dict[int, float], argb_to_hct: Dict[int, Hct]
) -> List[int]:
    filtered = []
    for color, cam in argb_to_hct.items():
        proportion = colors_to_excited_proportion[color]
        if cam.chroma >= _CUTOFF_CHROMA and proportion > _CUTOFF_EXCITED_PROPORTION:
            filtered.append(color)
    return filtered


def argb_to_proportion(argb_to_count: Dict[int, int]) -> Dict[int, float]:
    total_population = sum(argb_to_count.values())
    argb_to_hct = {key: Hct.from_int(key) for key in argb_to_count.keys()}
    hue_proportions = [0.0] * 360
    for argb in argb_to_hct.keys():
        cam = argb_to_hct[argb]
        hue = np.floor(cam.hue)
        hue_proportions[hue] += argb_to_count[argb] / total_population

    # Determine the proportion of the colors around each color, by summing the
    # proportions around each color's hue.
    int_to_proportion = {}
    for argb, cam in argb_to_hct.items():
        hue = np.round(cam.hue)

        excited_proportion = 0.0
        for i in range(hue - 15, hue + 15):
            neighbor_hue = sanitize_degrees(i)
            excited_proportion += hue_proportions[neighbor_hue]
        int_to_proportion[argb] = excited_proportion
    return int_to_proportion
