from typing import Dict, List, Optional

import numpy as np

from .point_provider import PointProvider
from .point_provider_lab import PointProviderLab
from .quantizer import Quantizer, QuantizerResult


class QuantizerWsmeans(Quantizer):
    def __init__(
        self,
        unique_pixels: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        super().__init__()
        self._unique_pixels = unique_pixels
        self._counts = counts

    def quantize(
        self,
        max_colors: int,
        starting_clusters: List[int] = None,
        point_provider: PointProvider = None,
        max_iterations: int = 5,
        return_input_pixel_to_cluster_pixel: bool = False,
    ) -> QuantizerResult:
        if starting_clusters is None:
            starting_clusters = []

        if point_provider is None:
            point_provider = PointProviderLab()

        point_count = self._unique_pixels.shape[0]
        points = point_provider.from_int(self._unique_pixels)

        cluster_count = min(max_colors, point_count)

        additional_clusters_needed = cluster_count - len(starting_clusters)
        clusters = np.array(
            [
                *[point_provider.from_int(e) for e in starting_clusters],
                *points[
                    np.random.RandomState(0x42688)
                    .choice(point_count, additional_clusters_needed, replace=False)
                    .astype(np.int32)
                ],
            ]
        )

        cluster_indices = np.arange(point_count) % cluster_count

        pixel_count_sums = np.zeros(cluster_count, dtype=np.int32)
        for iteration in range(max_iterations):
            points_moved = 0
            distance_to_index_matrix = np.linalg.norm(
                clusters[:, None, :] - clusters[None, :, :], axis=-1
            )
            distance_to_index_matrix.sort()

            previous_clusters = clusters[cluster_indices[:point_count]]
            previous_distances = np.sum(
                (points[:point_count] - previous_clusters) ** 2, axis=-1
            )
            filtered_clusters = np.broadcast_to(
                clusters, (points.shape[0], *clusters.shape)
            )[
                (
                    distance_to_index_matrix[cluster_indices[:point_count]]
                    < 4 * previous_distances[:, None]
                )
            ][
                :cluster_count
            ]
            distances = np.sum(
                (points[:, None] - filtered_clusters) ** 2,
                axis=-1,
            )
            if distances.size != 0:
                min_distance_indices = np.argmin(distances, axis=1)
                min_distances = distances[
                    np.arange(distances.shape[0]), min_distance_indices
                ]
                shorter_distance_indices = np.argwhere(
                    min_distances < previous_distances
                )
                cluster_indices[shorter_distance_indices] = min_distance_indices[
                    shorter_distance_indices
                ]
                points_moved += len(shorter_distance_indices)

            # if len(distances) == 0:
            #     continue

            if points_moved == 0 and iteration > 0:
                break

            component_sums = np.zeros((cluster_count, 3), dtype=np.float64)

            pixel_count_sums[:] = 0
            np.add.at(pixel_count_sums, cluster_indices, self._counts)
            np.add.at(component_sums, cluster_indices, points * self._counts[:, None])

            clusters = np.where(
                (pixel_count_sums == 0)[:, None],
                np.zeros((cluster_count, 3)),
                component_sums / pixel_count_sums[:, None],
            )

        cluster_argbs = []
        cluster_populations = []
        input_pixel_to_cluster_pixel: Optional[Dict[int, int]] = None
        for i in range(cluster_count):
            count = pixel_count_sums[i]
            if count == 0:
                continue

            possible_new_cluster = point_provider.to_int(clusters[i])
            if possible_new_cluster in cluster_argbs:
                continue

            cluster_argbs.append(possible_new_cluster)
            cluster_populations.append(count)

        if return_input_pixel_to_cluster_pixel:
            input_pixel_to_cluster_pixel = {}
            for i in range(len(self._unique_pixels)):
                input_pixel = self._unique_pixels[i]
                cluster_index = self._cluster_indices[i]
                cluster = clusters[cluster_index]
                cluster_pixel = point_provider.to_int(cluster)
                input_pixel_to_cluster_pixel[input_pixel] = cluster_pixel

        return QuantizerResult(
            dict(zip(cluster_argbs, cluster_populations)),
            input_pixel_to_cluster_pixel=input_pixel_to_cluster_pixel,
        )
