"""Stable capture-time distance policies for production and incremental use.

The original PhotoCluster research implementation normalizes a pair's time
separation by the oldest-to-newest span of the whole collection. That is useful
for reproducing the paper, but it is not stable for incremental processing:
adding one very old or very new image changes previously computed pair scores.

Production therefore supports a fixed-horizon policy where the distance of a
pair depends only on the two capture times and one frozen configuration value::

    distance = min(abs(t_a - t_b) / horizon_seconds, 1.0)

The legacy ``collection_span`` policy remains available for reproducibility and
A/B comparisons, but it must not be used by the future incremental updater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from smart_gallery_similarity.photocluster.global_features import (
    normalized_time_distance_matrix,
)


SUPPORTED_TIME_DISTANCE_MODES = ("collection_span", "fixed_horizon")


@dataclass(frozen=True)
class TimeDistanceConfig:
    """Configuration for capture-time distance calculation."""

    mode: str = "fixed_horizon"
    fixed_horizon_seconds: float = 86_400.0

    def validate(self) -> None:
        if self.mode not in SUPPORTED_TIME_DISTANCE_MODES:
            raise ValueError(
                "time distance mode must be one of "
                f"{SUPPORTED_TIME_DISTANCE_MODES}; found {self.mode!r}."
            )
        if not np.isfinite(self.fixed_horizon_seconds):
            raise ValueError("fixed_horizon_seconds must be finite.")
        if self.fixed_horizon_seconds <= 0:
            raise ValueError("fixed_horizon_seconds must be positive.")

    @property
    def is_incremental_safe(self) -> bool:
        """Whether old pairwise time distances stay unchanged as gallery grows."""

        return self.mode == "fixed_horizon"

    @property
    def fixed_horizon_hours(self) -> float:
        return float(self.fixed_horizon_seconds / 3600.0)


def fixed_horizon_time_distance_matrix(
    capture_seconds: Sequence[float] | np.ndarray,
    *,
    horizon_seconds: float,
) -> np.ndarray:
    """Return pairwise time distances using a collection-independent horizon.

    The output is symmetric, has a zero diagonal, and is clipped to ``[0, 1]``.
    Adding unrelated images to the collection cannot change an already computed
    pair's distance.
    """

    times = np.asarray(capture_seconds, dtype=np.float64)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("capture_seconds must be a non-empty 1-D array.")
    if not np.isfinite(times).all():
        raise ValueError("capture_seconds contains NaN or infinite values.")

    horizon = float(horizon_seconds)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon_seconds must be a positive finite number.")

    absolute_differences = np.abs(times[:, None] - times[None, :])
    distances = np.clip(absolute_differences / horizon, 0.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def build_time_distance_matrix(
    capture_seconds: Sequence[float] | np.ndarray,
    *,
    config: TimeDistanceConfig,
) -> np.ndarray:
    """Build a time-distance matrix from the configured policy."""

    config.validate()
    if config.mode == "collection_span":
        return normalized_time_distance_matrix(capture_seconds)
    return fixed_horizon_time_distance_matrix(
        capture_seconds,
        horizon_seconds=config.fixed_horizon_seconds,
    )
