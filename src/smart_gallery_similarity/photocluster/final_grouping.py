"""Convert pairwise PhotoCluster similarities into disjoint image groups.

The production pipeline uses deterministic agglomerative clustering inside each
initial set.  Single linkage is intentionally excluded because one bridge image
can otherwise join two different scenes.  The calibrated production operating
point is supplied by :class:`FinalGroupingConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class FinalGroupingError(RuntimeError):
    """Raised when final grouping cannot be completed safely."""


SUPPORTED_LINKAGES = ("average", "complete")


@dataclass(frozen=True)
class FinalGroupingConfig:
    """Fixed settings for final non-overlapping groups."""

    similarity_threshold: float
    linkage: str = "average"
    minimum_group_size: int = 2

    def validate(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1].")
        if self.linkage not in SUPPORTED_LINKAGES:
            raise ValueError(
                f"linkage must be one of {SUPPORTED_LINKAGES}; found {self.linkage!r}."
            )
        if self.minimum_group_size < 2:
            raise ValueError("minimum_group_size must be at least 2.")


@dataclass(frozen=True)
class FinalGroupingResult:
    """Disjoint groups and one assignment label for every image."""

    groups: tuple[tuple[int, ...], ...]
    assignments: np.ndarray
    singleton_rows: tuple[int, ...]


def validate_initial_sets(
    initial_sets: Sequence[dict[str, Any]],
    *,
    image_count: int,
) -> list[tuple[str, np.ndarray]]:
    """Validate that initial sets cover every image exactly once."""

    if image_count < 1:
        raise ValueError("image_count must be at least 1.")
    if not initial_sets:
        raise FinalGroupingError("initial_sets cannot be empty.")

    seen = np.zeros(image_count, dtype=bool)
    normalized: list[tuple[str, np.ndarray]] = []
    for item in initial_sets:
        set_id = str(item.get("set_id", "")).strip()
        rows = np.asarray(item.get("member_rows", []), dtype=np.int64)
        if not set_id:
            raise FinalGroupingError("Every initial set needs a set_id.")
        if rows.ndim != 1 or rows.size < 1:
            raise FinalGroupingError(f"Invalid member_rows for {set_id}.")
        if np.any(rows < 0) or np.any(rows >= image_count):
            raise FinalGroupingError(f"Out-of-range row inside {set_id}.")
        if np.unique(rows).size != rows.size:
            raise FinalGroupingError(f"Duplicate row inside {set_id}.")
        if np.any(seen[rows]):
            raise FinalGroupingError(
                f"An image appears in more than one initial set ({set_id})."
            )
        seen[rows] = True
        normalized.append((set_id, np.sort(rows)))

    missing = np.flatnonzero(~seen)
    if missing.size:
        raise FinalGroupingError(
            f"Initial sets do not cover {missing.size} image(s)."
        )
    return normalized


def validate_combined_similarity(
    similarity: np.ndarray,
    *,
    initial_sets: Sequence[dict[str, Any]],
    expected_count: int | None = None,
) -> np.ndarray:
    """Validate Step-4 scores, allowing NaN only across initial sets."""

    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise FinalGroupingError(
            f"combined_similarity must be square; found {matrix.shape}."
        )
    count = int(matrix.shape[0])
    if expected_count is not None and count != expected_count:
        raise FinalGroupingError(
            f"combined_similarity has {count} rows; expected {expected_count}."
        )

    normalized_sets = validate_initial_sets(initial_sets, image_count=count)
    finite_mask = np.zeros((count, count), dtype=bool)
    for _, rows in normalized_sets:
        finite_mask[np.ix_(rows, rows)] = True
        local = matrix[np.ix_(rows, rows)]
        if not np.isfinite(local).all():
            raise FinalGroupingError(
                "combined_similarity contains NaN/inf inside an initial set."
            )
        if np.any(local < -1e-7) or np.any(local > 1.0 + 1e-7):
            raise FinalGroupingError(
                "combined_similarity contains values outside [0, 1]."
            )
        if not np.allclose(local, local.T, atol=1e-6):
            raise FinalGroupingError(
                "combined_similarity is not symmetric inside an initial set."
            )
        if not np.allclose(np.diag(local), 1.0, atol=1e-6):
            raise FinalGroupingError(
                "combined_similarity diagonal must be one inside each set."
            )

    outside = matrix[~finite_mask]
    if outside.size and np.any(np.isfinite(outside)):
        finite_values = outside[np.isfinite(outside)]
        if np.any(np.abs(finite_values) > 1e-7):
            raise FinalGroupingError(
                "Cross-set combined similarities must be NaN or zero."
            )

    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_local_similarity(local_similarity: np.ndarray) -> np.ndarray:
    similarity = np.asarray(local_similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise FinalGroupingError(
            f"local_combined_similarity must be square; found {similarity.shape}."
        )
    if not np.isfinite(similarity).all():
        raise FinalGroupingError("local_combined_similarity contains NaN/inf.")
    if np.any(similarity < -1e-7) or np.any(similarity > 1.0 + 1e-7):
        raise FinalGroupingError(
            "local_combined_similarity contains values outside [0, 1]."
        )
    if not np.allclose(similarity, similarity.T, atol=1e-6):
        raise FinalGroupingError("local_combined_similarity must be symmetric.")
    if not np.allclose(np.diag(similarity), 1.0, atol=1e-6):
        raise FinalGroupingError(
            "local_combined_similarity diagonal must be one."
        )
    return np.ascontiguousarray(similarity, dtype=np.float64)


def _stable_relabel(labels: Iterable[int]) -> np.ndarray:
    """Relabel clusters by their smallest member for deterministic output."""

    values = np.asarray(list(labels), dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(values.tolist()):
        groups.setdefault(int(label), []).append(index)
    ordered = sorted(groups.values(), key=lambda members: (members[0], members))
    result = np.empty(values.size, dtype=np.int32)
    for stable_label, members in enumerate(ordered):
        result[np.asarray(members, dtype=np.int64)] = stable_label
    return result


def cluster_local_similarity(
    local_similarity: np.ndarray,
    *,
    similarity_threshold: float,
    linkage: str,
) -> np.ndarray:
    """Partition one initial set with average/complete agglomeration."""

    if linkage not in SUPPORTED_LINKAGES:
        raise ValueError(f"Unsupported linkage: {linkage}")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1].")

    similarity = _validate_local_similarity(local_similarity)
    count = int(similarity.shape[0])
    if count == 1:
        return np.zeros(1, dtype=np.int32)

    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance_threshold = max(1.0 - float(similarity_threshold), 0.0) + 1e-12
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
    )
    return _stable_relabel(model.fit_predict(distance))


def build_non_overlapping_groups(
    *,
    combined_similarity: np.ndarray,
    initial_sets: Sequence[dict[str, Any]],
    config: FinalGroupingConfig,
) -> FinalGroupingResult:
    """Cluster every initial set and return disjoint groups of size >= 2."""

    config.validate()
    matrix = validate_combined_similarity(
        combined_similarity,
        initial_sets=initial_sets,
    )
    normalized_sets = validate_initial_sets(
        initial_sets,
        image_count=matrix.shape[0],
    )

    groups: list[tuple[int, ...]] = []
    singleton_rows: list[int] = []
    for _, rows in normalized_sets:
        labels = cluster_local_similarity(
            matrix[np.ix_(rows, rows)],
            similarity_threshold=config.similarity_threshold,
            linkage=config.linkage,
        )
        for label in sorted(np.unique(labels).tolist()):
            members = tuple(sorted(rows[labels == label].astype(int).tolist()))
            if len(members) >= config.minimum_group_size:
                groups.append(members)
            else:
                singleton_rows.extend(members)

    groups.sort(key=lambda members: (members[0], len(members), members))
    assignments = np.full(matrix.shape[0], -1, dtype=np.int32)
    for group_index, members in enumerate(groups):
        rows = np.asarray(members, dtype=np.int64)
        if np.any(assignments[rows] != -1):
            raise FinalGroupingError("Final groups overlap.")
        assignments[rows] = group_index

    assigned_rows = set(np.flatnonzero(assignments >= 0).astype(int).tolist())
    singleton_rows = sorted(
        set(singleton_rows).union(set(range(matrix.shape[0])) - assigned_rows)
    )
    return FinalGroupingResult(
        groups=tuple(groups),
        assignments=assignments,
        singleton_rows=tuple(singleton_rows),
    )


def group_medoid(
    members: Sequence[int],
    *,
    combined_similarity: np.ndarray,
) -> int:
    """Return the most central member by mean within-group similarity."""

    rows = np.asarray(members, dtype=np.int64)
    if rows.ndim != 1 or rows.size < 1:
        raise ValueError("members cannot be empty.")
    local = np.asarray(combined_similarity, dtype=np.float64)[np.ix_(rows, rows)]
    if not np.isfinite(local).all():
        raise FinalGroupingError("Group contains non-finite similarity values.")
    if rows.size == 1:
        return int(rows[0])
    means = (local.sum(axis=1) - 1.0) / (rows.size - 1)
    best_value = float(means.max())
    candidates = rows[np.isclose(means, best_value, atol=1e-12)]
    return int(candidates.min())


def group_similarity_summary(
    members: Sequence[int],
    *,
    combined_similarity: np.ndarray,
) -> dict[str, float]:
    """Summarize unique pair similarities inside one saved group."""

    rows = np.asarray(members, dtype=np.int64)
    if rows.size < 2:
        raise ValueError("A saved group must contain at least two images.")
    local = np.asarray(combined_similarity, dtype=np.float64)[np.ix_(rows, rows)]
    values = local[np.triu_indices(rows.size, k=1)]
    if not np.isfinite(values).all():
        raise FinalGroupingError("Group contains non-finite pair similarities.")
    return {
        "minimum_similarity": float(values.min()),
        "mean_similarity": float(values.mean()),
        "median_similarity": float(np.median(values)),
        "maximum_similarity": float(values.max()),
    }
