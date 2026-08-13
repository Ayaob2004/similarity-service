"""Conservative candidate generation for incremental PhotoCluster updates.

The full PhotoCluster pipeline uses Affinity Propagation as a broad batch gate
before expensive SIFT matching. Affinity Propagation is not incrementally
stable, so everyday updates need a different gate that does not repartition the
whole gallery.

This module selects candidate pairs with three principles:

1. only pairs inside a fixed capture-time window are considered as ordinary
   seeds (the production time horizon is the default);
2. a cheap color+time upper bound rejects pairs that could not reach the final
   similarity threshold even with a perfect SIFT match;
3. previous groups are expanded around selected seeds so a new/modified image
   can later be assessed against the whole affected group rather than one bridge
   member only.

Every returned pair contains at least one Added/Modified asset. No SIFT work is
performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from smart_gallery_similarity.change_detection import CurrentGalleryItem, GalleryChangeSet
from smart_gallery_similarity.feature_cache import FeatureCache, FeatureCacheError
from smart_gallery_similarity.photocluster.global_features import (
    datetime_to_naive_epoch_seconds,
    parse_exif_datetime,
)
from smart_gallery_similarity.production_config import ProductionPhotoClusterConfig


class CandidateSelectionError(RuntimeError):
    """Raised when incremental candidate generation cannot continue safely."""


@dataclass(frozen=True)
class CandidateSelectionConfig:
    """Runtime policy for the cheap incremental candidate gate.

    ``max_seed_candidates_per_changed`` caps only the ordinary ranked seed list.
    Forced previous-group members and group expansion are allowed to exceed this
    cap because those pairs are important for safe group-level decisions later.
    """

    time_window_seconds: float
    max_seed_candidates_per_changed: int = 128
    expand_previous_groups: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.time_window_seconds) or self.time_window_seconds <= 0:
            raise ValueError("time_window_seconds must be a positive finite number.")
        if self.max_seed_candidates_per_changed < 1:
            raise ValueError("max_seed_candidates_per_changed must be at least 1.")


@dataclass(frozen=True)
class CandidatePair:
    """One unique pair selected for possible expensive SIFT comparison."""

    asset_id_a: str
    asset_id_b: str
    color_distance: float
    time_distance: float
    global_dissimilarity: float
    best_possible_similarity: float
    reasons: tuple[str, ...]
    previous_group_ids: tuple[str, ...]

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.asset_id_a, self.asset_id_b)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChangedAssetCandidateSummary:
    asset_id: str
    eligible_seed_pool_count: int
    selected_seed_count: int
    truncated_seed_count: int
    forced_previous_group_member_count: int
    expanded_previous_group_member_count: int
    final_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelectionResult:
    pairs: tuple[CandidatePair, ...]
    changed_summaries: tuple[ChangedAssetCandidateSummary, ...]
    candidate_group_ids: tuple[str, ...]
    config: CandidateSelectionConfig

    @property
    def unique_pair_count(self) -> int:
        return len(self.pairs)

    @property
    def truncated_changed_asset_count(self) -> int:
        return sum(summary.truncated_seed_count > 0 for summary in self.changed_summaries)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "unique_pair_count": self.unique_pair_count,
            "candidate_group_ids": list(self.candidate_group_ids),
            "truncated_changed_asset_count": self.truncated_changed_asset_count,
            "changed_assets": [summary.to_dict() for summary in self.changed_summaries],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("A candidate pair must contain two different asset IDs.")
    return (left, right) if left < right else (right, left)


def _capture_seconds(items: Sequence[CurrentGalleryItem]) -> np.ndarray:
    try:
        return np.asarray(
            [
                datetime_to_naive_epoch_seconds(parse_exif_datetime(item.capture_time))
                for item in items
            ],
            dtype=np.float64,
        )
    except Exception as exc:
        raise CandidateSelectionError(f"Could not parse capture times: {exc}") from exc


def _load_histogram_matrix(
    *,
    items: Sequence[CurrentGalleryItem],
    cache: FeatureCache,
    feature_fingerprint: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    try:
        for item in items:
            rows.append(
                cache.load_histogram(
                    asset_id=item.asset_id,
                    content_sha256=item.content_sha256,
                    feature_fingerprint=feature_fingerprint,
                )
            )
    except FeatureCacheError as exc:
        raise CandidateSelectionError(str(exc)) from exc

    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    matrix = np.stack(rows).astype(np.float32, copy=False)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise CandidateSelectionError("Cached HS histogram matrix is invalid.")
    return matrix


def _color_distance_vector(histograms: np.ndarray, row: int) -> np.ndarray:
    """Match the project's Bhattacharyya/Hellinger matrix implementation."""

    matrix = np.asarray(histograms, dtype=np.float64)
    if matrix.ndim != 2 or row < 0 or row >= matrix.shape[0]:
        raise ValueError("Invalid histogram matrix or row index.")
    sums = matrix.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise CandidateSelectionError("Every cached histogram must have positive mass.")
    square_roots = np.sqrt(matrix / sums)
    coefficients = square_roots @ square_roots[row]
    coefficients = np.clip(coefficients, 0.0, 1.0)
    distances = np.sqrt(np.clip(1.0 - coefficients, 0.0, 1.0))
    distances[row] = 0.0
    return distances.astype(np.float32)


def _time_distance_vector(
    capture_seconds: np.ndarray,
    row: int,
    *,
    horizon_seconds: float,
) -> np.ndarray:
    times = np.asarray(capture_seconds, dtype=np.float64)
    distances = np.clip(
        np.abs(times - float(times[row])) / float(horizon_seconds),
        0.0,
        1.0,
    )
    distances[row] = 0.0
    return distances.astype(np.float32)


def _previous_group_maps(
    previous_groups: Sequence[dict[str, object]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    by_asset: dict[str, str] = {}
    members_by_group: dict[str, tuple[str, ...]] = {}
    for group in previous_groups:
        group_id = str(group["group_id"])
        members = tuple(str(value) for value in group.get("member_asset_ids", []))
        members_by_group[group_id] = members
        for asset_id in members:
            if asset_id in by_asset:
                raise CandidateSelectionError(
                    f"asset_id={asset_id} appears in more than one previous group."
                )
            by_asset[asset_id] = group_id
    return by_asset, members_by_group


def select_incremental_candidates(
    *,
    current_items: Sequence[CurrentGalleryItem],
    change_set: GalleryChangeSet,
    previous_groups: Sequence[dict[str, object]],
    feature_cache: FeatureCache,
    production_config: ProductionPhotoClusterConfig,
    selection_config: CandidateSelectionConfig,
) -> CandidateSelectionResult:
    """Select unique current-gallery pairs requiring possible SIFT work.

    Candidate generation uses only cached HS histograms, capture times, the
    saved previous group structure, and frozen production settings. It never
    mutates gallery state.
    """

    production_config.validate()
    selection_config.validate()
    if not production_config.time_distance.is_incremental_safe:
        raise CandidateSelectionError(
            "Incremental candidate selection requires fixed-horizon time distance."
        )

    items = tuple(current_items)
    asset_ids = [item.asset_id for item in items]
    if len(asset_ids) != len(set(asset_ids)):
        raise CandidateSelectionError("Current gallery contains duplicate asset IDs.")
    index_by_asset = {asset_id: index for index, asset_id in enumerate(asset_ids)}

    changed_ids = tuple(change_set.reprocess_asset_ids)
    missing_changed = [asset_id for asset_id in changed_ids if asset_id not in index_by_asset]
    if missing_changed:
        raise CandidateSelectionError(
            "Changed assets missing from current gallery: " + ", ".join(missing_changed)
        )
    if not changed_ids or not items:
        return CandidateSelectionResult(
            pairs=(),
            changed_summaries=tuple(
                ChangedAssetCandidateSummary(
                    asset_id=asset_id,
                    eligible_seed_pool_count=0,
                    selected_seed_count=0,
                    truncated_seed_count=0,
                    forced_previous_group_member_count=0,
                    expanded_previous_group_member_count=0,
                    final_candidate_count=0,
                )
                for asset_id in changed_ids
            ),
            candidate_group_ids=(),
            config=selection_config,
        )

    histogram_matrix = _load_histogram_matrix(
        items=items,
        cache=feature_cache,
        feature_fingerprint=production_config.feature_fingerprint(),
    )
    capture_seconds = _capture_seconds(items)

    previous_group_by_asset, previous_members_by_group = _previous_group_maps(
        previous_groups
    )
    current_asset_set = set(asset_ids)

    global_cfg = production_config.global_features
    combined_cfg = production_config.combined_distance
    final_threshold = float(production_config.final_grouping.similarity_threshold)
    time_horizon = float(production_config.time_distance.fixed_horizon_seconds)

    pair_reasons: dict[tuple[str, str], set[str]] = {}
    pair_group_ids: dict[tuple[str, str], set[str]] = {}
    pair_metrics: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    summaries: list[ChangedAssetCandidateSummary] = []
    all_candidate_group_ids: set[str] = set()

    for changed_id in changed_ids:
        changed_row = index_by_asset[changed_id]
        color = _color_distance_vector(histogram_matrix, changed_row)
        time = _time_distance_vector(
            capture_seconds,
            changed_row,
            horizon_seconds=time_horizon,
        )
        absolute_time = np.abs(capture_seconds - capture_seconds[changed_row])

        global_dissimilarity = (
            global_cfg.color_weight * color + global_cfg.time_weight * time
        ) / float(global_cfg.color_weight + global_cfg.time_weight)

        # Optimistic upper bound: assume perfect SIFT distance = 0.
        best_possible_similarity = 1.0 - (
            combined_cfg.color_weight * color + combined_cfg.time_weight * time
        ) / float(combined_cfg.total_weight)

        ordinary_pool: list[int] = []
        for row, asset_id in enumerate(asset_ids):
            if asset_id == changed_id:
                continue
            if absolute_time[row] > selection_config.time_window_seconds:
                continue
            if best_possible_similarity[row] + 1e-12 < final_threshold:
                continue
            ordinary_pool.append(row)

        ordinary_pool.sort(
            key=lambda row: (
                float(global_dissimilarity[row]),
                float(time[row]),
                float(color[row]),
                asset_ids[row],
            )
        )
        selected_seed_rows = ordinary_pool[
            : selection_config.max_seed_candidates_per_changed
        ]
        selected_seed_ids = {asset_ids[row] for row in selected_seed_rows}
        selected_ids = set(selected_seed_ids)

        forced_previous_members: set[str] = set()
        old_group_id = previous_group_by_asset.get(changed_id)
        if old_group_id is not None:
            all_candidate_group_ids.add(old_group_id)
            forced_previous_members.update(
                asset_id
                for asset_id in previous_members_by_group.get(old_group_id, ())
                if asset_id in current_asset_set and asset_id != changed_id
            )
            selected_ids.update(forced_previous_members)

        expanded_members: set[str] = set()
        if selection_config.expand_previous_groups:
            for seed_id in tuple(selected_ids):
                seed_group = previous_group_by_asset.get(seed_id)
                if seed_group is None:
                    continue
                all_candidate_group_ids.add(seed_group)
                for member_id in previous_members_by_group.get(seed_group, ()):
                    if member_id in current_asset_set and member_id != changed_id:
                        expanded_members.add(member_id)
            selected_ids.update(expanded_members)

        for candidate_id in sorted(selected_ids):
            candidate_row = index_by_asset[candidate_id]
            key = _canonical_pair(changed_id, candidate_id)
            reasons = pair_reasons.setdefault(key, set())
            group_ids = pair_group_ids.setdefault(key, set())

            if candidate_id in selected_seed_ids:
                reasons.add("ranked_seed")
            if candidate_id in forced_previous_members:
                reasons.add("changed_previous_group")
            if candidate_id in expanded_members:
                reasons.add("expanded_previous_group")

            candidate_group = previous_group_by_asset.get(candidate_id)
            if candidate_group is not None:
                group_ids.add(candidate_group)
                all_candidate_group_ids.add(candidate_group)
            if old_group_id is not None:
                group_ids.add(old_group_id)

            metrics = (
                float(color[candidate_row]),
                float(time[candidate_row]),
                float(global_dissimilarity[candidate_row]),
                float(best_possible_similarity[candidate_row]),
            )
            existing = pair_metrics.get(key)
            if existing is not None and not np.allclose(existing, metrics, atol=1e-7):
                raise CandidateSelectionError(
                    f"Inconsistent symmetric metrics for pair {key[0]} / {key[1]}."
                )
            pair_metrics[key] = metrics

        summaries.append(
            ChangedAssetCandidateSummary(
                asset_id=changed_id,
                eligible_seed_pool_count=len(ordinary_pool),
                selected_seed_count=len(selected_seed_rows),
                truncated_seed_count=max(0, len(ordinary_pool) - len(selected_seed_rows)),
                forced_previous_group_member_count=len(forced_previous_members),
                expanded_previous_group_member_count=len(expanded_members),
                final_candidate_count=len(selected_ids),
            )
        )

    pairs: list[CandidatePair] = []
    for key in sorted(pair_metrics):
        color_distance, time_distance, global_distance, best_similarity = pair_metrics[key]
        pairs.append(
            CandidatePair(
                asset_id_a=key[0],
                asset_id_b=key[1],
                color_distance=color_distance,
                time_distance=time_distance,
                global_dissimilarity=global_distance,
                best_possible_similarity=best_similarity,
                reasons=tuple(sorted(pair_reasons.get(key, {"ranked_seed"}))),
                previous_group_ids=tuple(sorted(pair_group_ids.get(key, set()))),
            )
        )

    return CandidateSelectionResult(
        pairs=tuple(pairs),
        changed_summaries=tuple(summaries),
        candidate_group_ids=tuple(sorted(all_candidate_group_ids)),
        config=selection_config,
    )
