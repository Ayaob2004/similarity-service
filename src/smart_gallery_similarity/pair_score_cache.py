"""Persistent incremental pair-score cache.

Candidate generation is cheap; mutual SIFT matching is not. This module caches
pair components and final combined similarity in SQLite so rerunning the same
incremental preparation does not repeat BFMatcher work.

A pair can be reused at two levels:

* ``hit``: content, component fingerprint, capture times and score fingerprint
  all match; the full score is reused;
* ``rescored``: visual content/components still match, but capture time or
  score settings changed; SMR/color are reused and only the cheap final score
  is recomputed;
* ``miss`` / ``stale``: SIFT matching is computed because no usable visual
  component cache exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from smart_gallery_similarity.candidate_selection import CandidatePair
from smart_gallery_similarity.change_detection import CurrentGalleryItem
from smart_gallery_similarity.feature_cache import FeatureCache, FeatureCacheError
from smart_gallery_similarity.photocluster.sift_features import (
    mutual_best_matches,
    sift_matching_ratio,
)
from smart_gallery_similarity.production_config import ProductionPhotoClusterConfig
from smart_gallery_similarity.state_store import (
    GalleryStateStore,
    PairScoreRecord,
    StateStoreError,
)


class PairScoreCacheError(RuntimeError):
    """Raised when pair-score preparation cannot continue safely."""


PairCacheStatus = Literal["hit", "miss", "stale", "rescored"]


@dataclass(frozen=True)
class PreparedPairScore:
    asset_id_a: str
    asset_id_b: str
    cache_status: PairCacheStatus
    keypoint_count_a: int
    keypoint_count_b: int
    match_count: int
    smr: float
    color_distance: float
    time_distance: float
    combined_similarity: float
    reasons: tuple[str, ...]
    previous_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairScorePreparationResult:
    scores: tuple[PreparedPairScore, ...]

    @property
    def pair_count(self) -> int:
        return len(self.scores)

    @property
    def cache_hits(self) -> int:
        return sum(score.cache_status == "hit" for score in self.scores)

    @property
    def cache_misses(self) -> int:
        return sum(score.cache_status == "miss" for score in self.scores)

    @property
    def cache_stale_rebuilt(self) -> int:
        return sum(score.cache_status == "stale" for score in self.scores)

    @property
    def cache_rescored(self) -> int:
        return sum(score.cache_status == "rescored" for score in self.scores)

    @property
    def sift_match_computation_count(self) -> int:
        return self.cache_misses + self.cache_stale_rebuilt

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_stale_rebuilt": self.cache_stale_rebuilt,
            "cache_rescored": self.cache_rescored,
            "sift_match_computation_count": self.sift_match_computation_count,
            "scores": [score.to_dict() for score in self.scores],
        }


def _combined_similarity(
    *,
    smr: float,
    color_distance: float,
    time_distance: float,
    config: ProductionPhotoClusterConfig,
) -> float:
    weights = config.combined_distance
    sift_distance = 1.0 - float(smr)
    distance = (
        weights.sift_weight * sift_distance
        + weights.color_weight * float(color_distance)
        + weights.time_weight * float(time_distance)
    ) / weights.total_weight
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _record_visual_components_valid(
    record: Mapping[str, object],
    *,
    item_a: CurrentGalleryItem,
    item_b: CurrentGalleryItem,
    component_fingerprint: str,
) -> bool:
    return bool(
        str(record.get("component_fingerprint", "")) == component_fingerprint
        and str(record.get("content_sha256_a", "")) == item_a.content_sha256
        and str(record.get("content_sha256_b", "")) == item_b.content_sha256
    )


def _record_full_score_valid(
    record: Mapping[str, object],
    *,
    item_a: CurrentGalleryItem,
    item_b: CurrentGalleryItem,
    score_fingerprint: str,
) -> bool:
    return bool(
        str(record.get("score_fingerprint", "")) == score_fingerprint
        and str(record.get("capture_time_a", "")) == item_a.capture_time
        and str(record.get("capture_time_b", "")) == item_b.capture_time
    )


def prepare_candidate_pair_scores(
    *,
    candidate_pairs: Sequence[CandidatePair],
    current_items_by_asset_id: Mapping[str, CurrentGalleryItem],
    feature_cache: FeatureCache,
    state_store: GalleryStateStore,
    config: ProductionPhotoClusterConfig,
) -> PairScorePreparationResult:
    """Prepare/cache all candidate-pair scores without changing group state."""

    config.validate()
    component_fingerprint = config.pair_component_fingerprint()
    score_fingerprint = config.pair_score_fingerprint()
    feature_fingerprint = config.feature_fingerprint()

    results: list[PreparedPairScore] = []
    try:
        for candidate in candidate_pairs:
            item_a = current_items_by_asset_id.get(candidate.asset_id_a)
            item_b = current_items_by_asset_id.get(candidate.asset_id_b)
            if item_a is None or item_b is None:
                raise PairScoreCacheError(
                    "Candidate pair references an asset missing from the current gallery: "
                    f"{candidate.asset_id_a} / {candidate.asset_id_b}"
                )

            cached = state_store.pair_score(candidate.asset_id_a, candidate.asset_id_b)
            visual_valid = cached is not None and _record_visual_components_valid(
                cached,
                item_a=item_a,
                item_b=item_b,
                component_fingerprint=component_fingerprint,
            )
            full_valid = visual_valid and _record_full_score_valid(
                cached,
                item_a=item_a,
                item_b=item_b,
                score_fingerprint=score_fingerprint,
            )

            if full_valid:
                assert cached is not None
                result = PreparedPairScore(
                    asset_id_a=candidate.asset_id_a,
                    asset_id_b=candidate.asset_id_b,
                    cache_status="hit",
                    keypoint_count_a=int(cached["keypoint_count_a"]),
                    keypoint_count_b=int(cached["keypoint_count_b"]),
                    match_count=int(cached["match_count"]),
                    smr=float(cached["smr"]),
                    color_distance=float(cached["color_distance"]),
                    time_distance=float(cached["time_distance"]),
                    combined_similarity=float(cached["combined_similarity"]),
                    reasons=candidate.reasons,
                    previous_group_ids=candidate.previous_group_ids,
                )
                results.append(result)
                continue

            if visual_valid:
                assert cached is not None
                status: PairCacheStatus = "rescored"
                keypoint_count_a = int(cached["keypoint_count_a"])
                keypoint_count_b = int(cached["keypoint_count_b"])
                match_count = int(cached["match_count"])
                smr = float(cached["smr"])
            else:
                status = "miss" if cached is None else "stale"
                features_a = feature_cache.load(
                    asset_id=item_a.asset_id,
                    content_sha256=item_a.content_sha256,
                    feature_fingerprint=feature_fingerprint,
                )
                features_b = feature_cache.load(
                    asset_id=item_b.asset_id,
                    content_sha256=item_b.content_sha256,
                    feature_fingerprint=feature_fingerprint,
                )
                keypoint_count_a = int(features_a.descriptors.shape[0])
                keypoint_count_b = int(features_b.descriptors.shape[0])
                matches = mutual_best_matches(
                    features_a.descriptors,
                    features_b.descriptors,
                )
                match_count = len(matches)
                smr = sift_matching_ratio(
                    keypoint_count_a,
                    keypoint_count_b,
                    match_count,
                )

            similarity = _combined_similarity(
                smr=smr,
                color_distance=candidate.color_distance,
                time_distance=candidate.time_distance,
                config=config,
            )

            state_store.upsert_pair_score(
                PairScoreRecord(
                    asset_id_a=candidate.asset_id_a,
                    asset_id_b=candidate.asset_id_b,
                    content_sha256_a=item_a.content_sha256,
                    content_sha256_b=item_b.content_sha256,
                    capture_time_a=item_a.capture_time,
                    capture_time_b=item_b.capture_time,
                    component_fingerprint=component_fingerprint,
                    score_fingerprint=score_fingerprint,
                    keypoint_count_a=keypoint_count_a,
                    keypoint_count_b=keypoint_count_b,
                    match_count=match_count,
                    smr=smr,
                    color_distance=float(candidate.color_distance),
                    time_distance=float(candidate.time_distance),
                    combined_similarity=similarity,
                )
            )

            results.append(
                PreparedPairScore(
                    asset_id_a=candidate.asset_id_a,
                    asset_id_b=candidate.asset_id_b,
                    cache_status=status,
                    keypoint_count_a=keypoint_count_a,
                    keypoint_count_b=keypoint_count_b,
                    match_count=match_count,
                    smr=float(smr),
                    color_distance=float(candidate.color_distance),
                    time_distance=float(candidate.time_distance),
                    combined_similarity=similarity,
                    reasons=candidate.reasons,
                    previous_group_ids=candidate.previous_group_ids,
                )
            )
    except (FeatureCacheError, StateStoreError, ValueError) as exc:
        if isinstance(exc, PairScoreCacheError):
            raise
        raise PairScoreCacheError(str(exc)) from exc

    return PairScorePreparationResult(scores=tuple(results))
