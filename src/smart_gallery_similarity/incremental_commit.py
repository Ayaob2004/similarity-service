"""Atomic commit helpers for incremental Smart Gallery updates.

This stage turns the non-committing local grouping proposal into the next
canonical gallery baseline. It intentionally keeps the expensive derived caches
(feature NPZ files and valid pair scores) separate from authoritative state:

* gallery items and final groups are committed in one SQLite transaction;
* existing group IDs are retained;
* temporary ``pending_new_*`` IDs are replaced atomically with monotonic stable
  ``group_NNNN`` IDs;
* pair-score rows touching deleted assets are removed in that same transaction;
* feature-cache files for deleted assets are removed only *after* the database
  commit, so a pre-commit failure cannot destroy the last usable baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from smart_gallery_similarity.change_detection import CurrentGalleryItem, GalleryChangeSet
from smart_gallery_similarity.feature_cache import FeatureCache, FeatureCacheError
from smart_gallery_similarity.incremental_grouping import IncrementalGroupingPlan
from smart_gallery_similarity.production_config import ProductionPhotoClusterConfig
from smart_gallery_similarity.state_store import (
    GalleryStateGroup,
    GalleryStateItem,
    GalleryStateStore,
    IncrementalStateCommitResult,
    StateStoreError,
)


class IncrementalCommitError(RuntimeError):
    """Raised when a proposed incremental update cannot be committed safely."""


@dataclass(frozen=True)
class IncrementalCommitResult:
    """Authoritative state/cleanup result after one successful incremental commit."""

    database_result: IncrementalStateCommitResult
    deleted_feature_cache_count: int
    deleted_feature_cache_missing_count: int
    cleanup_warnings: tuple[str, ...]
    stable_groups: tuple[GalleryStateGroup, ...]

    @property
    def image_count(self) -> int:
        return self.database_result.image_count

    @property
    def group_count(self) -> int:
        return self.database_result.group_count

    @property
    def grouped_image_count(self) -> int:
        return self.database_result.grouped_image_count

    @property
    def ungrouped_image_count(self) -> int:
        return self.database_result.ungrouped_image_count

    @property
    def state_revision(self) -> int:
        return self.database_result.state_revision

    @property
    def group_id_assignments(self) -> tuple[tuple[str, str], ...]:
        return self.database_result.group_id_assignments

    def to_dict(self) -> dict[str, object]:
        return {
            "state_revision": self.state_revision,
            "image_count": self.image_count,
            "group_count": self.group_count,
            "grouped_image_count": self.grouped_image_count,
            "ungrouped_image_count": self.ungrouped_image_count,
            "group_id_assignments": [
                {"temporary_group_id": temporary, "stable_group_id": stable}
                for temporary, stable in self.group_id_assignments
            ],
            "removed_orphan_pair_score_count": (
                self.database_result.removed_orphan_pair_score_count
            ),
            "deleted_feature_cache_count": self.deleted_feature_cache_count,
            "deleted_feature_cache_missing_count": self.deleted_feature_cache_missing_count,
            "cleanup_warnings": list(self.cleanup_warnings),
            "groups": [asdict(group) for group in self.stable_groups],
        }


def _build_state_items(
    *,
    current_items: Sequence[CurrentGalleryItem],
    feature_cache: FeatureCache,
    feature_fingerprint: str,
) -> tuple[GalleryStateItem, ...]:
    """Convert the current scanned gallery into SQLite state rows."""

    items: list[GalleryStateItem] = []
    seen: set[str] = set()
    for item in current_items:
        if item.asset_id in seen:
            raise IncrementalCommitError(
                f"Current gallery contains duplicate asset_id={item.asset_id}."
            )
        seen.add(item.asset_id)

        # Every current asset must have a raw feature entry by this point: old
        # unchanged assets came from the full-build baseline, and Added/Modified
        # assets were ensured by Step 4. A missing cache file means the local state
        # directory is incomplete/corrupt and should not become authoritative.
        cache_path = feature_cache.entry_path(item.asset_id)
        if not cache_path.is_file():
            raise IncrementalCommitError(
                "Feature cache entry is missing before commit for "
                f"asset_id={item.asset_id}: {cache_path}"
            )

        items.append(
            GalleryStateItem(
                asset_id=item.asset_id,
                filename=item.filename,
                relative_path=item.relative_path,
                capture_time=item.capture_time,
                capture_time_source=item.capture_time_source,
                width=int(item.width),
                height=int(item.height),
                content_sha256=item.content_sha256,
                feature_cache_key=feature_cache.cache_key(item.asset_id),
                feature_fingerprint=feature_fingerprint,
            )
        )
    return tuple(items)


def _planned_groups_for_database(plan: IncrementalGroupingPlan) -> tuple[GalleryStateGroup, ...]:
    return tuple(
        GalleryStateGroup(
            group_id=group.group_id,
            representative_asset_id=group.representative_asset_id,
            member_asset_ids=tuple(group.member_asset_ids),
        )
        for group in plan.groups
    )


def _resolve_stable_groups(
    groups: Sequence[GalleryStateGroup],
    assignments: Sequence[tuple[str, str]],
) -> tuple[GalleryStateGroup, ...]:
    mapping = dict(assignments)
    return tuple(
        GalleryStateGroup(
            group_id=mapping.get(group.group_id, group.group_id),
            representative_asset_id=group.representative_asset_id,
            member_asset_ids=tuple(group.member_asset_ids),
        )
        for group in groups
    )


def commit_grouping_plan(
    *,
    state_store: GalleryStateStore,
    feature_cache: FeatureCache,
    current_items: Sequence[CurrentGalleryItem],
    change_set: GalleryChangeSet,
    grouping_plan: IncrementalGroupingPlan,
    config: ProductionPhotoClusterConfig,
    expected_state_revision: int,
) -> IncrementalCommitResult:
    """Commit one already-built local group plan and clean deleted feature files."""

    config.validate()
    if change_set.requires_full_rebuild:
        raise IncrementalCommitError(
            "Incremental commit is blocked because a full rebuild is required: "
            + ", ".join(change_set.rebuild_reasons)
        )

    current_ids = {item.asset_id for item in current_items}
    plan_ids = {
        asset_id
        for group in grouping_plan.groups
        for asset_id in group.member_asset_ids
    } | set(grouping_plan.ungrouped_asset_ids)
    if current_ids != plan_ids:
        missing = sorted(current_ids - plan_ids)
        extra = sorted(plan_ids - current_ids)
        raise IncrementalCommitError(
            "Grouping plan does not match the current gallery. "
            f"missing_from_plan={missing}, extra_in_plan={extra}"
        )

    feature_fingerprint = config.feature_fingerprint()
    state_items = _build_state_items(
        current_items=current_items,
        feature_cache=feature_cache,
        feature_fingerprint=feature_fingerprint,
    )
    planned_groups = _planned_groups_for_database(grouping_plan)

    try:
        database_result = state_store.commit_incremental_snapshot(
            items=state_items,
            groups=planned_groups,
            pipeline_fingerprint=config.fingerprint(),
            feature_fingerprint=feature_fingerprint,
            expected_state_revision=int(expected_state_revision),
        )
    except (StateStoreError, ValueError) as exc:
        raise IncrementalCommitError(str(exc)) from exc

    stable_groups = _resolve_stable_groups(
        planned_groups,
        database_result.group_id_assignments,
    )

    deleted_count = 0
    missing_count = 0
    cleanup_warnings: list[str] = []
    for change in change_set.deleted:
        try:
            removed = feature_cache.delete(change.asset_id)
        except FeatureCacheError as exc:
            # The database is already committed. A stale orphan NPZ is harmless,
            # so report it instead of pretending the authoritative commit failed.
            cleanup_warnings.append(str(exc))
            continue
        if removed:
            deleted_count += 1
        else:
            missing_count += 1

    return IncrementalCommitResult(
        database_result=database_result,
        deleted_feature_cache_count=deleted_count,
        deleted_feature_cache_missing_count=missing_count,
        cleanup_warnings=tuple(cleanup_warnings),
        stable_groups=stable_groups,
    )
