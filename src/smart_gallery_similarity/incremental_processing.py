"""Incremental feature preparation built on persistent gallery state.

Step 4 deliberately stops before candidate selection, pair matching, or group
updates. It converts the read-only change set from Step 3 into the minimum raw
feature work needed for the next stages:

- unchanged assets are skipped completely;
- added assets ensure a new HS/SIFT cache entry;
- modified assets ensure a cache entry, rebuilding it only when image bytes or
  feature-extraction settings require that;
- deleted assets are recorded but their cache files are kept until a later
  successful state commit, so a failed incremental update remains recoverable.

The authoritative SQLite snapshot is never mutated by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from smart_gallery_similarity.candidate_selection import (
    CandidateSelectionConfig,
    CandidateSelectionError,
    CandidateSelectionResult,
    select_incremental_candidates,
)
from smart_gallery_similarity.change_detection import (
    CurrentGalleryItem,
    GalleryChangeSet,
    detect_gallery_changes,
)
from smart_gallery_similarity.feature_cache import (
    FeatureCache,
    FeatureCacheError,
    compute_file_sha256,
)
from smart_gallery_similarity.gallery_pipeline import (
    GalleryImage,
    GalleryPipelineError,
    scan_gallery_folder,
)
from smart_gallery_similarity.pair_score_cache import (
    PairScoreCacheError,
    PairScorePreparationResult,
    prepare_candidate_pair_scores,
)
from smart_gallery_similarity.incremental_grouping import (
    IncrementalGroupingError,
    IncrementalGroupingPlan,
    build_incremental_grouping_plan,
)
from smart_gallery_similarity.incremental_safety import (
    IncrementalSafetyAssessment,
    IncrementalSafetyPolicy,
    assess_incremental_safety,
)
from smart_gallery_similarity.incremental_commit import (
    IncrementalCommitError,
    IncrementalCommitResult,
    commit_grouping_plan,
)
from smart_gallery_similarity.production_config import ProductionPhotoClusterConfig
from smart_gallery_similarity.state_store import GalleryStateStore, StateStoreError


class IncrementalProcessingError(RuntimeError):
    """Raised when an incremental preparation cannot safely continue."""


@dataclass(frozen=True)
class CurrentGallerySnapshot:
    """One current scan plus content hashes, ready for change detection."""

    images: tuple[GalleryImage, ...]
    items: tuple[CurrentGalleryItem, ...]
    warnings: tuple[str, ...]

    def images_by_asset_id(self) -> dict[str, GalleryImage]:
        return {image.asset_id: image for image in self.images}

    def items_by_asset_id(self) -> dict[str, CurrentGalleryItem]:
        return {item.asset_id: item for item in self.items}


@dataclass(frozen=True)
class PreparedFeatureAsset:
    """Feature-cache work performed for one Added/Modified asset."""

    asset_id: str
    change_status: Literal["added", "modified"]
    change_reasons: tuple[str, ...]
    cache_status: Literal["hit", "miss", "stale"]
    cache_key: str
    content_sha256: str


@dataclass(frozen=True)
class IncrementalFeaturePreparationResult:
    """Read-only state comparison plus the raw feature work performed."""

    change_set: GalleryChangeSet
    prepared_assets: tuple[PreparedFeatureAsset, ...]
    scan_warnings: tuple[str, ...]
    snapshot: CurrentGallerySnapshot

    @property
    def blocked_by_full_rebuild(self) -> bool:
        return self.change_set.requires_full_rebuild

    @property
    def cache_hits(self) -> int:
        return sum(asset.cache_status == "hit" for asset in self.prepared_assets)

    @property
    def cache_misses(self) -> int:
        return sum(asset.cache_status == "miss" for asset in self.prepared_assets)

    @property
    def cache_stale_rebuilt(self) -> int:
        return sum(asset.cache_status == "stale" for asset in self.prepared_assets)

    @property
    def feature_extraction_count(self) -> int:
        """Number of changed assets that actually required HS/SIFT extraction."""

        return self.cache_misses + self.cache_stale_rebuilt

    @property
    def unchanged_skipped_count(self) -> int:
        return len(self.change_set.unchanged)

    @property
    def deleted_deferred_count(self) -> int:
        return len(self.change_set.deleted)

    def to_dict(self) -> dict[str, object]:
        return {
            "change_set": self.change_set.to_dict(),
            "feature_preparation": {
                "blocked_by_full_rebuild": self.blocked_by_full_rebuild,
                "requested_asset_count": len(self.change_set.reprocess_asset_ids),
                "prepared_asset_count": len(self.prepared_assets),
                "feature_extraction_count": self.feature_extraction_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_stale_rebuilt": self.cache_stale_rebuilt,
                "unchanged_skipped_count": self.unchanged_skipped_count,
                "deleted_deferred_count": self.deleted_deferred_count,
                "prepared_assets": [asdict(asset) for asset in self.prepared_assets],
            },
            "warnings": list(self.scan_warnings),
        }


def scan_current_gallery(
    input_dir: Path,
    *,
    manifest_path: Path | None = None,
    recursive: bool = False,
) -> CurrentGallerySnapshot:
    """Scan current files and attach SHA-256 hashes for stable change detection."""

    images, warnings = scan_gallery_folder(
        input_dir,
        manifest_path=manifest_path,
        recursive=recursive,
        minimum_image_count=0,
    )

    items: list[CurrentGalleryItem] = []
    for image in images:
        items.append(
            CurrentGalleryItem(
                asset_id=image.asset_id,
                filename=image.filename,
                relative_path=image.relative_path,
                capture_time=image.capture_time,
                capture_time_source=image.capture_time_source,
                width=image.width,
                height=image.height,
                content_sha256=compute_file_sha256(image.absolute_path),
            )
        )

    return CurrentGallerySnapshot(
        images=tuple(images),
        items=tuple(items),
        warnings=tuple(warnings),
    )


def detect_snapshot_changes(
    *,
    store: GalleryStateStore,
    snapshot: CurrentGallerySnapshot,
    config: ProductionPhotoClusterConfig,
) -> GalleryChangeSet:
    """Compare a current snapshot with the last authoritative SQLite state."""

    if not store.has_authoritative_snapshot():
        raise IncrementalProcessingError(
            "State exists but has no authoritative full-build snapshot."
        )

    metadata = store.metadata()
    return detect_gallery_changes(
        previous_items=store.items(),
        current_items=snapshot.items,
        previous_group_by_asset_id=store.group_by_asset_id(),
        saved_pipeline_fingerprint=metadata.get("pipeline_fingerprint", ""),
        current_pipeline_fingerprint=config.fingerprint(),
        saved_feature_fingerprint=metadata.get("feature_fingerprint", ""),
        current_feature_fingerprint=config.feature_fingerprint(),
    )


def prepare_incremental_features(
    *,
    state_directory: Path,
    input_dir: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None = None,
    recursive: bool = False,
) -> IncrementalFeaturePreparationResult:
    """Ensure raw features only for Added/Modified assets.

    This function is intentionally non-committing. It may create/replace derived
    feature-cache files, but the authoritative SQLite gallery snapshot and group
    membership remain untouched. Deleted cache entries are also retained until a
    later successful incremental commit.
    """

    config.validate()
    if not config.time_distance.is_incremental_safe:
        raise IncrementalProcessingError(
            "Incremental processing requires an incremental-safe time policy."
        )

    state_root = Path(state_directory).expanduser().resolve()
    database = state_root / "state.sqlite3"
    if not database.is_file():
        raise IncrementalProcessingError(
            f"State database does not exist: {database}. Run a state-enabled full build first."
        )

    try:
        store = GalleryStateStore(database)
        snapshot = scan_current_gallery(
            input_dir,
            manifest_path=manifest_path,
            recursive=recursive,
        )
        change_set = detect_snapshot_changes(
            store=store,
            snapshot=snapshot,
            config=config,
        )
    except (GalleryPipelineError, StateStoreError, FeatureCacheError) as exc:
        raise IncrementalProcessingError(str(exc)) from exc

    # Never prepare work against a state created with incompatible pipeline
    # settings. The caller must perform a fresh full build first.
    if change_set.requires_full_rebuild:
        return IncrementalFeaturePreparationResult(
            change_set=change_set,
            prepared_assets=(),
            scan_warnings=snapshot.warnings,
            snapshot=snapshot,
        )

    images_by_id = snapshot.images_by_asset_id()
    items_by_id = snapshot.items_by_asset_id()
    cache = FeatureCache(state_root / "features")
    feature_fingerprint = config.feature_fingerprint()

    prepared: list[PreparedFeatureAsset] = []
    for change in change_set.changes:
        if change.status not in {"added", "modified"}:
            continue

        image = images_by_id.get(change.asset_id)
        current = items_by_id.get(change.asset_id)
        if image is None or current is None:
            raise IncrementalProcessingError(
                f"Changed current asset is missing from the scan: {change.asset_id}"
            )

        try:
            cache_result = cache.get_or_extract(
                asset_id=change.asset_id,
                image_path=image.absolute_path,
                content_sha256=current.content_sha256,
                feature_fingerprint=feature_fingerprint,
                global_config=config.global_features,
                sift_config=config.sift,
            )
        except FeatureCacheError as exc:
            raise IncrementalProcessingError(str(exc)) from exc

        prepared.append(
            PreparedFeatureAsset(
                asset_id=change.asset_id,
                change_status=change.status,
                change_reasons=change.reasons,
                cache_status=cache_result.status,
                cache_key=cache_result.cache_key,
                content_sha256=current.content_sha256,
            )
        )

    return IncrementalFeaturePreparationResult(
        change_set=change_set,
        prepared_assets=tuple(prepared),
        scan_warnings=snapshot.warnings,
        snapshot=snapshot,
    )


@dataclass(frozen=True)
class IncrementalPairPreparationResult:
    """Step-5 result: candidates plus cached/recomputed pair scores."""

    feature_preparation: IncrementalFeaturePreparationResult
    candidate_selection: CandidateSelectionResult | None
    pair_scores: PairScorePreparationResult | None

    @property
    def blocked_by_full_rebuild(self) -> bool:
        return self.feature_preparation.blocked_by_full_rebuild

    @property
    def change_set(self) -> GalleryChangeSet:
        return self.feature_preparation.change_set

    @property
    def candidate_pair_count(self) -> int:
        if self.candidate_selection is None:
            return 0
        return self.candidate_selection.unique_pair_count

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_preparation": self.feature_preparation.to_dict(),
            "candidate_selection": (
                None
                if self.candidate_selection is None
                else self.candidate_selection.to_dict()
            ),
            "pair_scores": None if self.pair_scores is None else self.pair_scores.to_dict(),
            "authoritative_state_modified": False,
            "derived_pair_cache_modified": bool(
                self.pair_scores is not None
                and (
                    self.pair_scores.cache_misses
                    + self.pair_scores.cache_stale_rebuilt
                    + self.pair_scores.cache_rescored
                )
                > 0
            ),
        }


def prepare_incremental_pairs(
    *,
    state_directory: Path,
    input_dir: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None = None,
    recursive: bool = False,
    candidate_time_window_seconds: float | None = None,
    max_seed_candidates_per_changed: int = 128,
    expand_previous_groups: bool = True,
) -> IncrementalPairPreparationResult:
    """Prepare changed features, select candidates, and cache pair scores.

    This is still a non-committing stage. The authoritative ``gallery_items``
    and final group tables remain unchanged. Only derived feature files and the
    SQLite ``pair_scores`` cache may be written.
    """

    feature_result = prepare_incremental_features(
        state_directory=state_directory,
        input_dir=input_dir,
        config=config,
        manifest_path=manifest_path,
        recursive=recursive,
    )
    if feature_result.blocked_by_full_rebuild:
        return IncrementalPairPreparationResult(
            feature_preparation=feature_result,
            candidate_selection=None,
            pair_scores=None,
        )

    state_root = Path(state_directory).expanduser().resolve()
    selection_config = CandidateSelectionConfig(
        time_window_seconds=(
            float(candidate_time_window_seconds)
            if candidate_time_window_seconds is not None
            else float(config.time_distance.fixed_horizon_seconds)
        ),
        max_seed_candidates_per_changed=int(max_seed_candidates_per_changed),
        expand_previous_groups=bool(expand_previous_groups),
    )

    try:
        store = GalleryStateStore(state_root / "state.sqlite3")
        cache = FeatureCache(state_root / "features")
        candidates = select_incremental_candidates(
            current_items=feature_result.snapshot.items,
            change_set=feature_result.change_set,
            previous_groups=store.groups(),
            feature_cache=cache,
            production_config=config,
            selection_config=selection_config,
        )
        scores = prepare_candidate_pair_scores(
            candidate_pairs=candidates.pairs,
            current_items_by_asset_id=feature_result.snapshot.items_by_asset_id(),
            feature_cache=cache,
            state_store=store,
            config=config,
        )
    except (
        CandidateSelectionError,
        PairScoreCacheError,
        FeatureCacheError,
        StateStoreError,
        ValueError,
    ) as exc:
        raise IncrementalProcessingError(str(exc)) from exc

    return IncrementalPairPreparationResult(
        feature_preparation=feature_result,
        candidate_selection=candidates,
        pair_scores=scores,
    )

@dataclass(frozen=True)
class IncrementalGroupPlanningResult:
    """Step-6/8 result: local group plan plus explicit safety assessment."""

    pair_preparation: IncrementalPairPreparationResult
    grouping_plan: IncrementalGroupingPlan | None
    safety: IncrementalSafetyAssessment | None

    @property
    def blocked_by_full_rebuild(self) -> bool:
        if self.pair_preparation.blocked_by_full_rebuild:
            return True
        return bool(self.safety is not None and self.safety.requires_full_rebuild)

    @property
    def blocked_incremental_commit(self) -> bool:
        if self.blocked_by_full_rebuild:
            return True
        return bool(self.safety is not None and not self.safety.allow_commit and not self.safety.is_noop)

    @property
    def change_set(self) -> GalleryChangeSet:
        return self.pair_preparation.change_set

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_preparation": self.pair_preparation.to_dict(),
            "grouping_plan": (
                None if self.grouping_plan is None else self.grouping_plan.to_dict()
            ),
            "safety": None if self.safety is None else self.safety.to_dict(),
            "authoritative_state_modified": False,
            "new_group_ids_are_temporary": True,
        }


def plan_incremental_groups(
    *,
    state_directory: Path,
    input_dir: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None = None,
    recursive: bool = False,
    candidate_time_window_seconds: float | None = None,
    max_seed_candidates_per_changed: int = 128,
    expand_previous_groups: bool = True,
    safety_policy: IncrementalSafetyPolicy | None = None,
) -> IncrementalGroupPlanningResult:
    """Prepare pair scores, build a local group plan, then assess safety.

    The function remains non-committing. It now also classifies the proposal as
    safe, no-op, retry-with-wider-candidates, or full-rebuild-required.
    """

    pair_result = prepare_incremental_pairs(
        state_directory=state_directory,
        input_dir=input_dir,
        config=config,
        manifest_path=manifest_path,
        recursive=recursive,
        candidate_time_window_seconds=candidate_time_window_seconds,
        max_seed_candidates_per_changed=max_seed_candidates_per_changed,
        expand_previous_groups=expand_previous_groups,
    )
    if pair_result.blocked_by_full_rebuild:
        return IncrementalGroupPlanningResult(
            pair_preparation=pair_result,
            grouping_plan=None,
            safety=None,
        )

    candidates = pair_result.candidate_selection
    scores = pair_result.pair_scores
    if candidates is None or scores is None:
        raise IncrementalProcessingError(
            "Pair preparation completed without candidate/score results."
        )

    state_root = Path(state_directory).expanduser().resolve()
    try:
        store = GalleryStateStore(state_root / "state.sqlite3")
        plan = build_incremental_grouping_plan(
            current_asset_ids=tuple(
                item.asset_id for item in pair_result.feature_preparation.snapshot.items
            ),
            previous_groups=store.groups(),
            change_set=pair_result.change_set,
            candidate_group_ids=candidates.candidate_group_ids,
            pair_scores=scores.scores,
            similarity_threshold=config.final_grouping.similarity_threshold,
            minimum_group_size=config.final_grouping.minimum_group_size,
        )
        safety = assess_incremental_safety(
            change_set=pair_result.change_set,
            candidate_selection=candidates,
            grouping_plan=plan,
            policy=safety_policy,
        )
    except (IncrementalGroupingError, StateStoreError, ValueError) as exc:
        raise IncrementalProcessingError(str(exc)) from exc

    return IncrementalGroupPlanningResult(
        pair_preparation=pair_result,
        grouping_plan=plan,
        safety=safety,
    )

@dataclass(frozen=True)
class IncrementalUpdateResult:
    """Complete incremental update result including Step-8 safety decision."""

    planning: IncrementalGroupPlanningResult
    commit: IncrementalCommitResult | None
    base_state_revision: int

    @property
    def blocked_by_full_rebuild(self) -> bool:
        return self.planning.blocked_by_full_rebuild

    @property
    def blocked_incremental_commit(self) -> bool:
        return self.planning.blocked_incremental_commit

    @property
    def no_changes(self) -> bool:
        return bool(self.planning.safety is not None and self.planning.safety.is_noop)

    @property
    def change_set(self) -> GalleryChangeSet:
        return self.planning.change_set

    def to_dict(self) -> dict[str, object]:
        return {
            "planning": self.planning.to_dict(),
            "commit": None if self.commit is None else self.commit.to_dict(),
            "base_state_revision": self.base_state_revision,
            "no_changes": self.no_changes,
            "blocked_incremental_commit": self.blocked_incremental_commit,
            "authoritative_state_modified": self.commit is not None,
        }


def commit_incremental_update(
    *,
    state_directory: Path,
    input_dir: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None = None,
    recursive: bool = False,
    candidate_time_window_seconds: float | None = None,
    max_seed_candidates_per_changed: int = 128,
    expand_previous_groups: bool = True,
    safety_policy: IncrementalSafetyPolicy | None = None,
) -> IncrementalUpdateResult:
    """Run the full incremental path, enforce safety, then commit atomically.

    No-op updates do not bump the authoritative state revision. Unsafe large/local
    updates are blocked before commit. Candidate truncation asks the caller to
    retry with a wider seed cap instead of silently committing incomplete local
    evidence.
    """

    config.validate()
    state_root = Path(state_directory).expanduser().resolve()
    database = state_root / "state.sqlite3"
    if not database.is_file():
        raise IncrementalProcessingError(
            f"State database does not exist: {database}. Run a state-enabled full build first."
        )

    try:
        store = GalleryStateStore(database)
        base_revision = store.state_revision()
    except StateStoreError as exc:
        raise IncrementalProcessingError(str(exc)) from exc

    planning = plan_incremental_groups(
        state_directory=state_root,
        input_dir=input_dir,
        config=config,
        manifest_path=manifest_path,
        recursive=recursive,
        candidate_time_window_seconds=candidate_time_window_seconds,
        max_seed_candidates_per_changed=max_seed_candidates_per_changed,
        expand_previous_groups=expand_previous_groups,
        safety_policy=safety_policy,
    )

    # Config incompatibility, a too-large update, or another hard rebuild rule.
    if planning.blocked_by_full_rebuild:
        return IncrementalUpdateResult(
            planning=planning,
            commit=None,
            base_state_revision=base_revision,
        )

    safety = planning.safety
    if safety is None:
        raise IncrementalProcessingError(
            "Incremental planning completed without a safety assessment."
        )

    # No changes should be a true no-op: do not rewrite SQLite and do not bump
    # state_revision merely because the client asked for another scan.
    if safety.is_noop:
        return IncrementalUpdateResult(
            planning=planning,
            commit=None,
            base_state_revision=base_revision,
        )

    # Candidate truncation is recoverable by widening the candidate cap. Other
    # safety decisions may also block commit without requiring a full rebuild.
    if not safety.allow_commit:
        return IncrementalUpdateResult(
            planning=planning,
            commit=None,
            base_state_revision=base_revision,
        )

    plan = planning.grouping_plan
    if plan is None:
        raise IncrementalProcessingError(
            "Incremental planning completed without a grouping plan."
        )

    snapshot = planning.pair_preparation.feature_preparation.snapshot
    cache = FeatureCache(state_root / "features")
    try:
        committed = commit_grouping_plan(
            state_store=store,
            feature_cache=cache,
            current_items=snapshot.items,
            change_set=planning.change_set,
            grouping_plan=plan,
            config=config,
            expected_state_revision=base_revision,
        )
    except IncrementalCommitError as exc:
        raise IncrementalProcessingError(str(exc)) from exc

    return IncrementalUpdateResult(
        planning=planning,
        commit=committed,
        base_state_revision=base_revision,
    )
