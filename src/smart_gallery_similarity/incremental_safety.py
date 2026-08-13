"""Safety and rebuild policy for incremental Smart Gallery updates.

Incremental PhotoCluster updates are intentionally local approximations of the
batch pipeline. This module decides when that approximation is still safe enough
to commit and when the caller should either widen the local candidate search or
fall back to a full authoritative rebuild.

The policy is deliberately separate from ``photocluster.production.json``.
These limits control *incremental orchestration*, not image similarity scores, so
changing them must not invalidate the frozen production pipeline fingerprint or
cached visual features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from smart_gallery_similarity.candidate_selection import CandidateSelectionResult
from smart_gallery_similarity.change_detection import GalleryChangeSet
from smart_gallery_similarity.incremental_grouping import IncrementalGroupingPlan


SafetyStatus = Literal[
    "safe",
    "no_changes",
    "retry_with_wider_candidates",
    "full_rebuild_required",
]


@dataclass(frozen=True)
class IncrementalSafetyPolicy:
    """Frozen defaults for deciding whether a local update may be committed.

    Fraction-based rules are ignored for very small galleries because one or two
    edits can otherwise look like an alarming percentage even though a local
    update is trivial and well covered by tests.
    """

    max_changed_fraction: float = 0.20
    max_changed_assets: int = 256
    max_deleted_fraction: float = 0.15
    max_deleted_assets: int = 128
    min_gallery_size_for_fraction_rules: int = 32
    max_local_assets: int = 512
    block_on_candidate_truncation: bool = True
    recommend_rebuild_after_bridge_block: bool = True

    def validate(self) -> None:
        if not 0.0 < self.max_changed_fraction <= 1.0:
            raise ValueError("max_changed_fraction must be in (0, 1].")
        if self.max_changed_assets < 1:
            raise ValueError("max_changed_assets must be at least 1.")
        if not 0.0 < self.max_deleted_fraction <= 1.0:
            raise ValueError("max_deleted_fraction must be in (0, 1].")
        if self.max_deleted_assets < 1:
            raise ValueError("max_deleted_assets must be at least 1.")
        if self.min_gallery_size_for_fraction_rules < 1:
            raise ValueError("min_gallery_size_for_fraction_rules must be at least 1.")
        if self.max_local_assets < 1:
            raise ValueError("max_local_assets must be at least 1.")


@dataclass(frozen=True)
class IncrementalSafetyAssessment:
    """Decision produced from one fully prepared local update plan."""

    status: SafetyStatus
    allow_commit: bool
    requires_full_rebuild: bool
    retry_with_wider_candidates: bool
    rebuild_recommended: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    changed_count: int
    change_fraction: float
    deleted_count: int
    deleted_fraction: float
    truncated_changed_asset_count: int
    local_asset_count: int
    blocked_existing_group_merge_count: int
    policy: IncrementalSafetyPolicy

    @property
    def is_noop(self) -> bool:
        return self.status == "no_changes"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "allow_commit": self.allow_commit,
            "requires_full_rebuild": self.requires_full_rebuild,
            "retry_with_wider_candidates": self.retry_with_wider_candidates,
            "rebuild_recommended": self.rebuild_recommended,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "changed_count": self.changed_count,
            "change_fraction": self.change_fraction,
            "deleted_count": self.deleted_count,
            "deleted_fraction": self.deleted_fraction,
            "truncated_changed_asset_count": self.truncated_changed_asset_count,
            "local_asset_count": self.local_asset_count,
            "blocked_existing_group_merge_count": self.blocked_existing_group_merge_count,
            "policy": asdict(self.policy),
        }


def assess_incremental_safety(
    *,
    change_set: GalleryChangeSet,
    candidate_selection: CandidateSelectionResult,
    grouping_plan: IncrementalGroupingPlan,
    policy: IncrementalSafetyPolicy | None = None,
) -> IncrementalSafetyAssessment:
    """Evaluate whether one prepared local update may be committed.

    Hard full-rebuild rules protect against updates that are no longer local.
    Candidate truncation is treated differently: the update is not committed,
    but the caller can retry with a larger ``--max-seed-candidates`` value before
    resorting to a full rebuild. A blocked bridge between two existing groups is
    safe because the grouping planner already prevents that merge; it merely
    recommends a future/full rebuild if exact batch reconciliation is desired.
    """

    effective = policy or IncrementalSafetyPolicy()
    effective.validate()

    changed_count = int(change_set.changed_count)
    change_fraction = float(change_set.change_fraction)
    deleted_count = len(change_set.deleted)
    deleted_fraction = (
        float(deleted_count / change_set.previous_count)
        if change_set.previous_count > 0
        else 0.0
    )
    truncated_count = int(candidate_selection.truncated_changed_asset_count)
    local_asset_count = len(grouping_plan.local_asset_ids)
    blocked_bridge_count = int(grouping_plan.blocked_existing_group_merge_count)

    if changed_count == 0:
        return IncrementalSafetyAssessment(
            status="no_changes",
            allow_commit=False,
            requires_full_rebuild=False,
            retry_with_wider_candidates=False,
            rebuild_recommended=False,
            reasons=("no_gallery_changes",),
            warnings=(),
            changed_count=0,
            change_fraction=0.0,
            deleted_count=0,
            deleted_fraction=0.0,
            truncated_changed_asset_count=truncated_count,
            local_asset_count=local_asset_count,
            blocked_existing_group_merge_count=blocked_bridge_count,
            policy=effective,
        )

    rebuild_reasons: list[str] = []
    warnings: list[str] = []

    # Fingerprint mismatches are already blocked earlier by change detection, but
    # keep the rule here too so this pure safety function remains self-contained.
    if change_set.requires_full_rebuild:
        rebuild_reasons.extend(change_set.rebuild_reasons)

    if changed_count > effective.max_changed_assets:
        rebuild_reasons.append("changed_asset_limit_exceeded")

    gallery_scale = max(change_set.previous_count, change_set.current_count)
    if (
        gallery_scale >= effective.min_gallery_size_for_fraction_rules
        and change_fraction > effective.max_changed_fraction
    ):
        rebuild_reasons.append("changed_fraction_exceeded")

    if deleted_count > effective.max_deleted_assets:
        rebuild_reasons.append("deleted_asset_limit_exceeded")

    if (
        change_set.previous_count >= effective.min_gallery_size_for_fraction_rules
        and deleted_fraction > effective.max_deleted_fraction
    ):
        rebuild_reasons.append("deleted_fraction_exceeded")

    if local_asset_count > effective.max_local_assets:
        rebuild_reasons.append("local_scope_limit_exceeded")

    if rebuild_reasons:
        return IncrementalSafetyAssessment(
            status="full_rebuild_required",
            allow_commit=False,
            requires_full_rebuild=True,
            retry_with_wider_candidates=False,
            rebuild_recommended=True,
            reasons=tuple(dict.fromkeys(rebuild_reasons)),
            warnings=tuple(warnings),
            changed_count=changed_count,
            change_fraction=change_fraction,
            deleted_count=deleted_count,
            deleted_fraction=deleted_fraction,
            truncated_changed_asset_count=truncated_count,
            local_asset_count=local_asset_count,
            blocked_existing_group_merge_count=blocked_bridge_count,
            policy=effective,
        )

    if effective.block_on_candidate_truncation and truncated_count > 0:
        return IncrementalSafetyAssessment(
            status="retry_with_wider_candidates",
            allow_commit=False,
            requires_full_rebuild=False,
            retry_with_wider_candidates=True,
            rebuild_recommended=False,
            reasons=("candidate_seed_truncated",),
            warnings=(
                "At least one changed asset hit the candidate seed cap; rerun with "
                "a larger --max-seed-candidates value before committing.",
            ),
            changed_count=changed_count,
            change_fraction=change_fraction,
            deleted_count=deleted_count,
            deleted_fraction=deleted_fraction,
            truncated_changed_asset_count=truncated_count,
            local_asset_count=local_asset_count,
            blocked_existing_group_merge_count=blocked_bridge_count,
            policy=effective,
        )

    rebuild_recommended = False
    if effective.recommend_rebuild_after_bridge_block and blocked_bridge_count > 0:
        rebuild_recommended = True
        warnings.append(
            "One or more attempted merges between different existing groups were "
            "blocked by the bridge-safety rule. The incremental result is safe and "
            "conservative, but a future full rebuild may reconcile those groups."
        )

    return IncrementalSafetyAssessment(
        status="safe",
        allow_commit=True,
        requires_full_rebuild=False,
        retry_with_wider_candidates=False,
        rebuild_recommended=rebuild_recommended,
        reasons=(),
        warnings=tuple(warnings),
        changed_count=changed_count,
        change_fraction=change_fraction,
        deleted_count=deleted_count,
        deleted_fraction=deleted_fraction,
        truncated_changed_asset_count=truncated_count,
        local_asset_count=local_asset_count,
        blocked_existing_group_merge_count=blocked_bridge_count,
        policy=effective,
    )
