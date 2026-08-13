"""Conservative local group planning for incremental Smart Gallery updates.

The full production pipeline uses Average-Linkage agglomerative clustering after
its batch candidate gate. During everyday incremental updates we deliberately do
not rerun Affinity Propagation or rebuild the whole gallery. Instead this module
reclusters only the local region touched by Added/Modified/Deleted assets and by
their Step-5 candidate pairs.

Safety rules
------------
* unaffected previous groups are preserved exactly;
* surviving unchanged members of an affected previous group start as one
  anchored block, because nothing about their mutual relationship changed;
* Added/Modified assets and previously ungrouped candidate neighbours start as
  singleton blocks;
* blocks are merged using the same production similarity threshold and an
  average of all cross-block pair similarities;
* a missing candidate pair is treated as similarity 0.0 (conservative reject);
* two blocks anchored to different existing groups are never merged. This
  prevents one new bridge image from fusing two previously independent groups;
* groups smaller than ``minimum_group_size`` are emitted as ungrouped assets.

This module only creates a plan. It never writes authoritative SQLite gallery or
group state. Stable creation of new group IDs and the atomic commit are handled
by the following incremental commit stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from smart_gallery_similarity.change_detection import GalleryChangeSet
from smart_gallery_similarity.pair_score_cache import PreparedPairScore


class IncrementalGroupingError(RuntimeError):
    """Raised when a safe local grouping plan cannot be constructed."""


@dataclass(frozen=True)
class PlannedGroup:
    """One group in the proposed post-update gallery partition."""

    group_id: str
    status: str
    representative_asset_id: str
    member_asset_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.member_asset_ids)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["size"] = self.size
        return payload


@dataclass(frozen=True)
class IncrementalGroupingPlan:
    """Complete non-overlapping proposed grouping for the current gallery."""

    groups: tuple[PlannedGroup, ...]
    ungrouped_asset_ids: tuple[str, ...]
    affected_existing_group_ids: tuple[str, ...]
    dissolved_group_ids: tuple[str, ...]
    local_asset_ids: tuple[str, ...]
    known_pair_score_count: int
    local_merge_count: int
    blocked_existing_group_merge_count: int

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def grouped_image_count(self) -> int:
        return sum(group.size for group in self.groups)

    @property
    def preserved_group_count(self) -> int:
        return sum(group.status == "preserved" for group in self.groups)

    @property
    def updated_group_count(self) -> int:
        return sum(group.status == "updated" for group in self.groups)

    @property
    def new_group_count(self) -> int:
        return sum(group.status == "new" for group in self.groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_count": self.group_count,
            "grouped_image_count": self.grouped_image_count,
            "ungrouped_image_count": len(self.ungrouped_asset_ids),
            "preserved_group_count": self.preserved_group_count,
            "updated_group_count": self.updated_group_count,
            "new_group_count": self.new_group_count,
            "affected_existing_group_ids": list(self.affected_existing_group_ids),
            "dissolved_group_ids": list(self.dissolved_group_ids),
            "local_asset_ids": list(self.local_asset_ids),
            "known_pair_score_count": self.known_pair_score_count,
            "local_merge_count": self.local_merge_count,
            "blocked_existing_group_merge_count": self.blocked_existing_group_merge_count,
            "groups": [group.to_dict() for group in self.groups],
            "ungrouped_asset_ids": list(self.ungrouped_asset_ids),
        }


@dataclass
class _Block:
    members: set[str]
    anchor_group_ids: set[str]


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("Pair requires two different asset IDs.")
    return (left, right) if left < right else (right, left)


def _score_lookup(scores: Sequence[PreparedPairScore]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for score in scores:
        key = _canonical_pair(score.asset_id_a, score.asset_id_b)
        value = float(score.combined_similarity)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise IncrementalGroupingError(
                f"Invalid combined similarity for pair {key[0]} / {key[1]}."
            )
        previous = lookup.get(key)
        if previous is not None and not np.isclose(previous, value, atol=1e-8):
            raise IncrementalGroupingError(
                f"Conflicting pair scores for {key[0]} / {key[1]}."
            )
        lookup[key] = value
    return lookup


def _cross_block_average(
    left: _Block,
    right: _Block,
    *,
    lookup: Mapping[tuple[str, str], float],
) -> float:
    """Average cross-block similarity; unknown pairs are conservative zeros."""

    values: list[float] = []
    for asset_a in sorted(left.members):
        for asset_b in sorted(right.members):
            key = _canonical_pair(asset_a, asset_b)
            values.append(float(lookup.get(key, 0.0)))
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _block_key(block: _Block) -> tuple[str, ...]:
    return tuple(sorted(block.members))


def _choose_new_group_representative(
    members: Sequence[str],
    *,
    lookup: Mapping[tuple[str, str], float],
) -> str:
    ordered = tuple(sorted(members))
    if len(ordered) == 1:
        return ordered[0]

    best_asset = ordered[0]
    best_mean = -1.0
    for asset_id in ordered:
        similarities: list[float] = []
        for other in ordered:
            if other == asset_id:
                continue
            similarities.append(float(lookup.get(_canonical_pair(asset_id, other), 0.0)))
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        if mean_similarity > best_mean + 1e-12:
            best_mean = mean_similarity
            best_asset = asset_id
    return best_asset


def build_incremental_grouping_plan(
    *,
    current_asset_ids: Sequence[str],
    previous_groups: Sequence[Mapping[str, object]],
    change_set: GalleryChangeSet,
    candidate_group_ids: Sequence[str],
    pair_scores: Sequence[PreparedPairScore],
    similarity_threshold: float,
    minimum_group_size: int = 2,
) -> IncrementalGroupingPlan:
    """Build a conservative local Average-Linkage plan without committing state."""

    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold must be inside [0, 1].")
    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2.")
    if change_set.requires_full_rebuild:
        raise IncrementalGroupingError(
            "Local grouping cannot run while the change set requires a full rebuild."
        )

    current_ids = tuple(str(value) for value in current_asset_ids)
    if len(current_ids) != len(set(current_ids)):
        raise IncrementalGroupingError("Current gallery contains duplicate asset IDs.")
    current_set = set(current_ids)

    previous_by_id: dict[str, Mapping[str, object]] = {}
    previous_group_by_asset: dict[str, str] = {}
    for group in previous_groups:
        group_id = str(group.get("group_id", "")).strip()
        if not group_id or group_id in previous_by_id:
            raise IncrementalGroupingError("Previous groups contain invalid/duplicate IDs.")
        members = tuple(str(value) for value in group.get("member_asset_ids", []))
        if len(members) < 2 or len(members) != len(set(members)):
            raise IncrementalGroupingError(f"Previous group {group_id} is invalid.")
        previous_by_id[group_id] = group
        for member in members:
            if member in previous_group_by_asset:
                raise IncrementalGroupingError(
                    f"Asset {member} appears in multiple previous groups."
                )
            previous_group_by_asset[member] = group_id

    change_by_asset = {change.asset_id: change for change in change_set.changes}
    modified_ids = {change.asset_id for change in change_set.modified}
    deleted_ids = {change.asset_id for change in change_set.deleted}
    changed_current_ids = {
        change.asset_id
        for change in (*change_set.added, *change_set.modified)
        if change.asset_id in current_set
    }

    lookup = _score_lookup(pair_scores)
    pair_endpoint_ids = {asset_id for pair in lookup for asset_id in pair}
    unknown_pair_assets = pair_endpoint_ids - current_set
    if unknown_pair_assets:
        raise IncrementalGroupingError(
            "Pair scores reference assets missing from the current gallery: "
            + ", ".join(sorted(unknown_pair_assets))
        )

    affected_group_ids = set(str(value) for value in change_set.affected_group_ids)
    affected_group_ids.update(str(value) for value in candidate_group_ids)
    missing_affected = affected_group_ids - set(previous_by_id)
    if missing_affected:
        raise IncrementalGroupingError(
            "Candidate/affected group IDs are missing from previous state: "
            + ", ".join(sorted(missing_affected))
        )

    # Pair endpoints can reveal an old group even when a caller omitted it from
    # candidate_group_ids. Pull it in defensively rather than partially touching
    # a group.
    for asset_id in pair_endpoint_ids:
        group_id = previous_group_by_asset.get(asset_id)
        if group_id is not None:
            affected_group_ids.add(group_id)

    planned: list[PlannedGroup] = []
    grouped_assets: set[str] = set()

    # Preserve all previous groups outside the local region exactly, excluding a
    # member only if it is no longer current (which should have made the group
    # affected; this is also checked defensively).
    for group_id in sorted(previous_by_id):
        if group_id in affected_group_ids:
            continue
        group = previous_by_id[group_id]
        members = tuple(str(value) for value in group["member_asset_ids"])
        if any(member not in current_set for member in members):
            raise IncrementalGroupingError(
                f"Unaffected previous group {group_id} contains a missing current asset."
            )
        if any(member in modified_ids for member in members):
            raise IncrementalGroupingError(
                f"Unaffected previous group {group_id} contains a modified asset."
            )
        representative = str(group.get("representative_asset_id", ""))
        if representative not in members:
            representative = min(members)
        planned.append(
            PlannedGroup(
                group_id=group_id,
                status="preserved",
                representative_asset_id=representative,
                member_asset_ids=members,
                source_group_ids=(group_id,),
            )
        )
        grouped_assets.update(members)

    blocks: list[_Block] = []
    local_asset_ids: set[str] = set(changed_current_ids)
    local_asset_ids.update(pair_endpoint_ids)

    # Surviving unchanged members of every affected old group stay together as
    # an anchored block. Modified members are removed first and must earn their
    # way back through fresh pair scores.
    for group_id in sorted(affected_group_ids):
        group = previous_by_id[group_id]
        members = tuple(str(value) for value in group["member_asset_ids"])
        survivors = {
            member
            for member in members
            if member in current_set and member not in modified_ids and member not in deleted_ids
        }
        local_asset_ids.update(member for member in members if member in current_set)
        if survivors:
            blocks.append(_Block(members=set(survivors), anchor_group_ids={group_id}))

    already_blocked = {member for block in blocks for member in block.members}
    for asset_id in sorted(local_asset_ids - already_blocked):
        if asset_id in current_set:
            blocks.append(_Block(members={asset_id}, anchor_group_ids=set()))

    merge_count = 0
    blocked_existing_group_merges = 0
    threshold = float(similarity_threshold)

    # Deterministic constrained average-linkage agglomeration.
    while len(blocks) >= 2:
        best: tuple[float, tuple[str, ...], tuple[str, ...], int, int] | None = None
        blocked_this_round = 0
        for left_index in range(len(blocks) - 1):
            left = blocks[left_index]
            for right_index in range(left_index + 1, len(blocks)):
                right = blocks[right_index]
                if left.anchor_group_ids and right.anchor_group_ids:
                    if left.anchor_group_ids != right.anchor_group_ids:
                        blocked_this_round += 1
                        continue
                similarity = _cross_block_average(left, right, lookup=lookup)
                candidate = (
                    similarity,
                    _block_key(left),
                    _block_key(right),
                    left_index,
                    right_index,
                )
                if best is None:
                    best = candidate
                    continue
                if candidate[0] > best[0] + 1e-12:
                    best = candidate
                elif np.isclose(candidate[0], best[0], atol=1e-12):
                    if (candidate[1], candidate[2]) < (best[1], best[2]):
                        best = candidate

        blocked_existing_group_merges += blocked_this_round
        if best is None or best[0] + 1e-12 < threshold:
            break

        _, _, _, left_index, right_index = best
        left = blocks[left_index]
        right = blocks[right_index]
        merged = _Block(
            members=set(left.members) | set(right.members),
            anchor_group_ids=set(left.anchor_group_ids) | set(right.anchor_group_ids),
        )
        for index in sorted((left_index, right_index), reverse=True):
            blocks.pop(index)
        blocks.append(merged)
        blocks.sort(key=_block_key)
        merge_count += 1

    old_group_final_members: dict[str, tuple[str, ...]] = {}
    ungrouped_local: set[str] = set()
    new_blocks: list[_Block] = []

    for block in sorted(blocks, key=_block_key):
        members = tuple(sorted(block.members))
        if len(members) < minimum_group_size:
            ungrouped_local.update(members)
            continue
        if len(block.anchor_group_ids) == 1:
            group_id = next(iter(block.anchor_group_ids))
            old_group_final_members[group_id] = members
        elif not block.anchor_group_ids:
            new_blocks.append(block)
        else:
            raise IncrementalGroupingError(
                "Internal error: one final block contains multiple existing group anchors."
            )

    # Existing affected groups that retained >=2 members preserve their IDs.
    dissolved_group_ids: list[str] = []
    for group_id in sorted(affected_group_ids):
        previous = previous_by_id[group_id]
        previous_members = tuple(str(value) for value in previous["member_asset_ids"])
        final_members = old_group_final_members.get(group_id)
        if final_members is None:
            dissolved_group_ids.append(group_id)
            continue

        previous_rep = str(previous.get("representative_asset_id", ""))
        representative = (
            previous_rep if previous_rep in final_members else min(final_members)
        )
        status = (
            "preserved"
            if set(final_members) == set(previous_members)
            else "updated"
        )
        planned.append(
            PlannedGroup(
                group_id=group_id,
                status=status,
                representative_asset_id=representative,
                member_asset_ids=final_members,
                source_group_ids=(group_id,),
            )
        )
        grouped_assets.update(final_members)

    # New groups receive temporary IDs only. The next commit stage will allocate
    # permanent stable IDs atomically.
    for index, block in enumerate(sorted(new_blocks, key=_block_key), start=1):
        members = tuple(sorted(block.members))
        source_group_ids = tuple(
            sorted(
                {
                    str(change_by_asset[asset_id].previous_group_id)
                    for asset_id in members
                    if asset_id in change_by_asset
                    and change_by_asset[asset_id].previous_group_id is not None
                }
            )
        )
        representative = _choose_new_group_representative(members, lookup=lookup)
        planned.append(
            PlannedGroup(
                group_id=f"pending_new_{index:04d}",
                status="new",
                representative_asset_id=representative,
                member_asset_ids=members,
                source_group_ids=source_group_ids,
            )
        )
        grouped_assets.update(members)

    planned.sort(key=lambda group: (group.group_id.casefold(), group.member_asset_ids))

    # Everything current and not in a final >=2 group is ungrouped. This also
    # includes untouched previously-ungrouped images outside the local region.
    ungrouped = tuple(sorted(current_set - grouped_assets))
    if ungrouped_local - set(ungrouped):
        raise IncrementalGroupingError("Local singleton accounting is inconsistent.")

    seen: set[str] = set()
    for group in planned:
        if group.size < minimum_group_size:
            raise IncrementalGroupingError(f"Planned group {group.group_id} is too small.")
        for asset_id in group.member_asset_ids:
            if asset_id not in current_set:
                raise IncrementalGroupingError(
                    f"Planned group {group.group_id} contains a missing current asset."
                )
            if asset_id in seen:
                raise IncrementalGroupingError(
                    f"Asset {asset_id} appears in more than one planned group."
                )
            seen.add(asset_id)
    if seen | set(ungrouped) != current_set or seen & set(ungrouped):
        raise IncrementalGroupingError("Planned groups do not partition the current gallery.")

    return IncrementalGroupingPlan(
        groups=tuple(planned),
        ungrouped_asset_ids=ungrouped,
        affected_existing_group_ids=tuple(sorted(affected_group_ids)),
        dissolved_group_ids=tuple(dissolved_group_ids),
        local_asset_ids=tuple(sorted(local_asset_ids)),
        known_pair_score_count=len(lookup),
        local_merge_count=merge_count,
        blocked_existing_group_merge_count=blocked_existing_group_merges,
    )
