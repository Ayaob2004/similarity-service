"""Pure gallery change detection for incremental processing.

This module compares the last authoritative SQLite snapshot with a newly observed
set of gallery items. It does **not** mutate state and it does **not** run image
similarity. Its only job is to classify stable ``asset_id`` values as:

- ``added``: present now, absent from the saved snapshot;
- ``modified``: same asset_id, but algorithm-relevant content/metadata changed;
- ``deleted``: present in the saved snapshot, absent now;
- ``unchanged``: algorithm-relevant fields are unchanged.

Filename/path/source-only differences are reported as metadata updates but are
kept in the ``unchanged`` bucket because they do not change PhotoCluster scores
when a stable asset_id is available. In local-folder mode a rename normally
changes the generated asset_id, so it naturally appears as delete + add.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence


ChangeStatus = Literal["added", "modified", "deleted", "unchanged"]


@dataclass(frozen=True)
class CurrentGalleryItem:
    """Current metadata needed to compare one asset with persistent state."""

    asset_id: str
    filename: str
    relative_path: str
    capture_time: str
    capture_time_source: str
    width: int
    height: int
    content_sha256: str


@dataclass(frozen=True)
class GalleryChange:
    """One classified asset change."""

    asset_id: str
    status: ChangeStatus
    reasons: tuple[str, ...] = ()
    previous_group_id: str | None = None
    metadata_changed: bool = False


@dataclass(frozen=True)
class GalleryChangeSet:
    """Complete read-only comparison between saved and current gallery state."""

    changes: tuple[GalleryChange, ...]
    previous_count: int
    current_count: int
    affected_group_ids: tuple[str, ...]
    current_pipeline_fingerprint: str
    saved_pipeline_fingerprint: str
    current_feature_fingerprint: str
    saved_feature_fingerprint: str
    requires_full_rebuild: bool
    rebuild_reasons: tuple[str, ...]

    def by_status(self, status: ChangeStatus) -> tuple[GalleryChange, ...]:
        return tuple(change for change in self.changes if change.status == status)

    @property
    def added(self) -> tuple[GalleryChange, ...]:
        return self.by_status("added")

    @property
    def modified(self) -> tuple[GalleryChange, ...]:
        return self.by_status("modified")

    @property
    def deleted(self) -> tuple[GalleryChange, ...]:
        return self.by_status("deleted")

    @property
    def unchanged(self) -> tuple[GalleryChange, ...]:
        return self.by_status("unchanged")

    @property
    def metadata_update_count(self) -> int:
        return sum(change.metadata_changed for change in self.unchanged)

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    @property
    def union_count(self) -> int:
        return len(self.changes)

    @property
    def change_fraction(self) -> float:
        if self.union_count == 0:
            return 0.0
        return float(self.changed_count / self.union_count)

    @property
    def reprocess_asset_ids(self) -> tuple[str, ...]:
        """Assets that need fresh incremental image processing later."""

        return tuple(
            change.asset_id
            for change in self.changes
            if change.status in {"added", "modified"}
        )

    @property
    def removed_asset_ids(self) -> tuple[str, ...]:
        return tuple(change.asset_id for change in self.deleted)

    def to_dict(self) -> dict[str, object]:
        counts = {
            "added": len(self.added),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
            "metadata_updates_within_unchanged": self.metadata_update_count,
        }
        return {
            "previous_count": self.previous_count,
            "current_count": self.current_count,
            "counts": counts,
            "changed_count": self.changed_count,
            "change_fraction": self.change_fraction,
            "affected_group_ids": list(self.affected_group_ids),
            "reprocess_asset_ids": list(self.reprocess_asset_ids),
            "removed_asset_ids": list(self.removed_asset_ids),
            "fingerprints": {
                "saved_pipeline": self.saved_pipeline_fingerprint,
                "current_pipeline": self.current_pipeline_fingerprint,
                "saved_features": self.saved_feature_fingerprint,
                "current_features": self.current_feature_fingerprint,
            },
            "requires_full_rebuild": self.requires_full_rebuild,
            "rebuild_reasons": list(self.rebuild_reasons),
            "changes": [asdict(change) for change in self.changes],
        }


def _normalize_previous_items(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("Saved state contains an empty asset_id.")
        if asset_id in result:
            raise ValueError(f"Saved state contains duplicate asset_id={asset_id!r}.")
        result[asset_id] = row
    return result


def _normalize_current_items(
    items: Sequence[CurrentGalleryItem],
) -> dict[str, CurrentGalleryItem]:
    result: dict[str, CurrentGalleryItem] = {}
    for item in items:
        asset_id = str(item.asset_id).strip()
        if not asset_id:
            raise ValueError("Current gallery contains an empty asset_id.")
        if asset_id in result:
            raise ValueError(f"Current gallery contains duplicate asset_id={asset_id!r}.")
        if len(item.content_sha256) != 64:
            raise ValueError(
                f"Current gallery has an invalid SHA-256 for asset_id={asset_id!r}."
            )
        result[asset_id] = item
    return result


def detect_gallery_changes(
    *,
    previous_items: Sequence[Mapping[str, object]],
    current_items: Sequence[CurrentGalleryItem],
    previous_group_by_asset_id: Mapping[str, str] | None = None,
    saved_pipeline_fingerprint: str = "",
    current_pipeline_fingerprint: str = "",
    saved_feature_fingerprint: str = "",
    current_feature_fingerprint: str = "",
) -> GalleryChangeSet:
    """Classify current gallery against the last saved authoritative snapshot.

    ``modified`` is intentionally limited to fields that can change similarity
    behavior or cached image content: file bytes, capture time, and dimensions.
    Path/name/source-only changes are tracked as metadata updates without forcing
    image feature extraction or local reclustering.

    A configuration fingerprint mismatch does not silently relabel every asset as
    modified. Instead the result explicitly requires a full rebuild, because a
    global algorithm setting change can invalidate pair/group decisions beyond
    the assets that changed on disk.
    """

    previous = _normalize_previous_items(previous_items)
    current = _normalize_current_items(current_items)
    group_map = dict(previous_group_by_asset_id or {})

    rebuild_reasons: list[str] = []
    if (
        saved_pipeline_fingerprint
        and current_pipeline_fingerprint
        and saved_pipeline_fingerprint != current_pipeline_fingerprint
    ):
        rebuild_reasons.append("pipeline_fingerprint_changed")
    if (
        saved_feature_fingerprint
        and current_feature_fingerprint
        and saved_feature_fingerprint != current_feature_fingerprint
    ):
        rebuild_reasons.append("feature_fingerprint_changed")

    changes: list[GalleryChange] = []
    affected_groups: set[str] = set()

    all_asset_ids = sorted(set(previous) | set(current), key=str.casefold)
    for asset_id in all_asset_ids:
        old = previous.get(asset_id)
        new = current.get(asset_id)
        old_group = group_map.get(asset_id)

        if old is None and new is not None:
            changes.append(GalleryChange(asset_id=asset_id, status="added"))
            continue

        if old is not None and new is None:
            changes.append(
                GalleryChange(
                    asset_id=asset_id,
                    status="deleted",
                    reasons=("asset_missing",),
                    previous_group_id=old_group,
                )
            )
            if old_group:
                affected_groups.add(old_group)
            continue

        assert old is not None and new is not None

        algorithm_reasons: list[str] = []
        if str(old.get("content_sha256", "")) != new.content_sha256:
            algorithm_reasons.append("content_sha256")
        if str(old.get("capture_time", "")) != new.capture_time:
            algorithm_reasons.append("capture_time")
        if int(old.get("width", 0)) != int(new.width) or int(old.get("height", 0)) != int(
            new.height
        ):
            algorithm_reasons.append("dimensions")

        metadata_changed = any(
            (
                str(old.get("filename", "")) != new.filename,
                str(old.get("relative_path", "")) != new.relative_path,
                str(old.get("capture_time_source", "")) != new.capture_time_source,
            )
        )

        if algorithm_reasons:
            changes.append(
                GalleryChange(
                    asset_id=asset_id,
                    status="modified",
                    reasons=tuple(algorithm_reasons),
                    previous_group_id=old_group,
                    metadata_changed=metadata_changed,
                )
            )
            if old_group:
                affected_groups.add(old_group)
        else:
            changes.append(
                GalleryChange(
                    asset_id=asset_id,
                    status="unchanged",
                    previous_group_id=old_group,
                    metadata_changed=metadata_changed,
                )
            )

    return GalleryChangeSet(
        changes=tuple(changes),
        previous_count=len(previous),
        current_count=len(current),
        affected_group_ids=tuple(sorted(affected_groups, key=str.casefold)),
        current_pipeline_fingerprint=current_pipeline_fingerprint,
        saved_pipeline_fingerprint=saved_pipeline_fingerprint,
        current_feature_fingerprint=current_feature_fingerprint,
        saved_feature_fingerprint=saved_feature_fingerprint,
        requires_full_rebuild=bool(rebuild_reasons),
        rebuild_reasons=tuple(rebuild_reasons),
    )
