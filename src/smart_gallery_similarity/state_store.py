"""SQLite-backed persistent gallery state used by incremental processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Sequence


class StateStoreError(RuntimeError):
    """Raised when gallery state cannot be read or updated safely."""


@dataclass(frozen=True)
class GalleryStateItem:
    asset_id: str
    filename: str
    relative_path: str
    capture_time: str
    capture_time_source: str
    width: int
    height: int
    content_sha256: str
    feature_cache_key: str
    feature_fingerprint: str


@dataclass(frozen=True)
class GalleryStateGroup:
    group_id: str
    representative_asset_id: str
    member_asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairScoreRecord:
    """One derived pair score cached independently of group membership."""

    asset_id_a: str
    asset_id_b: str
    content_sha256_a: str
    content_sha256_b: str
    capture_time_a: str
    capture_time_b: str
    component_fingerprint: str
    score_fingerprint: str
    keypoint_count_a: int
    keypoint_count_b: int
    match_count: int
    smr: float
    color_distance: float
    time_distance: float
    combined_similarity: float


@dataclass(frozen=True)
class IncrementalStateCommitResult:
    """Database-side result of one atomic incremental snapshot commit."""

    state_revision: int
    group_id_assignments: tuple[tuple[str, str], ...]
    removed_orphan_pair_score_count: int
    image_count: int
    group_count: int
    grouped_image_count: int

    @property
    def ungrouped_image_count(self) -> int:
        return self.image_count - self.grouped_image_count


class GalleryStateStore:
    """Small transactional state database; source image bytes are never stored."""

    SCHEMA_VERSION = 2

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS gallery_items (
                        asset_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        capture_time TEXT NOT NULL,
                        capture_time_source TEXT NOT NULL,
                        width INTEGER NOT NULL CHECK(width > 0),
                        height INTEGER NOT NULL CHECK(height > 0),
                        content_sha256 TEXT NOT NULL,
                        feature_cache_key TEXT NOT NULL,
                        feature_fingerprint TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS groups (
                        group_id TEXT PRIMARY KEY,
                        representative_asset_id TEXT NOT NULL,
                        member_count INTEGER NOT NULL CHECK(member_count >= 2)
                    );

                    CREATE TABLE IF NOT EXISTS group_members (
                        group_id TEXT NOT NULL,
                        asset_id TEXT NOT NULL UNIQUE,
                        member_position INTEGER NOT NULL CHECK(member_position >= 0),
                        PRIMARY KEY(group_id, asset_id),
                        FOREIGN KEY(group_id) REFERENCES groups(group_id) ON DELETE CASCADE,
                        FOREIGN KEY(asset_id) REFERENCES gallery_items(asset_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_gallery_items_capture_time
                    ON gallery_items(capture_time);

                    CREATE INDEX IF NOT EXISTS idx_group_members_group
                    ON group_members(group_id, member_position);

                    CREATE TABLE IF NOT EXISTS pair_scores (
                        asset_id_a TEXT NOT NULL,
                        asset_id_b TEXT NOT NULL,
                        content_sha256_a TEXT NOT NULL,
                        content_sha256_b TEXT NOT NULL,
                        capture_time_a TEXT NOT NULL,
                        capture_time_b TEXT NOT NULL,
                        component_fingerprint TEXT NOT NULL,
                        score_fingerprint TEXT NOT NULL,
                        keypoint_count_a INTEGER NOT NULL CHECK(keypoint_count_a >= 0),
                        keypoint_count_b INTEGER NOT NULL CHECK(keypoint_count_b >= 0),
                        match_count INTEGER NOT NULL CHECK(match_count >= 0),
                        smr REAL NOT NULL CHECK(smr >= 0.0 AND smr <= 1.0),
                        color_distance REAL NOT NULL CHECK(color_distance >= 0.0 AND color_distance <= 1.0),
                        time_distance REAL NOT NULL CHECK(time_distance >= 0.0 AND time_distance <= 1.0),
                        combined_similarity REAL NOT NULL CHECK(combined_similarity >= 0.0 AND combined_similarity <= 1.0),
                        updated_utc TEXT NOT NULL,
                        PRIMARY KEY(asset_id_a, asset_id_b),
                        CHECK(asset_id_a < asset_id_b)
                    );

                    CREATE INDEX IF NOT EXISTS idx_pair_scores_a
                    ON pair_scores(asset_id_a);

                    CREATE INDEX IF NOT EXISTS idx_pair_scores_b
                    ON pair_scores(asset_id_b);
                    """
                )
                connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION)),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(
                f"Could not initialize gallery state '{self.database_path}': {exc}"
            ) from exc

    @staticmethod
    def _validate_snapshot(
        *,
        items: Sequence[GalleryStateItem],
        groups: Sequence[GalleryStateGroup],
    ) -> tuple[set[str], int]:
        """Validate a complete non-overlapping gallery/group snapshot."""

        item_ids = [item.asset_id for item in items]
        if len(item_ids) != len(set(item_ids)):
            raise StateStoreError("State snapshot contains duplicate asset_id values.")
        known_ids = set(item_ids)

        group_ids = [str(group.group_id) for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise StateStoreError("State snapshot contains duplicate group_id values.")

        seen_members: set[str] = set()
        for group in groups:
            if not str(group.group_id).strip():
                raise StateStoreError("State snapshot contains an empty group_id.")
            if len(group.member_asset_ids) < 2:
                raise StateStoreError(f"Group {group.group_id} has fewer than two members.")
            if group.representative_asset_id not in group.member_asset_ids:
                raise StateStoreError(
                    f"Representative is not a member of group {group.group_id}."
                )
            if len(group.member_asset_ids) != len(set(group.member_asset_ids)):
                raise StateStoreError(f"Group {group.group_id} contains duplicate members.")
            for asset_id in group.member_asset_ids:
                if asset_id not in known_ids:
                    raise StateStoreError(
                        f"Group {group.group_id} references unknown asset_id={asset_id}."
                    )
                if asset_id in seen_members:
                    raise StateStoreError(
                        f"asset_id={asset_id} appears in more than one final group."
                    )
                seen_members.add(asset_id)
        return known_ids, len(seen_members)

    @staticmethod
    def _numeric_group_sequence(group_id: str) -> int | None:
        match = re.fullmatch(r"group_(\d+)", str(group_id))
        return None if match is None else int(match.group(1))

    def state_revision(self) -> int:
        """Return the optimistic-concurrency revision of authoritative state."""

        metadata = self.metadata()
        raw = metadata.get("state_revision")
        if raw is None:
            # Databases created before Incremental Step 7 already contain a valid
            # full-build baseline. Treat that historical baseline as revision 1.
            return 1 if metadata.get("last_full_build_utc") else 0
        try:
            value = int(raw)
        except ValueError as exc:
            raise StateStoreError("Invalid state_revision metadata.") from exc
        if value < 0:
            raise StateStoreError("state_revision cannot be negative.")
        return value

    def replace_full_snapshot(
        self,
        *,
        items: Sequence[GalleryStateItem],
        groups: Sequence[GalleryStateGroup],
        pipeline_fingerprint: str,
        feature_fingerprint: str,
    ) -> None:
        """Atomically replace state after a successful authoritative full build."""

        self._validate_snapshot(items=items, groups=groups)
        current_revision = self.state_revision()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM group_members")
                connection.execute("DELETE FROM groups")
                connection.execute("DELETE FROM gallery_items")
                # A full build is a new authoritative baseline. Derived pair
                # scores from an older incremental lineage are cheap to rebuild
                # lazily and should not survive across baselines.
                connection.execute("DELETE FROM pair_scores")

                connection.executemany(
                    """
                    INSERT INTO gallery_items(
                        asset_id, filename, relative_path, capture_time,
                        capture_time_source, width, height, content_sha256,
                        feature_cache_key, feature_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.asset_id,
                            item.filename,
                            item.relative_path,
                            item.capture_time,
                            item.capture_time_source,
                            int(item.width),
                            int(item.height),
                            item.content_sha256,
                            item.feature_cache_key,
                            item.feature_fingerprint,
                        )
                        for item in items
                    ],
                )

                for group in groups:
                    connection.execute(
                        """
                        INSERT INTO groups(group_id, representative_asset_id, member_count)
                        VALUES (?, ?, ?)
                        """,
                        (
                            group.group_id,
                            group.representative_asset_id,
                            len(group.member_asset_ids),
                        ),
                    )
                    connection.executemany(
                        """
                        INSERT INTO group_members(group_id, asset_id, member_position)
                        VALUES (?, ?, ?)
                        """,
                        [
                            (group.group_id, asset_id, position)
                            for position, asset_id in enumerate(group.member_asset_ids)
                        ],
                    )

                metadata = {
                    "schema_version": str(self.SCHEMA_VERSION),
                    "pipeline_fingerprint": str(pipeline_fingerprint),
                    "feature_fingerprint": str(feature_fingerprint),
                    "last_full_build_utc": now,
                    "last_incremental_update_utc": "",
                    "state_revision": str(current_revision + 1),
                    "next_group_sequence": str(
                        max(
                            [
                                sequence
                                for sequence in (
                                    self._numeric_group_sequence(group.group_id)
                                    for group in groups
                                )
                                if sequence is not None
                            ],
                            default=0,
                        )
                        + 1
                    ),
                    "image_count": str(len(items)),
                    "group_count": str(len(groups)),
                }
                connection.executemany(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    list(metadata.items()),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not save gallery state: {exc}") from exc

    def commit_incremental_snapshot(
        self,
        *,
        items: Sequence[GalleryStateItem],
        groups: Sequence[GalleryStateGroup],
        pipeline_fingerprint: str,
        feature_fingerprint: str,
        expected_state_revision: int,
        temporary_group_prefix: str = "pending_new_",
    ) -> IncrementalStateCommitResult:
        """Atomically commit one incremental gallery/group snapshot.

        Existing stable group IDs are preserved. Temporary group IDs generated by
        local planning are replaced with monotonically increasing ``group_NNNN``
        IDs inside the same SQLite transaction. The sequence is never reused, even
        when old groups dissolve. Derived pair scores are preserved, except rows
        that reference assets no longer present in the committed gallery.
        """

        self._validate_snapshot(items=items, groups=groups)
        if int(expected_state_revision) < 0:
            raise ValueError("expected_state_revision cannot be negative.")

        temporary_ids = [
            group.group_id
            for group in groups
            if str(group.group_id).startswith(temporary_group_prefix)
        ]
        if len(temporary_ids) != len(set(temporary_ids)):
            raise StateStoreError("Incremental snapshot contains duplicate temporary group IDs.")

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                metadata_rows = connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
                metadata = {str(row["key"]): str(row["value"]) for row in metadata_rows}
                raw_revision = metadata.get("state_revision")
                if raw_revision is None:
                    current_revision = 1 if metadata.get("last_full_build_utc") else 0
                else:
                    current_revision = int(raw_revision)
                if current_revision != int(expected_state_revision):
                    raise StateStoreError(
                        "Authoritative state changed while the incremental update was "
                        f"being prepared (expected revision {expected_state_revision}, "
                        f"found {current_revision}). Re-run the incremental update."
                    )

                existing_group_ids = {
                    str(row[0])
                    for row in connection.execute("SELECT group_id FROM groups").fetchall()
                }
                stable_input_ids = {
                    str(group.group_id)
                    for group in groups
                    if not str(group.group_id).startswith(temporary_group_prefix)
                }
                unknown_stable_ids = stable_input_ids - existing_group_ids
                if unknown_stable_ids:
                    raise StateStoreError(
                        "Incremental plan contains unknown stable group IDs: "
                        + ", ".join(sorted(unknown_stable_ids))
                    )

                numeric_existing = [
                    sequence
                    for sequence in (
                        self._numeric_group_sequence(group_id)
                        for group_id in existing_group_ids
                    )
                    if sequence is not None
                ]
                try:
                    metadata_next = int(metadata.get("next_group_sequence", "0") or "0")
                except ValueError as exc:
                    raise StateStoreError("Invalid next_group_sequence metadata.") from exc
                next_sequence = max(metadata_next, max(numeric_existing, default=0) + 1, 1)

                assignments: dict[str, str] = {}
                reserved = set(existing_group_ids) | stable_input_ids
                for temporary_id in sorted(temporary_ids, key=str.casefold):
                    while True:
                        stable_id = f"group_{next_sequence:04d}"
                        next_sequence += 1
                        if stable_id not in reserved:
                            break
                    assignments[temporary_id] = stable_id
                    reserved.add(stable_id)

                resolved_groups = [
                    GalleryStateGroup(
                        group_id=assignments.get(group.group_id, group.group_id),
                        representative_asset_id=group.representative_asset_id,
                        member_asset_ids=tuple(group.member_asset_ids),
                    )
                    for group in groups
                ]
                self._validate_snapshot(items=items, groups=resolved_groups)

                connection.execute("DELETE FROM group_members")
                connection.execute("DELETE FROM groups")
                connection.execute("DELETE FROM gallery_items")

                connection.executemany(
                    """
                    INSERT INTO gallery_items(
                        asset_id, filename, relative_path, capture_time,
                        capture_time_source, width, height, content_sha256,
                        feature_cache_key, feature_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.asset_id, item.filename, item.relative_path,
                            item.capture_time, item.capture_time_source, int(item.width),
                            int(item.height), item.content_sha256, item.feature_cache_key,
                            item.feature_fingerprint,
                        )
                        for item in items
                    ],
                )

                for group in resolved_groups:
                    connection.execute(
                        """
                        INSERT INTO groups(group_id, representative_asset_id, member_count)
                        VALUES (?, ?, ?)
                        """,
                        (group.group_id, group.representative_asset_id, len(group.member_asset_ids)),
                    )
                    connection.executemany(
                        """
                        INSERT INTO group_members(group_id, asset_id, member_position)
                        VALUES (?, ?, ?)
                        """,
                        [
                            (group.group_id, asset_id, position)
                            for position, asset_id in enumerate(group.member_asset_ids)
                        ],
                    )

                before_pair_count = int(
                    connection.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]
                )
                connection.execute(
                    """
                    DELETE FROM pair_scores
                    WHERE NOT EXISTS (
                        SELECT 1 FROM gallery_items AS a
                        WHERE a.asset_id = pair_scores.asset_id_a
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM gallery_items AS b
                        WHERE b.asset_id = pair_scores.asset_id_b
                    )
                    """
                )
                after_pair_count = int(
                    connection.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]
                )

                new_revision = current_revision + 1
                committed_metadata = {
                    "schema_version": str(self.SCHEMA_VERSION),
                    "pipeline_fingerprint": str(pipeline_fingerprint),
                    "feature_fingerprint": str(feature_fingerprint),
                    "last_incremental_update_utc": now,
                    "state_revision": str(new_revision),
                    "next_group_sequence": str(next_sequence),
                    "image_count": str(len(items)),
                    "group_count": str(len(resolved_groups)),
                }
                connection.executemany(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    list(committed_metadata.items()),
                )

                grouped_count = sum(len(group.member_asset_ids) for group in resolved_groups)
                return IncrementalStateCommitResult(
                    state_revision=new_revision,
                    group_id_assignments=tuple(sorted(assignments.items())),
                    removed_orphan_pair_score_count=before_pair_count - after_pair_count,
                    image_count=len(items),
                    group_count=len(resolved_groups),
                    grouped_image_count=grouped_count,
                )
        except StateStoreError:
            raise
        except (sqlite3.Error, ValueError) as exc:
            raise StateStoreError(f"Could not commit incremental gallery state: {exc}") from exc

    def metadata(self) -> dict[str, str]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT key, value FROM metadata ORDER BY key"
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not read gallery metadata: {exc}") from exc
        return {str(row["key"]): str(row["value"]) for row in rows}

    def items(self) -> list[dict[str, object]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT asset_id, filename, relative_path, capture_time,
                           capture_time_source, width, height, content_sha256,
                           feature_cache_key, feature_fingerprint
                    FROM gallery_items
                    ORDER BY relative_path COLLATE NOCASE, asset_id
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not read gallery items: {exc}") from exc
        return [dict(row) for row in rows]

    def items_by_asset_id(self) -> dict[str, dict[str, object]]:
        """Return the saved gallery snapshot keyed by stable asset_id."""

        return {str(row["asset_id"]): row for row in self.items()}

    def group_by_asset_id(self) -> dict[str, str]:
        """Return previous final-group membership for grouped assets only."""

        result: dict[str, str] = {}
        for group in self.groups():
            group_id = str(group["group_id"])
            for asset_id in group["member_asset_ids"]:
                key = str(asset_id)
                if key in result:
                    raise StateStoreError(
                        f"asset_id={key} appears in more than one saved group."
                    )
                result[key] = group_id
        return result

    def has_authoritative_snapshot(self) -> bool:
        """Whether a successful full build has committed a usable baseline."""

        metadata = self.metadata()
        return bool(metadata.get("last_full_build_utc"))

    def groups(self) -> list[dict[str, object]]:
        try:
            with self._connect() as connection:
                group_rows = connection.execute(
                    """
                    SELECT group_id, representative_asset_id, member_count
                    FROM groups
                    ORDER BY group_id
                    """
                ).fetchall()
                member_rows = connection.execute(
                    """
                    SELECT group_id, asset_id, member_position
                    FROM group_members
                    ORDER BY group_id, member_position
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not read groups: {exc}") from exc

        members: dict[str, list[str]] = {}
        for row in member_rows:
            members.setdefault(str(row["group_id"]), []).append(str(row["asset_id"]))
        return [
            {
                "group_id": str(row["group_id"]),
                "representative_asset_id": str(row["representative_asset_id"]),
                "member_count": int(row["member_count"]),
                "member_asset_ids": members.get(str(row["group_id"]), []),
            }
            for row in group_rows
        ]

    @staticmethod
    def _canonical_pair(asset_id_a: str, asset_id_b: str) -> tuple[str, str, bool]:
        left = str(asset_id_a).strip()
        right = str(asset_id_b).strip()
        if not left or not right:
            raise ValueError("Pair asset IDs cannot be empty.")
        if left == right:
            raise ValueError("Pair score requires two different asset IDs.")
        if left < right:
            return left, right, False
        return right, left, True

    def pair_score(self, asset_id_a: str, asset_id_b: str) -> dict[str, object] | None:
        """Return one cached pair score in canonical asset-ID order."""

        left, right, _ = self._canonical_pair(asset_id_a, asset_id_b)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT asset_id_a, asset_id_b, content_sha256_a, content_sha256_b,
                           capture_time_a, capture_time_b, component_fingerprint,
                           score_fingerprint, keypoint_count_a, keypoint_count_b,
                           match_count, smr, color_distance, time_distance,
                           combined_similarity, updated_utc
                    FROM pair_scores
                    WHERE asset_id_a = ? AND asset_id_b = ?
                    """,
                    (left, right),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not read pair score: {exc}") from exc
        return None if row is None else dict(row)

    def upsert_pair_score(self, record: PairScoreRecord) -> None:
        """Insert or atomically replace one derived pair-score cache entry."""

        left, right, swapped = self._canonical_pair(record.asset_id_a, record.asset_id_b)
        if swapped:
            content_a, content_b = record.content_sha256_b, record.content_sha256_a
            time_a, time_b = record.capture_time_b, record.capture_time_a
            keypoints_a, keypoints_b = record.keypoint_count_b, record.keypoint_count_a
        else:
            content_a, content_b = record.content_sha256_a, record.content_sha256_b
            time_a, time_b = record.capture_time_a, record.capture_time_b
            keypoints_a, keypoints_b = record.keypoint_count_a, record.keypoint_count_b

        numeric = (
            float(record.smr),
            float(record.color_distance),
            float(record.time_distance),
            float(record.combined_similarity),
        )
        if any(value < 0.0 or value > 1.0 for value in numeric):
            raise StateStoreError("Pair score values must be inside [0, 1].")
        if record.match_count < 0 or keypoints_a < 0 or keypoints_b < 0:
            raise StateStoreError("Pair keypoint/match counts cannot be negative.")
        if record.match_count > min(keypoints_a, keypoints_b):
            raise StateStoreError("Pair match_count exceeds the smaller keypoint count.")

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO pair_scores(
                        asset_id_a, asset_id_b, content_sha256_a, content_sha256_b,
                        capture_time_a, capture_time_b, component_fingerprint,
                        score_fingerprint, keypoint_count_a, keypoint_count_b,
                        match_count, smr, color_distance, time_distance,
                        combined_similarity, updated_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id_a, asset_id_b) DO UPDATE SET
                        content_sha256_a=excluded.content_sha256_a,
                        content_sha256_b=excluded.content_sha256_b,
                        capture_time_a=excluded.capture_time_a,
                        capture_time_b=excluded.capture_time_b,
                        component_fingerprint=excluded.component_fingerprint,
                        score_fingerprint=excluded.score_fingerprint,
                        keypoint_count_a=excluded.keypoint_count_a,
                        keypoint_count_b=excluded.keypoint_count_b,
                        match_count=excluded.match_count,
                        smr=excluded.smr,
                        color_distance=excluded.color_distance,
                        time_distance=excluded.time_distance,
                        combined_similarity=excluded.combined_similarity,
                        updated_utc=excluded.updated_utc
                    """,
                    (
                        left, right, content_a, content_b, time_a, time_b,
                        record.component_fingerprint, record.score_fingerprint,
                        int(keypoints_a), int(keypoints_b), int(record.match_count),
                        float(record.smr), float(record.color_distance),
                        float(record.time_distance), float(record.combined_similarity), now,
                    ),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not save pair score: {exc}") from exc

    def delete_pair_scores_for_assets(self, asset_ids: Iterable[str]) -> int:
        """Delete derived pair rows touching any supplied asset ID."""

        ids = tuple(sorted({str(value) for value in asset_ids if str(value)}))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        sql = (
            f"DELETE FROM pair_scores WHERE asset_id_a IN ({placeholders}) "
            f"OR asset_id_b IN ({placeholders})"
        )
        try:
            with self._connect() as connection:
                cursor = connection.execute(sql, (*ids, *ids))
                return int(cursor.rowcount if cursor.rowcount is not None else 0)
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not delete pair scores: {exc}") from exc

    def pair_score_count(self) -> int:
        try:
            with self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0])
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not count pair scores: {exc}") from exc

    def summary(self) -> dict[str, object]:
        metadata = self.metadata()
        try:
            with self._connect() as connection:
                image_count = int(
                    connection.execute("SELECT COUNT(*) FROM gallery_items").fetchone()[0]
                )
                group_count = int(
                    connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
                )
                grouped_count = int(
                    connection.execute("SELECT COUNT(*) FROM group_members").fetchone()[0]
                )
                pair_score_count = int(
                    connection.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not summarize gallery state: {exc}") from exc
        return {
            "database": str(self.database_path),
            "schema_version": int(metadata.get("schema_version", self.SCHEMA_VERSION)),
            "pipeline_fingerprint": metadata.get("pipeline_fingerprint", ""),
            "feature_fingerprint": metadata.get("feature_fingerprint", ""),
            "last_full_build_utc": metadata.get("last_full_build_utc", ""),
            "last_incremental_update_utc": metadata.get("last_incremental_update_utc", ""),
            "state_revision": self.state_revision(),
            "next_group_sequence": int(metadata.get("next_group_sequence", "0") or "0"),
            "image_count": image_count,
            "group_count": group_count,
            "grouped_image_count": grouped_count,
            "ungrouped_image_count": image_count - grouped_count,
            "pair_score_count": pair_score_count,
        }
