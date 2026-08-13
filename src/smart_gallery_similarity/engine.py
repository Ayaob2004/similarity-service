"""High-level production API for full and incremental gallery analysis.

The first call for a gallery creates an authoritative full PhotoCluster state.
Later calls detect changes and use the incremental path automatically. Unsafe
large changes fall back to an isolated full rebuild which replaces the previous
state only after the rebuild succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal
from uuid import uuid4

from smart_gallery_similarity.gallery_pipeline import (
    GalleryPipelineError,
    run_gallery_folder_pipeline,
)
from smart_gallery_similarity.incremental_processing import (
    IncrementalProcessingError,
    IncrementalUpdateResult,
    commit_incremental_update,
)
from smart_gallery_similarity.production_config import (
    ProductionConfigError,
    ProductionPhotoClusterConfig,
    load_production_config,
)
from smart_gallery_similarity.state_store import GalleryStateStore, StateStoreError


AnalysisMode = Literal["full_build", "incremental_update", "no_changes", "full_rebuild"]
_GALLERY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SmartGalleryError(RuntimeError):
    """Raised when a production gallery analysis cannot complete safely."""


@dataclass(frozen=True)
class SimilarityGroup:
    group_id: str
    representative_asset_id: str
    asset_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "representative_asset_id": self.representative_asset_id,
            "asset_ids": list(self.asset_ids),
        }


@dataclass(frozen=True)
class GalleryAnalysisResult:
    gallery_id: str
    mode: AnalysisMode
    state_directory: Path
    image_count: int
    group_count: int
    grouped_image_count: int
    ungrouped_image_count: int
    state_revision: int
    groups: tuple[SimilarityGroup, ...]
    ungrouped_asset_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, object]

    @property
    def groups_file(self) -> Path:
        return self.state_directory / "groups.json"

    @property
    def report_file(self) -> Path:
        return self.state_directory / "last_run.json"

    @property
    def previews_directory(self) -> Path:
        return self.state_directory / "previews"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "gallery_id": self.gallery_id,
            "mode": self.mode,
            "state_directory": str(self.state_directory),
            "image_count": self.image_count,
            "group_count": self.group_count,
            "grouped_image_count": self.grouped_image_count,
            "ungrouped_image_count": self.ungrouped_image_count,
            "state_revision": self.state_revision,
            "groups": [group.to_dict() for group in self.groups],
            "ungrouped_asset_ids": list(self.ungrouped_asset_ids),
            "previews_directory": (
                str(self.previews_directory)
                if self.previews_directory.is_dir()
                else None
            ),
            "warnings": list(self.warnings),
            "statistics": self.statistics,
        }


def _validate_gallery_id(gallery_id: str) -> str:
    value = str(gallery_id).strip()
    if not _GALLERY_ID.fullmatch(value):
        raise SmartGalleryError(
            "gallery_id must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-' (maximum 128 characters)."
        )
    return value


def _snapshot(store: GalleryStateStore) -> tuple[
    tuple[SimilarityGroup, ...], tuple[str, ...]
]:
    groups = tuple(
        SimilarityGroup(
            group_id=str(record["group_id"]),
            representative_asset_id=str(record["representative_asset_id"]),
            asset_ids=tuple(str(value) for value in record["member_asset_ids"]),
        )
        for record in store.groups()
    )
    grouped = {asset_id for group in groups for asset_id in group.asset_ids}
    ungrouped = tuple(
        str(item["asset_id"])
        for item in store.items()
        if str(item["asset_id"]) not in grouped
    )
    return groups, ungrouped


def _export_group_previews(
    *,
    store: GalleryStateStore,
    groups: tuple[SimilarityGroup, ...],
    input_directory: Path,
    state_directory: Path,
) -> Path:
    """Atomically rebuild preview folders from the authoritative state.

    Preview folders are convenience copies for local inspection only. They are
    not used by the similarity algorithm or the incremental caches. Rebuilding
    the complete preview tree prevents stale folders after images are deleted,
    modified, or moved between groups.
    """

    source_root = Path(input_directory).expanduser().resolve()
    if not source_root.is_dir():
        raise SmartGalleryError(f"Input folder does not exist: {source_root}")

    item_by_asset = store.items_by_asset_id()
    preview_root = state_directory / "previews"
    staging = state_directory / f".previews-build-{uuid4().hex}"
    backup = state_directory / f".previews-backup-{uuid4().hex}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)

    moved_old = False
    try:
        for group in groups:
            group_directory = staging / group.group_id
            group_directory.mkdir(parents=False, exist_ok=False)

            for position, asset_id in enumerate(group.asset_ids, start=1):
                item = item_by_asset.get(asset_id)
                if item is None:
                    raise SmartGalleryError(
                        f"Preview export could not find asset_id={asset_id} in state."
                    )

                relative_path = Path(str(item["relative_path"]))
                source = (source_root / relative_path).resolve()
                try:
                    source.relative_to(source_root)
                except ValueError as exc:
                    raise SmartGalleryError(
                        f"Unsafe preview source path for asset_id={asset_id}: "
                        f"{relative_path}"
                    ) from exc

                if not source.is_file():
                    raise SmartGalleryError(
                        f"Preview source image does not exist for "
                        f"asset_id={asset_id}: {source}"
                    )

                destination = group_directory / f"{position:03d}_{source.name}"
                shutil.copy2(source, destination)

        if preview_root.exists():
            preview_root.rename(backup)
            moved_old = True
        staging.rename(preview_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if not preview_root.exists() and moved_old and backup.exists():
            backup.rename(preview_root)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)

    return preview_root


def _write_public_outputs(result: GalleryAnalysisResult) -> None:
    result.state_directory.mkdir(parents=True, exist_ok=True)
    result.groups_file.write_text(
        json.dumps(
            [list(group.asset_ids) for group in result.groups],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result.report_file.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _promote_state(*, prepared: Path, target: Path) -> None:
    """Replace a gallery state only after a full build has succeeded."""

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
    moved_old = False
    try:
        if target.exists():
            target.rename(backup)
            moved_old = True
        prepared.rename(target)
    except Exception:
        if not target.exists() and moved_old and backup.exists():
            backup.rename(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _result_from_store(
    *,
    gallery_id: str,
    mode: AnalysisMode,
    state_directory: Path,
    input_directory: Path,
    export_previews: bool,
    warnings: tuple[str, ...] | list[str],
    statistics: dict[str, object],
) -> GalleryAnalysisResult:
    try:
        store = GalleryStateStore(state_directory / "state.sqlite3")
        summary = store.summary()
        groups, ungrouped = _snapshot(store)
    except StateStoreError as exc:
        raise SmartGalleryError(str(exc)) from exc

    result = GalleryAnalysisResult(
        gallery_id=gallery_id,
        mode=mode,
        state_directory=state_directory,
        image_count=int(summary["image_count"]),
        group_count=int(summary["group_count"]),
        grouped_image_count=int(summary["grouped_image_count"]),
        ungrouped_image_count=int(summary["ungrouped_image_count"]),
        state_revision=int(summary["state_revision"]),
        groups=groups,
        ungrouped_asset_ids=ungrouped,
        warnings=tuple(dict.fromkeys(str(value) for value in warnings if str(value))),
        statistics=statistics,
    )
    if export_previews:
        _export_group_previews(
            store=store,
            groups=groups,
            input_directory=input_directory,
            state_directory=state_directory,
        )
    _write_public_outputs(result)
    return result


def _full_build(
    *,
    gallery_id: str,
    input_dir: Path,
    state_directory: Path,
    runtime_directory: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None,
    recursive: bool,
    mode: AnalysisMode,
    reason: str | None,
    export_previews: bool,
) -> GalleryAnalysisResult:
    work_root = runtime_directory / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    prepared_state = state_directory.parent / f".{state_directory.name}.build-{uuid4().hex}"
    shutil.rmtree(prepared_state, ignore_errors=True)

    warnings: tuple[str, ...] = ()
    statistics: dict[str, object] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="full-build-", dir=work_root) as temp:
            output = Path(temp) / "output"
            pipeline_result = run_gallery_folder_pipeline(
                input_dir=input_dir,
                output_dir=output,
                config=config,
                manifest_path=manifest_path,
                recursive=recursive,
                overwrite=False,
                export_previews=False,
                state_directory=prepared_state,
                debug_artifacts=False,
            )
            warnings = pipeline_result.warnings
            report_path = output / "run_report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                persistent = report.get("persistent_state", {})
                statistics = {
                    "elapsed_seconds": report.get("elapsed_seconds", 0.0),
                    "initial_set_count": report.get("initial_set_count", 0),
                    "matched_within_set_pair_count": report.get(
                        "matched_within_set_pair_count", 0
                    ),
                    "largest_group_size": report.get("largest_group_size", 0),
                    "feature_cache": {
                        "hits": persistent.get("feature_cache_hits", 0),
                        "misses": persistent.get("feature_cache_misses", 0),
                        "stale_rebuilt": persistent.get(
                            "feature_cache_stale_rebuilt", 0
                        ),
                    },
                }
        _promote_state(prepared=prepared_state, target=state_directory)
    except (GalleryPipelineError, StateStoreError, OSError, ValueError) as exc:
        shutil.rmtree(prepared_state, ignore_errors=True)
        raise SmartGalleryError(str(exc)) from exc
    except Exception:
        shutil.rmtree(prepared_state, ignore_errors=True)
        raise
    finally:
        try:
            work_root.rmdir()
        except OSError:
            pass

    if reason:
        statistics["full_rebuild_reason"] = reason
    return _result_from_store(
        gallery_id=gallery_id,
        mode=mode,
        state_directory=state_directory,
        input_directory=input_dir,
        export_previews=export_previews,
        warnings=warnings,
        statistics=statistics,
    )


def _incremental_statistics(update: IncrementalUpdateResult) -> dict[str, object]:
    changes = update.change_set
    planning = update.planning
    pair_preparation = planning.pair_preparation
    features = pair_preparation.feature_preparation
    pair_scores = pair_preparation.pair_scores
    safety = planning.safety
    return {
        "changes": {
            "added": len(changes.added),
            "modified": len(changes.modified),
            "deleted": len(changes.deleted),
            "unchanged": len(changes.unchanged),
        },
        "feature_extractions": features.feature_extraction_count,
        "candidate_pairs": (
            0
            if pair_preparation.candidate_selection is None
            else pair_preparation.candidate_selection.unique_pair_count
        ),
        "sift_pair_matches": (
            0 if pair_scores is None else pair_scores.sift_match_computation_count
        ),
        "pair_cache_hits": 0 if pair_scores is None else pair_scores.cache_hits,
        "pair_cache_misses": 0 if pair_scores is None else pair_scores.cache_misses,
        "safety_status": None if safety is None else safety.status,
        "rebuild_recommended": False if safety is None else safety.rebuild_recommended,
    }


def analyze_gallery(
    *,
    input_dir: Path,
    gallery_id: str,
    runtime_dir: Path = Path("runtime"),
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    recursive: bool = False,
    fresh: bool = False,
    export_previews: bool = False,
    candidate_caps: tuple[int, ...] = (128, 256, 512, 1024),
) -> GalleryAnalysisResult:
    """Analyze one gallery, choosing full or incremental processing automatically.

    ``fresh=True`` forces an isolated full build. The existing state is kept
    until the replacement build succeeds. Candidate truncation is retried with
    wider caps before the engine falls back to a full rebuild.
    """

    key = _validate_gallery_id(gallery_id)
    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise SmartGalleryError(f"Input folder does not exist: {source}")
    if not candidate_caps or any(int(value) < 1 for value in candidate_caps):
        raise SmartGalleryError("candidate_caps must contain positive integers.")

    runtime = Path(runtime_dir).expanduser().resolve()
    state_directory = runtime / "galleries" / key
    config = load_production_config(config_path)

    database = state_directory / "state.sqlite3"
    has_state = False
    if database.is_file():
        try:
            has_state = GalleryStateStore(database).has_authoritative_snapshot()
        except StateStoreError as exc:
            raise SmartGalleryError(str(exc)) from exc

    if fresh or not has_state:
        mode: AnalysisMode = "full_rebuild" if has_state else "full_build"
        return _full_build(
            gallery_id=key,
            input_dir=source,
            state_directory=state_directory,
            runtime_directory=runtime,
            config=config,
            manifest_path=manifest_path,
            recursive=recursive,
            mode=mode,
            reason="fresh_requested" if fresh and has_state else None,
            export_previews=export_previews,
        )

    last_update: IncrementalUpdateResult | None = None
    for cap in tuple(dict.fromkeys(int(value) for value in candidate_caps)):
        try:
            last_update = commit_incremental_update(
                state_directory=state_directory,
                input_dir=source,
                config=config,
                manifest_path=manifest_path,
                recursive=recursive,
                max_seed_candidates_per_changed=cap,
            )
        except IncrementalProcessingError as exc:
            raise SmartGalleryError(str(exc)) from exc

        if last_update.blocked_by_full_rebuild:
            reasons = list(last_update.change_set.rebuild_reasons)
            safety = last_update.planning.safety
            if safety is not None:
                reasons.extend(safety.reasons)
            return _full_build(
                gallery_id=key,
                input_dir=source,
                state_directory=state_directory,
                runtime_directory=runtime,
                config=config,
                manifest_path=manifest_path,
                recursive=recursive,
                mode="full_rebuild",
                reason=",".join(dict.fromkeys(reasons)) or "incremental_safety_rule",
                export_previews=export_previews,
            )

        if not last_update.blocked_incremental_commit:
            break

        safety = last_update.planning.safety
        if safety is None or not safety.retry_with_wider_candidates:
            return _full_build(
                gallery_id=key,
                input_dir=source,
                state_directory=state_directory,
                runtime_directory=runtime,
                config=config,
                manifest_path=manifest_path,
                recursive=recursive,
                mode="full_rebuild",
                reason="incremental_commit_blocked",
                export_previews=export_previews,
            )
    else:
        return _full_build(
            gallery_id=key,
            input_dir=source,
            state_directory=state_directory,
            runtime_directory=runtime,
            config=config,
            manifest_path=manifest_path,
            recursive=recursive,
            mode="full_rebuild",
            reason="candidate_limits_exhausted",
            export_previews=export_previews,
        )

    if last_update is None:
        raise SmartGalleryError("Incremental update did not produce a result.")

    safety = last_update.planning.safety
    warnings: tuple[str, ...] = () if safety is None else safety.warnings
    return _result_from_store(
        gallery_id=key,
        mode="no_changes" if last_update.no_changes else "incremental_update",
        state_directory=state_directory,
        input_directory=source,
        export_previews=export_previews,
        warnings=warnings,
        statistics=_incremental_statistics(last_update),
    )
