"""Generic PhotoCluster runner for a folder of user-owned images.

This module deliberately does not depend on California-ND or ground truth. It is
also the reusable core that a future FastAPI endpoint can call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from smart_gallery_similarity.feature_cache import (
    FeatureCache,
    FeatureCacheError,
    compute_file_sha256,
)
from smart_gallery_similarity.photocluster.combined_distance import (
    build_combined_distance_matrices,
)
from smart_gallery_similarity.photocluster.final_grouping import (
    build_non_overlapping_groups,
    group_medoid,
    group_similarity_summary,
)
from smart_gallery_similarity.photocluster.global_features import (
    bhattacharyya_distance_matrix,
    combine_feature_distances,
    datetime_to_naive_epoch_seconds,
    extract_hs_histogram,
    parse_exif_datetime,
)
from smart_gallery_similarity.photocluster.initial_partition import (
    build_initial_sets,
    run_initial_affinity_propagation,
)
from smart_gallery_similarity.photocluster.sift_features import (
    extract_sift_features,
    load_sift_feature_file,
    mutual_best_matches,
    save_sift_feature_file,
    sift_matching_ratio,
)
from smart_gallery_similarity.production_config import (
    ProductionPhotoClusterConfig,
)
from smart_gallery_similarity.state_store import (
    GalleryStateGroup,
    GalleryStateItem,
    GalleryStateStore,
    StateStoreError,
)
from smart_gallery_similarity.time_policy import build_time_distance_matrix


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXIF_DATETIME = 306
EXIF_DATETIME_ORIGINAL = 36867


class GalleryPipelineError(RuntimeError):
    """Raised when a local gallery run cannot continue."""


@dataclass(frozen=True)
class GalleryImage:
    row: int
    asset_id: str
    filename: str
    relative_path: str
    absolute_path: Path
    capture_time: str
    capture_time_source: str
    width: int
    height: int

    def public_record(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "asset_id": self.asset_id,
            "image_id": self.asset_id,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "capture_time": self.capture_time,
            "capture_time_source": self.capture_time_source,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class GalleryPipelineResult:
    output_directory: Path
    groups: tuple[tuple[str, ...], ...]
    image_count: int
    group_count: int
    grouped_image_count: int
    ungrouped_image_count: int
    warnings: tuple[str, ...]


def _clean_exif(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise GalleryPipelineError(f"Manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GalleryPipelineError(f"Could not read manifest: {exc}") from exc
    if not isinstance(payload, list):
        raise GalleryPipelineError("Manifest must be a JSON list of image records.")

    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise GalleryPipelineError(f"Manifest item {index} is not an object.")
        relative_path = str(item.get("relative_path") or item.get("filename") or "").strip()
        if not relative_path:
            raise GalleryPipelineError(
                f"Manifest item {index} needs relative_path or filename."
            )
        key = Path(relative_path).as_posix().casefold()
        if key in result:
            raise GalleryPipelineError(f"Duplicate manifest path: {relative_path}")
        result[key] = item
    return result


def _default_asset_id(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"local_{digest}"


def scan_gallery_folder(
    input_dir: Path,
    *,
    manifest_path: Path | None = None,
    recursive: bool = False,
    limit: int | None = None,
    minimum_image_count: int = 2,
) -> tuple[list[GalleryImage], list[str]]:
    """Scan an image folder and resolve asset IDs and capture times.

    Capture-time precedence is: manifest, EXIF, file modification time. The
    fallback makes local smoke tests possible, but Flutter should later send the
    real capture time explicitly.
    """

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise GalleryPipelineError(f"Input folder does not exist: {root}")
    if minimum_image_count < 0:
        raise GalleryPipelineError("minimum_image_count cannot be negative.")
    if limit is not None and limit < minimum_image_count:
        raise GalleryPipelineError(
            f"--limit must be at least {minimum_image_count}."
        )

    manifest = _load_manifest(manifest_path)
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda value: value.relative_to(root).as_posix().casefold(),
    )
    if limit is not None:
        paths = paths[:limit]
    if len(paths) < minimum_image_count:
        raise GalleryPipelineError(
            f"At least {minimum_image_count} supported image(s) are required. "
            "Supported formats: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    images: list[GalleryImage] = []
    warnings: list[str] = []
    seen_asset_ids: set[str] = set()

    for row, path in enumerate(paths):
        relative_path = path.relative_to(root).as_posix()
        manifest_item = manifest.get(relative_path.casefold(), {})
        try:
            with Image.open(path) as image:
                image.load()
                exif = image.getexif()
                exif_time = _clean_exif(
                    exif.get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
                )
                width = int(image.width)
                height = int(image.height)
        except (OSError, UnidentifiedImageError) as exc:
            raise GalleryPipelineError(f"Cannot decode '{path}': {exc}") from exc

        manifest_time = str(manifest_item.get("capture_time") or "").strip()
        if manifest_time:
            capture_time = manifest_time
            source = "manifest"
        elif exif_time:
            capture_time = exif_time
            source = "exif"
        else:
            capture_time = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y:%m:%d %H:%M:%S"
            )
            source = "file_mtime"
            warnings.append(
                f"{relative_path}: capture time missing; file modification time was used."
            )

        # Validate the accepted formats now rather than failing halfway through.
        try:
            parse_exif_datetime(capture_time)
        except Exception as exc:  # Converted to a domain-specific, file-aware error.
            raise GalleryPipelineError(
                f"Invalid capture_time for '{relative_path}': {capture_time!r}"
            ) from exc

        asset_id = str(manifest_item.get("asset_id") or "").strip()
        if not asset_id:
            asset_id = _default_asset_id(relative_path)
        if asset_id in seen_asset_ids:
            raise GalleryPipelineError(f"Duplicate asset_id: {asset_id}")
        seen_asset_ids.add(asset_id)

        images.append(
            GalleryImage(
                row=row,
                asset_id=asset_id,
                filename=path.name,
                relative_path=relative_path,
                absolute_path=path,
                capture_time=capture_time,
                capture_time_source=source,
                width=width,
                height=height,
            )
        )

    unused_manifest = sorted(set(manifest) - {item.relative_path.casefold() for item in images})
    if unused_manifest:
        warnings.append(
            f"Manifest contains {len(unused_manifest)} path(s) not found in the input folder."
        )
    return images, warnings


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise GalleryPipelineError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_gallery_folder_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    config: ProductionPhotoClusterConfig,
    manifest_path: Path | None = None,
    recursive: bool = False,
    limit: int | None = None,
    overwrite: bool = False,
    export_previews: bool = False,
    state_directory: Path | None = None,
    debug_artifacts: bool = False,
) -> GalleryPipelineResult:
    """Run the complete fixed-config PhotoCluster pipeline.

    Production runs persist only ``groups.json`` and ``run_report.json`` in the
    supplied output directory. Large matrices and diagnostic JSON files are
    written only when ``debug_artifacts=True``. When a persistent state directory
    is supplied, HS/SIFT features are loaded directly from the feature cache and
    are not duplicated into a per-run descriptor directory.
    """

    config.validate()
    if state_directory is not None and not config.time_distance.is_incremental_safe:
        raise GalleryPipelineError(
            "Persistent incremental state requires an incremental-safe time policy."
        )

    output = Path(output_dir).expanduser().resolve()
    _prepare_output(output, overwrite=overwrite)
    started = time.perf_counter()

    descriptor_dir: Path | None = None
    try:
        images, warnings = scan_gallery_folder(
            input_dir,
            manifest_path=manifest_path,
            recursive=recursive,
            limit=limit,
        )
        count = len(images)
        records = [item.public_record() for item in images]
        keypoint_counts = np.zeros(count, dtype=np.int32)

        state_root: Path | None = None
        cache: FeatureCache | None = None
        state_store: GalleryStateStore | None = None
        feature_fingerprint = config.feature_fingerprint()
        content_hashes: dict[str, str] = {}
        cache_keys: dict[str, str] = {}
        cache_hits = 0
        cache_misses = 0
        cache_stale = 0

        if state_directory is not None:
            state_root = Path(state_directory).expanduser().resolve()
            state_root.mkdir(parents=True, exist_ok=True)
            cache = FeatureCache(state_root / "features")
            state_store = GalleryStateStore(state_root / "state.sqlite3")
        else:
            descriptor_dir = output / "_sift_work"
            descriptor_dir.mkdir(parents=True, exist_ok=False)

        # Step 1: per-image features and stable capture-time distances.
        histogram_rows: list[np.ndarray] = []
        if cache is not None:
            for item in images:
                content_hash = compute_file_sha256(item.absolute_path)
                content_hashes[item.asset_id] = content_hash
                cache_result = cache.get_or_extract(
                    asset_id=item.asset_id,
                    image_path=item.absolute_path,
                    content_sha256=content_hash,
                    feature_fingerprint=feature_fingerprint,
                    global_config=config.global_features,
                    sift_config=config.sift,
                )
                cache_keys[item.asset_id] = cache_result.cache_key
                if cache_result.status == "hit":
                    cache_hits += 1
                elif cache_result.status == "miss":
                    cache_misses += 1
                else:
                    cache_stale += 1

                feature = cache_result.features
                histogram_rows.append(feature.hs_histogram)
                keypoint_counts[item.row] = int(feature.descriptors.shape[0])
        else:
            histogram_rows = [
                extract_hs_histogram(
                    item.absolute_path,
                    hue_bins=config.global_features.hue_bins,
                    saturation_bins=config.global_features.saturation_bins,
                )
                for item in images
            ]

        histograms = np.stack(histogram_rows).astype(np.float32, copy=False)
        capture_seconds = np.asarray(
            [
                datetime_to_naive_epoch_seconds(parse_exif_datetime(item.capture_time))
                for item in images
            ],
            dtype=np.float64,
        )
        color_distance = bhattacharyya_distance_matrix(histograms)
        time_distance = build_time_distance_matrix(
            capture_seconds,
            config=config.time_distance,
        )
        global_dissimilarity = combine_feature_distances(
            color_distance,
            time_distance,
            color_weight=config.global_features.color_weight,
            time_weight=config.global_features.time_weight,
        )

        # Step 2: broad initial partition.
        partition = run_initial_affinity_propagation(
            global_dissimilarity,
            config=config.initial_partition,
        )
        if not partition.converged:
            warnings.append(
                "Initial Affinity Propagation did not formally converge; its "
                "returned partition was still used."
            )
        initial_sets = build_initial_sets(
            labels=partition.labels,
            exemplar_rows=partition.exemplar_rows,
            records=records,
            dissimilarity=global_dissimilarity,
        )

        # Step 3: extract SIFT only when no persistent feature cache is enabled.
        if cache is None:
            assert descriptor_dir is not None
            for item in images:
                feature = extract_sift_features(item.absolute_path, config=config.sift)
                keypoint_counts[item.row] = int(feature["descriptors"].shape[0])
                save_sift_feature_file(
                    descriptor_dir / f"row_{item.row:04d}.npz",
                    keypoints=feature["keypoints"],
                    descriptors=feature["descriptors"],
                    original_size=feature["original_size"],
                    processed_size=feature["processed_size"],
                )

        computed_pair_mask = np.eye(count, dtype=bool)
        sift_distance = np.full((count, count), np.nan, dtype=np.float32)
        np.fill_diagonal(sift_distance, 0.0)
        matched_pair_count = 0

        for initial_set in initial_sets:
            rows = [int(row) for row in initial_set["member_rows"]]
            if cache is not None:
                features = {
                    row: cache.load(
                        asset_id=images[row].asset_id,
                        content_sha256=content_hashes[images[row].asset_id],
                        feature_fingerprint=feature_fingerprint,
                    )
                    for row in rows
                }
            else:
                assert descriptor_dir is not None
                features = {
                    row: load_sift_feature_file(
                        descriptor_dir / f"row_{row:04d}.npz"
                    )
                    for row in rows
                }

            for left_index, row_a in enumerate(rows):
                for row_b in rows[left_index + 1 :]:
                    if cache is not None:
                        descriptors_a = features[row_a].descriptors
                        descriptors_b = features[row_b].descriptors
                    else:
                        descriptors_a = features[row_a]["descriptors"]
                        descriptors_b = features[row_b]["descriptors"]
                    matches = mutual_best_matches(descriptors_a, descriptors_b)
                    smr = sift_matching_ratio(
                        int(descriptors_a.shape[0]),
                        int(descriptors_b.shape[0]),
                        len(matches),
                    )
                    distance = float(1.0 - smr)
                    sift_distance[row_a, row_b] = distance
                    sift_distance[row_b, row_a] = distance
                    computed_pair_mask[row_a, row_b] = True
                    computed_pair_mask[row_b, row_a] = True
                    matched_pair_count += 1

        # Step 4: fixed 3:2:1 combined score.
        combined_distance, combined_similarity = build_combined_distance_matrices(
            sift_distance=sift_distance,
            color_distance=color_distance,
            time_distance=time_distance,
            computed_pair_mask=computed_pair_mask,
            config=config.combined_distance,
        )

        # Final production grouping.
        grouping = build_non_overlapping_groups(
            combined_similarity=combined_similarity,
            initial_sets=initial_sets,
            config=config.final_grouping,
        )

        minimal_groups: list[list[str]] = []
        filename_groups: list[list[str]] = []
        detailed_groups: list[dict[str, Any]] = []
        row_to_set = {
            int(row): str(initial_set["set_id"])
            for initial_set in initial_sets
            for row in initial_set["member_rows"]
        }
        for index, members_tuple in enumerate(grouping.groups, start=1):
            members = list(members_tuple)
            group_id = f"group_{index:04d}"
            asset_ids = [images[row].asset_id for row in members]
            filenames = [images[row].relative_path for row in members]
            representative_row = group_medoid(
                members,
                combined_similarity=combined_similarity,
            )
            summary = group_similarity_summary(
                members,
                combined_similarity=combined_similarity,
            )
            minimal_groups.append(asset_ids)
            filename_groups.append(filenames)
            detailed_groups.append(
                {
                    "group_id": group_id,
                    "initial_set_id": row_to_set[members[0]],
                    "size": len(members),
                    "representative_asset_id": images[representative_row].asset_id,
                    "representative_filename": images[representative_row].relative_path,
                    **summary,
                    "members": [records[row] for row in members],
                }
            )

        if export_previews:
            preview_root = output / "previews"
            for group_index, members_tuple in enumerate(grouping.groups, start=1):
                group_dir = preview_root / f"group_{group_index:04d}"
                group_dir.mkdir(parents=True, exist_ok=False)
                for member_index, row in enumerate(members_tuple, start=1):
                    source = images[row].absolute_path
                    destination = group_dir / f"{member_index:03d}_{source.name}"
                    shutil.copy2(source, destination)

        grouped_count = int(sum(len(group) for group in grouping.groups))

        # Minimal production output.
        _write_json(output / "groups.json", minimal_groups)

        if debug_artifacts:
            _write_json(
                output / "config_snapshot.json",
                {
                    "config_fingerprint": config.fingerprint(),
                    "feature_fingerprint": feature_fingerprint,
                    "config": config.to_dict(),
                },
            )
            np.save(
                output / "combined_similarity.npy",
                combined_similarity,
                allow_pickle=False,
            )
            np.save(
                output / "combined_distance.npy",
                combined_distance,
                allow_pickle=False,
            )
            _write_json(output / "image_records.json", records)
            _write_json(output / "initial_sets.json", {"sets": initial_sets})
            _write_json(output / "groups_filenames.json", filename_groups)
            _write_json(output / "groups_detailed.json", detailed_groups)

        state_summary: dict[str, Any] = {"enabled": False}
        if state_store is not None and state_root is not None:
            state_items = [
                GalleryStateItem(
                    asset_id=item.asset_id,
                    filename=item.filename,
                    relative_path=item.relative_path,
                    capture_time=item.capture_time,
                    capture_time_source=item.capture_time_source,
                    width=item.width,
                    height=item.height,
                    content_sha256=content_hashes[item.asset_id],
                    feature_cache_key=cache_keys[item.asset_id],
                    feature_fingerprint=feature_fingerprint,
                )
                for item in images
            ]
            state_groups = [
                GalleryStateGroup(
                    group_id=str(group["group_id"]),
                    representative_asset_id=str(group["representative_asset_id"]),
                    member_asset_ids=tuple(
                        str(member["asset_id"]) for member in group["members"]
                    ),
                )
                for group in detailed_groups
            ]
            state_store.replace_full_snapshot(
                items=state_items,
                groups=state_groups,
                pipeline_fingerprint=config.fingerprint(),
                feature_fingerprint=feature_fingerprint,
            )
            state_summary = {
                "enabled": True,
                "state_directory": str(state_root),
                "database": str(state_store.database_path),
                "feature_cache_directory": str(state_root / "features"),
                "feature_fingerprint": feature_fingerprint,
                "feature_cache_hits": cache_hits,
                "feature_cache_misses": cache_misses,
                "feature_cache_stale_rebuilt": cache_stale,
            }

        elapsed = float(time.perf_counter() - started)
        report = {
            "status": "passed",
            "input_directory": str(Path(input_dir).expanduser().resolve()),
            "output_directory": str(output),
            "image_count": count,
            "initial_set_count": len(initial_sets),
            "matched_within_set_pair_count": matched_pair_count,
            "group_count": len(grouping.groups),
            "grouped_image_count": grouped_count,
            "ungrouped_image_count": count - grouped_count,
            "largest_group_size": max((len(group) for group in grouping.groups), default=0),
            "config_fingerprint": config.fingerprint(),
            "feature_fingerprint": feature_fingerprint,
            "capture_time_sources": {
                source: sum(item.capture_time_source == source for item in images)
                for source in ("manifest", "exif", "file_mtime")
            },
            "persistent_state": state_summary,
            "warnings": warnings,
            "elapsed_seconds": elapsed,
            "debug_artifacts": bool(debug_artifacts),
        }
        _write_json(output / "run_report.json", report)

        if descriptor_dir is not None and descriptor_dir.exists() and not debug_artifacts:
            shutil.rmtree(descriptor_dir, ignore_errors=True)

        return GalleryPipelineResult(
            output_directory=output,
            groups=tuple(tuple(group) for group in minimal_groups),
            image_count=count,
            group_count=len(minimal_groups),
            grouped_image_count=grouped_count,
            ungrouped_image_count=count - grouped_count,
            warnings=tuple(warnings),
        )
    except Exception:
        if output.exists() and not (output / "run_report.json").is_file():
            shutil.rmtree(output, ignore_errors=True)
        raise
