"""Persistent per-image feature cache for production and incremental processing.

The cache stores only derived features and metadata. It never copies or keeps the
source image itself. A cache entry is reusable only when both the image content
hash and the feature-extraction fingerprint match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from smart_gallery_similarity.photocluster.global_features import (
    GlobalFeatureConfig,
    extract_hs_histogram,
)
from smart_gallery_similarity.photocluster.sift_features import (
    SIFTConfig,
    extract_sift_features,
)


class FeatureCacheError(RuntimeError):
    """Raised when a persistent feature cache entry cannot be used."""


@dataclass(frozen=True)
class CachedImageFeatures:
    """All expensive image-derived features needed by the current pipeline."""

    hs_histogram: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


@dataclass(frozen=True)
class FeatureCacheResult:
    """Result of one cache lookup/extraction."""

    features: CachedImageFeatures
    status: Literal["hit", "miss", "stale"]
    cache_key: str


def compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 content fingerprint used only for change detection.

    This is not the optional Copy Detection feature. Here the hash is simply a
    reliable way to know whether the bytes behind one stable asset_id changed.
    """

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FeatureCacheError(f"File does not exist: {image_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()
    try:
        with image_path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise FeatureCacheError(f"Could not hash '{image_path}': {exc}") from exc
    return digest.hexdigest()


def feature_extraction_fingerprint(
    *,
    global_config: GlobalFeatureConfig,
    sift_config: SIFTConfig,
) -> str:
    """Fingerprint only settings that change cached HS/SIFT feature values.

    Pair-score weights, time horizon, linkage and final threshold deliberately do
    not invalidate raw image features because those settings do not change HS or
    SIFT extraction.
    """

    global_config.validate()
    sift_config.validate()
    payload = {
        "hs_histogram": {
            "hue_bins": int(global_config.hue_bins),
            "saturation_bins": int(global_config.saturation_bins),
        },
        "sift": asdict(sift_config),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _string_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_string_scalar(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise FeatureCacheError(f"Invalid scalar metadata field: {name}")
    return str(array.reshape(-1)[0])


class FeatureCache:
    """Filesystem-backed cache for HS histograms and SIFT descriptors."""

    FORMAT_VERSION = 1

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _asset_digest(asset_id: str) -> str:
        value = str(asset_id).strip()
        if not value:
            raise ValueError("asset_id cannot be empty.")
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def entry_path(self, asset_id: str) -> Path:
        """Return an opaque path; raw asset IDs never become filenames."""

        digest = self._asset_digest(asset_id)
        return self.root / digest[:2] / f"{digest}.npz"

    def cache_key(self, asset_id: str) -> str:
        return self.entry_path(asset_id).relative_to(self.root).as_posix()

    def _load_if_valid(
        self,
        *,
        asset_id: str,
        content_sha256: str,
        feature_fingerprint: str,
    ) -> tuple[CachedImageFeatures | None, Literal["missing", "stale"]]:
        path = self.entry_path(asset_id)
        if not path.is_file():
            return None, "missing"

        try:
            with np.load(path, allow_pickle=False) as payload:
                format_version = int(np.asarray(payload["format_version"]).reshape(-1)[0])
                saved_asset_id = _read_string_scalar(payload["asset_id"], name="asset_id")
                saved_content_hash = _read_string_scalar(
                    payload["content_sha256"], name="content_sha256"
                )
                saved_feature_fingerprint = _read_string_scalar(
                    payload["feature_fingerprint"], name="feature_fingerprint"
                )

                if (
                    format_version != self.FORMAT_VERSION
                    or saved_asset_id != asset_id
                    or saved_content_hash != content_sha256
                    or saved_feature_fingerprint != feature_fingerprint
                ):
                    return None, "stale"

                histogram = np.asarray(payload["hs_histogram"], dtype=np.float32)
                keypoints = np.asarray(payload["keypoints"], dtype=np.float32)
                descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
                original_size_array = np.asarray(payload["original_size"], dtype=np.int32)
                processed_size_array = np.asarray(payload["processed_size"], dtype=np.int32)
        except (OSError, ValueError, KeyError) as exc:
            # A corrupt/old cache file is recoverable: mark it stale and rebuild.
            return None, "stale"

        if histogram.ndim != 1 or histogram.size < 2:
            return None, "stale"
        if keypoints.ndim != 2 or keypoints.shape[1:] != (7,):
            return None, "stale"
        if descriptors.ndim != 2 or descriptors.shape[1:] != (128,):
            return None, "stale"
        if keypoints.shape[0] != descriptors.shape[0]:
            return None, "stale"
        if original_size_array.shape != (2,) or processed_size_array.shape != (2,):
            return None, "stale"
        if not (
            np.isfinite(histogram).all()
            and np.isfinite(keypoints).all()
            and np.isfinite(descriptors).all()
        ):
            return None, "stale"

        return (
            CachedImageFeatures(
                hs_histogram=np.ascontiguousarray(histogram, dtype=np.float32),
                keypoints=np.ascontiguousarray(keypoints, dtype=np.float32),
                descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
                original_size=(
                    int(original_size_array[0]),
                    int(original_size_array[1]),
                ),
                processed_size=(
                    int(processed_size_array[0]),
                    int(processed_size_array[1]),
                ),
            ),
            "stale",
        )

    def _save(
        self,
        *,
        asset_id: str,
        content_sha256: str,
        feature_fingerprint: str,
        features: CachedImageFeatures,
    ) -> Path:
        path = self.entry_path(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")

        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    format_version=np.asarray([self.FORMAT_VERSION], dtype=np.int32),
                    asset_id=_string_scalar(asset_id),
                    content_sha256=_string_scalar(content_sha256),
                    feature_fingerprint=_string_scalar(feature_fingerprint),
                    hs_histogram=np.asarray(features.hs_histogram, dtype=np.float32),
                    keypoints=np.asarray(features.keypoints, dtype=np.float32),
                    descriptors=np.asarray(features.descriptors, dtype=np.float32),
                    original_size=np.asarray(features.original_size, dtype=np.int32),
                    processed_size=np.asarray(features.processed_size, dtype=np.int32),
                )
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise FeatureCacheError(f"Could not save feature cache '{path}': {exc}") from exc
        return path

    def get_or_extract(
        self,
        *,
        asset_id: str,
        image_path: Path,
        content_sha256: str,
        feature_fingerprint: str,
        global_config: GlobalFeatureConfig,
        sift_config: SIFTConfig,
    ) -> FeatureCacheResult:
        """Load a valid cache entry or extract and atomically replace it."""

        expected_histogram_size = (
            int(global_config.hue_bins) + int(global_config.saturation_bins)
        )
        cached, prior_state = self._load_if_valid(
            asset_id=asset_id,
            content_sha256=content_sha256,
            feature_fingerprint=feature_fingerprint,
        )
        if cached is not None and cached.hs_histogram.shape == (expected_histogram_size,):
            return FeatureCacheResult(
                features=cached,
                status="hit",
                cache_key=self.cache_key(asset_id),
            )

        status: Literal["miss", "stale"] = (
            "miss" if prior_state == "missing" else "stale"
        )
        histogram = extract_hs_histogram(
            image_path,
            hue_bins=global_config.hue_bins,
            saturation_bins=global_config.saturation_bins,
        )
        sift = extract_sift_features(image_path, config=sift_config)
        features = CachedImageFeatures(
            hs_histogram=np.ascontiguousarray(histogram, dtype=np.float32),
            keypoints=np.ascontiguousarray(sift["keypoints"], dtype=np.float32),
            descriptors=np.ascontiguousarray(sift["descriptors"], dtype=np.float32),
            original_size=tuple(int(x) for x in sift["original_size"]),
            processed_size=tuple(int(x) for x in sift["processed_size"]),
        )
        self._save(
            asset_id=asset_id,
            content_sha256=content_sha256,
            feature_fingerprint=feature_fingerprint,
            features=features,
        )
        return FeatureCacheResult(
            features=features,
            status=status,
            cache_key=self.cache_key(asset_id),
        )

    def load_histogram(
        self,
        *,
        asset_id: str,
        content_sha256: str,
        feature_fingerprint: str,
    ) -> np.ndarray:
        """Load only the cached HS histogram for cheap candidate selection.

        NumPy's NPZ reader decompresses only the requested member, so this avoids
        loading large SIFT descriptor arrays for every gallery image.
        """

        path = self.entry_path(asset_id)
        if not path.is_file():
            raise FeatureCacheError(
                f"Feature cache entry is unavailable for asset_id={asset_id}"
            )

        try:
            with np.load(path, allow_pickle=False) as payload:
                format_version = int(
                    np.asarray(payload["format_version"]).reshape(-1)[0]
                )
                saved_asset_id = _read_string_scalar(
                    payload["asset_id"], name="asset_id"
                )
                saved_content_hash = _read_string_scalar(
                    payload["content_sha256"], name="content_sha256"
                )
                saved_feature_fingerprint = _read_string_scalar(
                    payload["feature_fingerprint"], name="feature_fingerprint"
                )
                histogram = np.asarray(payload["hs_histogram"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise FeatureCacheError(
                f"Could not load feature-cache histogram for asset_id={asset_id}: {exc}"
            ) from exc

        if (
            format_version != self.FORMAT_VERSION
            or saved_asset_id != asset_id
            or saved_content_hash != content_sha256
            or saved_feature_fingerprint != feature_fingerprint
        ):
            raise FeatureCacheError(
                f"Feature cache entry is stale for asset_id={asset_id}"
            )
        if histogram.ndim != 1 or histogram.size < 2:
            raise FeatureCacheError(
                f"Invalid cached histogram for asset_id={asset_id}"
            )
        if not np.isfinite(histogram).all() or np.any(histogram < 0):
            raise FeatureCacheError(
                f"Non-finite or negative cached histogram for asset_id={asset_id}"
            )
        return np.ascontiguousarray(histogram, dtype=np.float32)

    def delete(self, asset_id: str) -> bool:
        """Delete one derived cache entry after an authoritative state commit.

        Returns ``True`` only when a file was actually removed. Missing entries are
        harmless and return ``False``. Empty two-character shard directories are
        removed opportunistically.
        """

        path = self.entry_path(asset_id)
        try:
            if not path.exists():
                return False
            path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass
            return True
        except OSError as exc:
            raise FeatureCacheError(
                f"Could not delete feature cache for asset_id={asset_id}: {exc}"
            ) from exc

    def load(
        self,
        *,
        asset_id: str,
        content_sha256: str,
        feature_fingerprint: str,
    ) -> CachedImageFeatures:
        """Load an entry that is expected to have already been ensured."""

        cached, _ = self._load_if_valid(
            asset_id=asset_id,
            content_sha256=content_sha256,
            feature_fingerprint=feature_fingerprint,
        )
        if cached is None:
            raise FeatureCacheError(f"Feature cache entry is unavailable for asset_id={asset_id}")
        return cached
