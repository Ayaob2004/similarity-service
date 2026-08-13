"""Global color and time features used by the PhotoCluster first stage.

The PhotoCluster paper uses two global features for the initial partition:
Hue-Saturation color histograms and capture timestamps, with equal weights.

The paper does not state the histogram bin counts. This implementation uses
all 256 values of Pillow's 8-bit Hue and Saturation channels by default and
keeps the counts configurable so later experiments are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps


EXIF_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


class GlobalFeatureError(RuntimeError):
    """Raised when PhotoCluster global feature extraction cannot continue."""


@dataclass(frozen=True)
class GlobalFeatureConfig:
    """Configuration for PhotoCluster global features."""

    hue_bins: int = 256
    saturation_bins: int = 256
    color_weight: float = 1.0
    time_weight: float = 1.0

    def validate(self) -> None:
        if self.hue_bins < 2:
            raise ValueError("hue_bins must be at least 2.")
        if self.saturation_bins < 2:
            raise ValueError("saturation_bins must be at least 2.")
        if self.color_weight < 0 or self.time_weight < 0:
            raise ValueError("Feature weights cannot be negative.")
        if self.color_weight + self.time_weight <= 0:
            raise ValueError("At least one global feature weight must be positive.")

    @property
    def feature_dimension(self) -> int:
        return self.hue_bins + self.saturation_bins


def parse_exif_datetime(value: str | datetime | None) -> datetime:
    """Parse a capture-time value without applying a local timezone."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    if value is None:
        raise GlobalFeatureError("Capture time is missing.")

    text = str(value).strip()
    if not text:
        raise GlobalFeatureError("Capture time is empty.")

    for date_format in EXIF_DATETIME_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    raise GlobalFeatureError(
        f"Unsupported capture-time format: {text!r}. "
        f"Expected one of: {EXIF_DATETIME_FORMATS}."
    )


def datetime_to_naive_epoch_seconds(value: datetime) -> float:
    """Convert a naive datetime to deterministic seconds from 1970-01-01."""
    epoch = datetime(1970, 1, 1)
    return float((value.replace(tzinfo=None) - epoch).total_seconds())


def _normalized_histogram(
    values: np.ndarray,
    *,
    bins: int,
) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32, copy=False)
    total = float(histogram.sum())
    if total <= 0:
        raise GlobalFeatureError("An empty color histogram was produced.")
    histogram /= total
    return histogram


def extract_hs_histogram(
    image_path: Path,
    *,
    hue_bins: int = 256,
    saturation_bins: int = 256,
) -> np.ndarray:
    """Extract independent normalized Hue and Saturation histograms.

    Value/brightness is deliberately omitted, matching the feature definition
    cited by PhotoCluster. The two independently L1-normalized histograms are
    concatenated into one vector.
    """
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise GlobalFeatureError(f"Image does not exist: {image_path}")

    try:
        with Image.open(image_path) as source:
            source = ImageOps.exif_transpose(source)
            hsv = source.convert("RGB").convert("HSV")
            pixels = np.asarray(hsv, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise GlobalFeatureError(
            f"Could not read image '{image_path}': {exc}"
        ) from exc

    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise GlobalFeatureError(
            f"Unexpected HSV image shape {pixels.shape} for {image_path}."
        )

    hue = _normalized_histogram(pixels[..., 0], bins=hue_bins)
    saturation = _normalized_histogram(
        pixels[..., 1], bins=saturation_bins
    )
    return np.concatenate([hue, saturation]).astype(np.float32, copy=False)


def bhattacharyya_distance_matrix(histograms: np.ndarray) -> np.ndarray:
    """Compute all-pairs Bhattacharyya distance in [0, 1].

    Each Hue and Saturation histogram is independently normalized, so the
    concatenated vector sums to two. Dividing each complete vector by its sum
    gives Hue and Saturation equal contribution and turns it into a valid
    probability distribution before calculating the Bhattacharyya/Hellinger
    distance sqrt(1 - BC).
    """
    matrix = np.asarray(histograms, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"histograms must be a 2-D matrix, found {matrix.shape}."
        )
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("histograms matrix is empty or too narrow.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("histograms contain invalid values.")

    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every histogram row must have a positive sum.")

    probabilities = matrix / row_sums
    square_roots = np.sqrt(probabilities)
    coefficients = square_roots @ square_roots.T
    coefficients = np.clip(coefficients, 0.0, 1.0)
    distances = np.sqrt(np.clip(1.0 - coefficients, 0.0, 1.0))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def normalized_time_distance_matrix(
    capture_seconds: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute pairwise absolute time differences normalized to [0, 1].

    The normalization range is the interval from the oldest to the newest
    image in the collection, as described by the source used by PhotoCluster.
    """
    times = np.asarray(capture_seconds, dtype=np.float64)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("capture_seconds must be a non-empty 1-D array.")
    if not np.isfinite(times).all():
        raise ValueError("capture_seconds contains NaN or infinite values.")

    span = float(times.max() - times.min())
    absolute_differences = np.abs(times[:, None] - times[None, :])
    if span <= 0:
        return np.zeros((times.size, times.size), dtype=np.float32)

    distances = np.clip(absolute_differences / span, 0.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def combine_feature_distances(
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    *,
    color_weight: float = 1.0,
    time_weight: float = 1.0,
) -> np.ndarray:
    """Combine normalized feature distances using PhotoCluster equation 4.

    Equation 4 is equivalent to the weighted mean of normalized feature
    distances. PhotoCluster assigns equal importance to color and timestamps
    in the initial global partition, so the default is their arithmetic mean.
    """
    color = np.asarray(color_distance, dtype=np.float64)
    time = np.asarray(time_distance, dtype=np.float64)

    if color.shape != time.shape or color.ndim != 2:
        raise ValueError(
            "color_distance and time_distance must be equally sized 2-D matrices."
        )
    if color.shape[0] != color.shape[1]:
        raise ValueError("Distance matrices must be square.")
    if color_weight < 0 or time_weight < 0:
        raise ValueError("Feature weights cannot be negative.")

    denominator = float(color_weight + time_weight)
    if denominator <= 0:
        raise ValueError("At least one feature weight must be positive.")

    combined = (
        color_weight * color + time_weight * time
    ) / denominator
    combined = np.clip(combined, 0.0, 1.0)
    np.fill_diagonal(combined, 0.0)
    return combined.astype(np.float32)


def _matrix_checks(name: str, matrix: np.ndarray, count: int) -> list[str]:
    errors: list[str] = []
    if matrix.shape != (count, count):
        errors.append(
            f"{name} shape is {matrix.shape}; expected {(count, count)}."
        )
        return errors
    if not np.isfinite(matrix).all():
        errors.append(f"{name} contains NaN or infinite values.")
    if np.any(matrix < -1e-7) or np.any(matrix > 1.0 + 1e-7):
        errors.append(f"{name} contains values outside [0, 1].")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        errors.append(f"{name} is not symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-7):
        errors.append(f"{name} diagonal is not zero.")
    return errors


def validate_global_feature_artifacts(
    *,
    histograms: np.ndarray,
    capture_seconds: np.ndarray,
    color_distance: np.ndarray,
    time_distance: np.ndarray,
    global_dissimilarity: np.ndarray,
    records: Sequence[dict[str, Any]],
    config: GlobalFeatureConfig,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate all artifacts produced by PhotoCluster step 1."""
    config.validate()
    errors: list[str] = []
    warnings: list[str] = []

    histogram_matrix = np.asarray(histograms)
    times = np.asarray(capture_seconds)
    count = int(histogram_matrix.shape[0]) if histogram_matrix.ndim >= 1 else 0

    if histogram_matrix.ndim != 2:
        errors.append(
            f"hs_histograms must be 2-D, found {histogram_matrix.shape}."
        )
    elif histogram_matrix.shape[1] != config.feature_dimension:
        errors.append(
            "HS histogram dimension is "
            f"{histogram_matrix.shape[1]}; expected {config.feature_dimension}."
        )

    if expected_count is not None and count != expected_count:
        errors.append(f"Expected {expected_count} images, found {count}.")

    if times.shape != (count,):
        errors.append(
            f"capture_times shape is {times.shape}; expected {(count,)}."
        )
    elif not np.isfinite(times).all():
        errors.append("capture_times contains NaN or infinite values.")

    if histogram_matrix.ndim == 2 and histogram_matrix.shape[1] == config.feature_dimension:
        if not np.isfinite(histogram_matrix).all():
            errors.append("HS histograms contain NaN or infinite values.")
        if np.any(histogram_matrix < 0):
            errors.append("HS histograms contain negative values.")
        hue_sums = histogram_matrix[:, : config.hue_bins].sum(axis=1)
        saturation_sums = histogram_matrix[:, config.hue_bins :].sum(axis=1)
        if not np.allclose(hue_sums, 1.0, atol=1e-5):
            errors.append("Hue histograms are not independently L1-normalized.")
        if not np.allclose(saturation_sums, 1.0, atol=1e-5):
            errors.append(
                "Saturation histograms are not independently L1-normalized."
            )

    for name, matrix in (
        ("color_distance", np.asarray(color_distance)),
        ("time_distance", np.asarray(time_distance)),
        ("global_dissimilarity", np.asarray(global_dissimilarity)),
    ):
        errors.extend(_matrix_checks(name, matrix, count))

    if not errors:
        expected_combined = combine_feature_distances(
            color_distance,
            time_distance,
            color_weight=config.color_weight,
            time_weight=config.time_weight,
        )
        if not np.allclose(
            global_dissimilarity,
            expected_combined,
            atol=1e-6,
        ):
            errors.append(
                "global_dissimilarity does not match the configured weighted combination."
            )

    if len(records) != count:
        errors.append(
            f"Record count is {len(records)}; expected {count}."
        )
    else:
        rows = [record.get("row") for record in records]
        if rows != list(range(count)):
            errors.append("Record rows are not the ordered sequence 0..N-1.")
        image_ids = [record.get("image_id") for record in records]
        if len(image_ids) != len(set(image_ids)):
            errors.append("Duplicate image_id values were found.")

    if histogram_matrix.dtype != np.float32:
        warnings.append(
            f"Recommended histogram dtype is float32; found {histogram_matrix.dtype}."
        )

    upper = np.triu_indices(count, k=1) if count > 1 else (np.array([], dtype=int), np.array([], dtype=int))

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "image_count": count,
            "histogram_shape": list(histogram_matrix.shape),
            "histogram_dtype": str(histogram_matrix.dtype),
            "capture_time_span_seconds": (
                float(times.max() - times.min()) if times.size else None
            ),
            "color_distance_minimum": (
                float(np.asarray(color_distance)[upper].min())
                if count > 1 else 0.0
            ),
            "color_distance_maximum": (
                float(np.asarray(color_distance)[upper].max())
                if count > 1 else 0.0
            ),
            "time_distance_minimum": (
                float(np.asarray(time_distance)[upper].min())
                if count > 1 else 0.0
            ),
            "time_distance_maximum": (
                float(np.asarray(time_distance)[upper].max())
                if count > 1 else 0.0
            ),
            "global_dissimilarity_minimum": (
                float(np.asarray(global_dissimilarity)[upper].min())
                if count > 1 else 0.0
            ),
            "global_dissimilarity_maximum": (
                float(np.asarray(global_dissimilarity)[upper].max())
                if count > 1 else 0.0
            ),
            "matrices_are_symmetric": not any(
                "not symmetric" in error for error in errors
            ),
            "matrix_diagonals_are_zero": not any(
                "diagonal is not zero" in error for error in errors
            ),
            "records_are_sequential": (
                [record.get("row") for record in records]
                == list(range(len(records)))
            ),
        },
    }
