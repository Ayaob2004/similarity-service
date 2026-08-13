"""Load and fingerprint the frozen Smart Gallery production configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

from smart_gallery_similarity.photocluster.combined_distance import CombinedDistanceConfig
from smart_gallery_similarity.photocluster.final_grouping import FinalGroupingConfig
from smart_gallery_similarity.photocluster.global_features import GlobalFeatureConfig
from smart_gallery_similarity.photocluster.initial_partition import InitialPartitionConfig
from smart_gallery_similarity.photocluster.sift_features import SIFTConfig
from smart_gallery_similarity.time_policy import TimeDistanceConfig


class ProductionConfigError(RuntimeError):
    """Raised when the production config is missing or invalid."""


@dataclass(frozen=True)
class ProductionPhotoClusterConfig:
    """All settings used by the production full/incremental pipeline."""

    version: int
    global_features: GlobalFeatureConfig
    time_distance: TimeDistanceConfig
    initial_partition: InitialPartitionConfig
    sift: SIFTConfig
    combined_distance: CombinedDistanceConfig
    final_grouping: FinalGroupingConfig

    def validate(self) -> None:
        if self.version < 1:
            raise ProductionConfigError("Config version must be at least 1.")
        self.global_features.validate()
        self.time_distance.validate()
        self.initial_partition.validate()
        self.sift.validate()
        self.combined_distance.validate()
        self.final_grouping.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "global_features": asdict(self.global_features),
            "time_distance": asdict(self.time_distance),
            "initial_partition": asdict(self.initial_partition),
            "sift": asdict(self.sift),
            "combined_distance": asdict(self.combined_distance),
            "final_grouping": asdict(self.final_grouping),
        }

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint(self.to_dict())

    def feature_fingerprint(self) -> str:
        return self._fingerprint(
            {
                "hs_histogram": {
                    "hue_bins": int(self.global_features.hue_bins),
                    "saturation_bins": int(self.global_features.saturation_bins),
                },
                "sift": asdict(self.sift),
            }
        )

    def pair_component_fingerprint(self) -> str:
        return self._fingerprint(
            {
                "algorithm": "sift-crosscheck-smr+bhattacharyya-v1",
                "feature_fingerprint": self.feature_fingerprint(),
            }
        )

    def pair_score_fingerprint(self) -> str:
        return self._fingerprint(
            {
                "component_fingerprint": self.pair_component_fingerprint(),
                "time_distance": asdict(self.time_distance),
                "combined_distance": asdict(self.combined_distance),
            }
        )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProductionConfigError(f"Missing or invalid config section: {key}")
    return value


def _time_distance_config(payload: dict[str, Any], *, version: int) -> TimeDistanceConfig:
    value = payload.get("time_distance")
    if value is None:
        if version <= 1:
            return TimeDistanceConfig(
                mode="collection_span",
                fixed_horizon_seconds=86_400.0,
            )
        raise ProductionConfigError("Missing config section: time_distance")
    if not isinstance(value, dict):
        raise ProductionConfigError("Invalid config section: time_distance")
    return TimeDistanceConfig(
        mode=str(value.get("mode", "")).strip(),
        fixed_horizon_seconds=float(value.get("fixed_horizon_seconds", 86_400.0)),
    )


def _config_from_payload(payload: object) -> ProductionPhotoClusterConfig:
    if not isinstance(payload, dict):
        raise ProductionConfigError("Production config root must be a JSON object.")

    try:
        version = int(payload.get("version", 1))
        global_section = _mapping(payload, "global_features")
        partition_section = _mapping(payload, "initial_partition")
        sift_section = _mapping(payload, "sift")
        combined_section = _mapping(payload, "combined_distance")
        grouping_section = _mapping(payload, "final_grouping")

        config = ProductionPhotoClusterConfig(
            version=version,
            global_features=GlobalFeatureConfig(
                hue_bins=int(global_section["hue_bins"]),
                saturation_bins=int(global_section["saturation_bins"]),
                color_weight=float(global_section["color_weight"]),
                time_weight=float(global_section["time_weight"]),
            ),
            time_distance=_time_distance_config(payload, version=version),
            initial_partition=InitialPartitionConfig(
                preference_quantile=float(partition_section["preference_quantile"]),
                damping=float(partition_section["damping"]),
                max_iter=int(partition_section["max_iter"]),
                convergence_iter=int(partition_section["convergence_iter"]),
                random_state=int(partition_section["random_state"]),
            ),
            sift=SIFTConfig(
                nfeatures=int(sift_section["nfeatures"]),
                n_octave_layers=int(sift_section["n_octave_layers"]),
                contrast_threshold=float(sift_section["contrast_threshold"]),
                edge_threshold=float(sift_section["edge_threshold"]),
                sigma=float(sift_section["sigma"]),
                max_edge=int(sift_section["max_edge"]),
            ),
            combined_distance=CombinedDistanceConfig(
                sift_weight=float(combined_section["sift_weight"]),
                color_weight=float(combined_section["color_weight"]),
                time_weight=float(combined_section["time_weight"]),
            ),
            final_grouping=FinalGroupingConfig(
                similarity_threshold=float(grouping_section["similarity_threshold"]),
                linkage=str(grouping_section["linkage"]),
                minimum_group_size=int(grouping_section["minimum_group_size"]),
            ),
        )
        config.validate()
    except ProductionConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionConfigError(f"Invalid production config: {exc}") from exc
    return config


def load_production_config(path: Path | None = None) -> ProductionPhotoClusterConfig:
    """Load an explicit JSON config or the config packaged with the library."""

    if path is None:
        source_name = "packaged photocluster.production.json"
        try:
            text = files("smart_gallery_similarity").joinpath(
                "config", "photocluster.production.json"
            ).read_text(encoding="utf-8")
        except OSError as exc:
            raise ProductionConfigError(
                f"Could not read {source_name}: {exc}"
            ) from exc
    else:
        config_path = Path(path).expanduser().resolve()
        source_name = str(config_path)
        if not config_path.is_file():
            raise ProductionConfigError(
                f"Production config does not exist: {config_path}"
            )
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProductionConfigError(
                f"Could not read production config '{config_path}': {exc}"
            ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProductionConfigError(
            f"Could not parse production config '{source_name}': {exc}"
        ) from exc
    return _config_from_payload(payload)
