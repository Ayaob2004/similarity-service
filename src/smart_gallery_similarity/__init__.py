"""Production image-similarity engine for Smart Gallery."""

from smart_gallery_similarity.engine import (
    GalleryAnalysisResult,
    SimilarityGroup,
    SmartGalleryError,
    analyze_gallery,
)

__all__ = [
    "GalleryAnalysisResult",
    "SimilarityGroup",
    "SmartGalleryError",
    "analyze_gallery",
]

__version__ = "1.0.1"
