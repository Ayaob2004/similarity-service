"""Minimal command-line interface for the production similarity engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from smart_gallery_similarity.engine import SmartGalleryError, analyze_gallery
from smart_gallery_similarity.production_config import ProductionConfigError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-gallery",
        description=(
            "Analyze an image gallery and return non-overlapping groups of similar "
            "asset IDs. The first run is full; later runs are incremental."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="Analyze or update one gallery.")
    analyze.add_argument("--input-dir", required=True, type=Path)
    analyze.add_argument("--gallery-id", required=True)
    analyze.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    analyze.add_argument("--manifest", type=Path, default=None)
    analyze.add_argument("--config", type=Path, default=None)
    analyze.add_argument("--recursive", action="store_true")
    analyze.add_argument(
        "--fresh",
        action="store_true",
        help="Force a safe full rebuild; old state is replaced only after success.",
    )
    analyze.add_argument(
        "--previews",
        action="store_true",
        help=(
            "Rebuild runtime/galleries/<gallery-id>/previews with one folder "
            "per current similarity group."
        ),
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command != "analyze":
        parser.print_help()
        return 0

    try:
        result = analyze_gallery(
            input_dir=args.input_dir,
            gallery_id=args.gallery_id,
            runtime_dir=args.runtime_dir,
            manifest_path=args.manifest,
            config_path=args.config,
            recursive=args.recursive,
            fresh=args.fresh,
            export_previews=args.previews,
        )
    except (SmartGalleryError, ProductionConfigError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"[OK] Mode: {result.mode}")
    print(f"[INFO] Images: {result.image_count}")
    print(f"[INFO] Groups: {result.group_count}")
    print(f"[INFO] Grouped images: {result.grouped_image_count}")
    print(f"[INFO] Ungrouped images: {result.ungrouped_image_count}")
    print(f"[INFO] State revision: {result.state_revision}")
    for warning in result.warnings:
        print(f"[WARNING] {warning}")
    print(f"[OK] Groups file: {result.groups_file}")
    if args.previews:
        print(f"[OK] Preview folders: {result.previews_directory}")
    print(f"[OK] State directory: {result.state_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
