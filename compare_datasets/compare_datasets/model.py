"""Data containers for the dataset comparison tool."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imagehash


@dataclass
class ImageRecord:
    path: Path
    rel: str
    name: str
    cls: str
    sha256: str = ""
    phash: imagehash.ImageHash | None = None
    dhash: imagehash.ImageHash | None = None
    ahash: imagehash.ImageHash | None = None
    error: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    file_format: str = ""
    readable: bool = False


@dataclass
class Settings:
    match_content: bool = False
    near_dup_threshold: int = 10
    dhash_threshold: int = 12
    ssim_threshold: float = 0.95
    top: int = 20
    workers: int = 1


@dataclass
class Comparison:
    root_a: Path
    root_b: Path
    records_a: list[ImageRecord]
    records_b: list[ImageRecord]
    matched: list[tuple[ImageRecord, ImageRecord]]
    only_a: list[ImageRecord]
    only_b: list[ImageRecord]
    ambiguous: list[str]
    content_groups: list[tuple[str, list[ImageRecord], list[ImageRecord]]]
    sha_mismatch_rows: list[list[object]]
    phash_rows: list[tuple[int, ImageRecord, ImageRecord]]
    dhash_dists: list[int]
    ahash_dists: list[int]
    phash_missing: int
    ssim_rows: list[tuple[float, ImageRecord, ImageRecord]]
    ssim_errors: int
    counts_a: Counter
    counts_b: Counter
    all_classes: list[str]
    errors: list[ImageRecord]
    runtime: float


@dataclass
class AuditResult:
    root: Path
    records: list[ImageRecord]
    manifest_rows: list[dict]
    exact_rows: list[dict]
    near_rows: list[dict]
    grouped_rows: list[dict]
    summary: dict
    runtime: float
    near_candidate_count: int = 0
    near_csv_written: bool = False
