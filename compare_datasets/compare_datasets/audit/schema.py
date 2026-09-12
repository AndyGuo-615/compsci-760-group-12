"""CSV column schemas and writer for the audit exports."""

from __future__ import annotations

import csv
from pathlib import Path

MANIFEST_FIELDS = [
    "image_id", "image_path", "blood_group", "file_size", "file_format",
    "readable", "width", "height", "sha256", "phash64", "dhash64", "error",
]
EXACT_FIELDS = [
    "exact_group", "image_id", "image_path", "blood_group", "sha256",
    "group_size", "label_count", "label_conflict",
]
NEAR_FIELDS = [
    "image_id_a", "image_id_b", "image_path_a", "image_path_b", "label_a",
    "label_b", "phash_distance", "dhash_distance", "ssim", "related",
    "label_conflict",
]
GROUPED_FIELDS = [
    "image_id", "image_path", "blood_group", "group_id", "group_reason",
    "group_size", "group_label_count", "label_conflict",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
