"""Manifest and exact-duplicate tables.

Ref: Proposal Section 7.1 (data manifest) and Section 7.4 (exact duplicate
detection).
"""

from __future__ import annotations

from collections import defaultdict

from ..model import ImageRecord
from .util import _hash_int, audit_label


def build_manifest_rows(records: list[ImageRecord]) -> list[dict]:
    rows = []
    for i, r in enumerate(records):
        rows.append({
            "image_id": i,
            "image_path": r.rel,
            "blood_group": audit_label(r),
            "file_size": r.file_size,
            "file_format": r.file_format,
            "readable": r.readable,
            "width": r.width,
            "height": r.height,
            "sha256": r.sha256,
            "phash64": f"{_hash_int(r.phash):016x}" if r.phash is not None else "",
            "dhash64": f"{_hash_int(r.dhash):016x}" if r.dhash is not None else "",
            "error": r.error,
        })
    return rows


def exact_duplicate_rows(records: list[ImageRecord]) -> list[dict]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.sha256:
            by_hash[r.sha256].append(i)

    ordered = sorted(h for h, ids in by_hash.items() if len(ids) > 1)
    rows: list[dict] = []
    for gi, h in enumerate(ordered, 1):
        ids = by_hash[h]
        labels = {audit_label(records[i]) for i in ids}
        for i in ids:
            rows.append({
                "exact_group": f"EXACT_{gi:05d}",
                "image_id": i,
                "image_path": records[i].rel,
                "blood_group": audit_label(records[i]),
                "sha256": h,
                "group_size": len(ids),
                "label_count": len(labels),
                "label_conflict": len(labels) > 1,
            })
    rows.sort(key=lambda row: (row["exact_group"], row["image_path"]))
    return rows
