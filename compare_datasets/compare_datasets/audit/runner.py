"""Audit orchestration and CSV/JSON export."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..hashing import compute_hashes
from ..io_utils import find_images, make_record
from ..model import AuditResult, Settings
from .grouping import assign_groups
from .manifest import build_manifest_rows, exact_duplicate_rows
from .near import near_duplicate_candidates, score_near_pairs
from .parallel import score_near_pairs_streaming
from .schema import (
    EXACT_FIELDS,
    GROUPED_FIELDS,
    MANIFEST_FIELDS,
    NEAR_FIELDS,
    _write_csv,
)
from .summary import build_summary


def run_audit(
    root: Path,
    settings: Settings,
    manifest_only: bool = False,
    export_dir: Path | None = None,
) -> AuditResult:
    started = time.time()

    print(f"Scanning: {root}", file=sys.stderr)
    files = find_images(root)
    print(f"  found {len(files)} images", file=sys.stderr)
    records = [make_record(p, root) for p in files]

    print("Hashing (SHA-256 + perceptual)...", file=sys.stderr)
    for i, rec in enumerate(records, 1):
        compute_hashes(rec)
        if i % 500 == 0 or i == len(records):
            print(f"  {i}/{len(records)}", file=sys.stderr)

    manifest_rows = build_manifest_rows(records)
    exact_rows = exact_duplicate_rows(records)
    near_rows: list[dict] = []
    grouped_rows: list[dict] = []
    near_candidate_count = 0
    near_pairs_skipped = 0
    near_failed_chunks = 0
    near_csv_written = False
    workers = settings.workers if settings.workers > 0 else (os.cpu_count() or 1)

    if not manifest_only:
        print("Searching near-duplicates (vectorised Hamming)...", file=sys.stderr)
        pairs = near_duplicate_candidates(
            records, settings.near_dup_threshold, settings.dhash_threshold
        )
        print(f"  {len(pairs)} hash-level candidate pair(s)", file=sys.stderr)
        if workers > 1 and export_dir is not None:
            (
                near_rows,
                near_candidate_count,
                near_pairs_skipped,
                near_failed_chunks,
            ) = score_near_pairs_streaming(
                records,
                pairs,
                settings.ssim_threshold,
                workers,
                export_dir / "near_duplicate_pairs.csv",
                export_dir / "_near_shards",
            )
            near_csv_written = True
        else:
            near_rows, near_pairs_skipped = score_near_pairs(
                records, pairs, settings.ssim_threshold
            )
            near_candidate_count = len(pairs)
        print("Grouping (union-find)...", file=sys.stderr)
        grouped_rows = assign_groups(records, exact_rows, near_rows)

    runtime = time.time() - started
    summary = build_summary(
        records,
        manifest_rows,
        exact_rows,
        near_rows,
        grouped_rows,
        {
            "phash_threshold": settings.near_dup_threshold,
            "dhash_threshold": settings.dhash_threshold,
            "ssim_threshold": settings.ssim_threshold,
            "manifest_only": manifest_only,
            "workers": workers,
        },
        runtime,
        near_candidate_count=near_candidate_count,
        near_pairs_skipped=near_pairs_skipped,
        near_failed_chunks=near_failed_chunks,
    )
    return AuditResult(
        root=root,
        records=records,
        manifest_rows=manifest_rows,
        exact_rows=exact_rows,
        near_rows=near_rows,
        grouped_rows=grouped_rows,
        summary=summary,
        runtime=runtime,
        near_candidate_count=near_candidate_count,
        near_csv_written=near_csv_written,
    )


def export_audit(result: AuditResult, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(export_dir / "manifest.csv", MANIFEST_FIELDS, result.manifest_rows)
    _write_csv(export_dir / "exact_duplicates.csv", EXACT_FIELDS, result.exact_rows)
    if not result.near_csv_written:
        _write_csv(export_dir / "near_duplicate_pairs.csv", NEAR_FIELDS, result.near_rows)
    _write_csv(export_dir / "grouped_manifest.csv", GROUPED_FIELDS, result.grouped_rows)
    (export_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
