"""Scan / hash / match pipeline producing a Comparison result."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from .hashing import compute_hashes, compute_ssim, hamming
from .io_utils import find_images, load_gray, make_record
from .matching import match_by_content, match_records
from .model import Comparison, ImageRecord, Settings


def progress(label: str, i: int, n: int, step: int = 500) -> None:
    if n and (i % step == 0 or i == n):
        print(f"  {label}: {i}/{n}", file=sys.stderr)


def run_comparison(root_a: Path, root_b: Path, settings: Settings) -> Comparison:
    started = time.time()

    print(f"Scanning A: {root_a}", file=sys.stderr)
    files_a = find_images(root_a)
    print(f"  found {len(files_a)} images", file=sys.stderr)
    print(f"Scanning B: {root_b}", file=sys.stderr)
    files_b = find_images(root_b)
    print(f"  found {len(files_b)} images", file=sys.stderr)

    records_a = [make_record(p, root_a) for p in files_a]
    records_b = [make_record(p, root_b) for p in files_b]

    print("Hashing A (SHA-256 + perceptual)...", file=sys.stderr)
    for i, rec in enumerate(records_a, 1):
        compute_hashes(rec)
        progress("A", i, len(records_a))
    print("Hashing B (SHA-256 + perceptual)...", file=sys.stderr)
    for i, rec in enumerate(records_b, 1):
        compute_hashes(rec)
        progress("B", i, len(records_b))

    matched, only_a, only_b, ambiguous = match_records(records_a, records_b)
    print(f"Matched {len(matched)} file(s) by name", file=sys.stderr)

    content_groups: list[tuple[str, list[ImageRecord], list[ImageRecord]]] = []
    if settings.match_content:
        content_groups = match_by_content(records_a, records_b, key=lambda r: r.sha256)
        n_shared = len(content_groups)
        n_a_content = sum(len(g[1]) for g in content_groups)
        n_b_content = sum(len(g[2]) for g in content_groups)
        print(
            f"Content match (SHA-256 sort-merge): {n_shared} shared hash(es), "
            f"{n_a_content} A file(s), {n_b_content} B file(s)",
            file=sys.stderr,
        )

    sha_mismatch_rows: list[list[object]] = []
    for a, b in matched:
        if a.sha256 and b.sha256 and a.sha256 != b.sha256:
            sha_mismatch_rows.append([a.cls, a.name, a.sha256[:16], b.sha256[:16]])

    phash_rows: list[tuple[int, ImageRecord, ImageRecord]] = []
    dhash_dists: list[int] = []
    ahash_dists: list[int] = []
    phash_missing = 0
    for a, b in matched:
        if a.phash is None or b.phash is None:
            phash_missing += 1
            continue
        d_ph = hamming(a.phash, b.phash)
        phash_rows.append((d_ph, a, b))
        dhash_dists.append(hamming(a.dhash, b.dhash))
        ahash_dists.append(hamming(a.ahash, b.ahash))
    phash_rows.sort(key=lambda t: (-t[0], t[1].cls, t[1].name))

    ssim_rows: list[tuple[float, ImageRecord, ImageRecord]] = []
    ssim_errors = 0
    print("Computing SSIM on matched pairs...", file=sys.stderr)
    for i, (a, b) in enumerate(matched, 1):
        try:
            ga = load_gray(a.path)
            gb = load_gray(b.path)
            ssim_rows.append((compute_ssim(ga, gb), a, b))
        except Exception:  # noqa: BLE001
            ssim_errors += 1
        progress("SSIM", i, len(matched))
    ssim_rows.sort(key=lambda t: (t[0], t[1].cls, t[1].name))

    counts_a = Counter(r.cls for r in records_a)
    counts_b = Counter(r.cls for r in records_b)
    all_classes = sorted(set(counts_a) | set(counts_b))
    errors = [r for r in records_a + records_b if r.error]

    return Comparison(
        root_a=root_a,
        root_b=root_b,
        records_a=records_a,
        records_b=records_b,
        matched=matched,
        only_a=only_a,
        only_b=only_b,
        ambiguous=ambiguous,
        content_groups=content_groups,
        sha_mismatch_rows=sha_mismatch_rows,
        phash_rows=phash_rows,
        dhash_dists=dhash_dists,
        ahash_dists=ahash_dists,
        phash_missing=phash_missing,
        ssim_rows=ssim_rows,
        ssim_errors=ssim_errors,
        counts_a=counts_a,
        counts_b=counts_b,
        all_classes=all_classes,
        errors=errors,
        runtime=time.time() - started,
    )
