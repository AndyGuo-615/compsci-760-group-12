"""Near-duplicate candidate search and serial SSIM scoring.

Ref: Proposal Section 7.5 (near duplicate detection); Survey Section VI.A
(exact and near-duplicate detection).

The candidate search is vectorised with NumPy (bounded memory) rather than a
Python double loop, so it stays usable on datasets of tens of thousands of
images.  It is still an all-pairs Hamming search; it does not construct the
O(n^2) pair matrix at once.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ..config import AUDIT_HAMMING_CHUNK
from ..hashing import compute_ssim, hamming
from ..io_utils import load_gray
from ..model import ImageRecord
from .util import _hash_int, _log_progress, audit_label


def near_duplicate_candidates(
    records: list[ImageRecord],
    phash_threshold: int,
    dhash_threshold: int,
    chunk: int = AUDIT_HAMMING_CHUNK,
    log: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Vectorised all-pairs pHash + dHash Hamming search (bounded memory).

    Returns ``(i, j, phash_distance, dhash_distance)`` for i < j where the pHash
    distance is within *phash_threshold* and the dHash distance is within
    *dhash_threshold*.  Exact byte-duplicates (same SHA-256) are excluded.

    When *log* is true, prints progress (chunk count, elapsed time, ETA and the
    running candidate-pair count) to stderr.
    """
    index = [i for i, r in enumerate(records) if r.phash is not None and r.dhash is not None]
    if len(index) < 2:
        if log:
            print(f"  near-dup: only {len(index)} hashable image(s), skipping", file=sys.stderr)
        return []

    values = np.array([_hash_int(records[i].phash) for i in index], dtype=np.uint64)
    n = len(index)
    n_chunks = (n + chunk - 1) // chunk
    log_every = max(1, n_chunks // 20)
    pairs: list[tuple[int, int, int, int]] = []
    started = time.time()
    if log:
        print(
            f"  near-dup: comparing {n} images in {n_chunks} chunk(s) of {chunk} "
            f"(pHash<={phash_threshold}, dHash<={dhash_threshold})",
            file=sys.stderr,
        )

    for ci, start in enumerate(range(0, n, chunk), 1):
        end = min(start + chunk, n)
        block = values[start:end]
        dist = np.bitwise_count(block[:, None] ^ values[None, :])
        for local in range(end - start):
            gi = start + local
            row = dist[local, gi + 1:]
            js = np.nonzero(row <= phash_threshold)[0]
            if js.size == 0:
                continue
            a = index[gi]
            for offset in js:
                j = gi + 1 + int(offset)
                b = index[j]
                if records[a].sha256 and records[a].sha256 == records[b].sha256:
                    continue
                d_dhash = hamming(records[a].dhash, records[b].dhash)
                if d_dhash > dhash_threshold:
                    continue
                pairs.append((a, b, int(row[int(offset)]), d_dhash))
        if log and (ci % log_every == 0 or ci == n_chunks):
            _log_progress(
                "near-dup", ci, n_chunks, started, f"{len(pairs)} candidate pair(s)"
            )
    return pairs


def score_near_pairs(
    records: list[ImageRecord],
    pairs: list[tuple[int, int, int, int]],
    ssim_threshold: float,
    log: bool = True,
) -> tuple[list[dict], int]:
    """Confirm candidate pairs with SSIM and record distances.

    Returns ``(rows, skipped)`` where *skipped* is the number of candidate pairs
    whose images could not be loaded or scored - never dropped silently.

    When *log* is true, prints progress (elapsed time, ETA and the running
    accepted-pair count) to stderr.
    """
    rows: list[dict] = []
    skipped = 0
    total = len(pairs)
    log_every = max(1, total // 20)
    started = time.time()
    if log:
        print(f"  SSIM: verifying {total} candidate pair(s)...", file=sys.stderr)

    for idx, (a, b, d_ph, d_dh) in enumerate(pairs, 1):
        try:
            s = compute_ssim(load_gray(records[a].path), load_gray(records[b].path))
        except Exception:  # noqa: BLE001 - count the unreadable pair, never silent
            skipped += 1
            if log and skipped <= 5:
                print(
                    f"  SSIM: skipped {records[a].rel} <-> {records[b].rel}",
                    file=sys.stderr,
                )
            if log and (idx % log_every == 0 or idx == total):
                _log_progress(
                    "SSIM", idx, total, started, f"{len(rows)} scored, {skipped} skipped"
                )
            continue
        rows.append({
            "image_id_a": a,
            "image_id_b": b,
            "image_path_a": records[a].rel,
            "image_path_b": records[b].rel,
            "label_a": audit_label(records[a]),
            "label_b": audit_label(records[b]),
            "phash_distance": d_ph,
            "dhash_distance": d_dh,
            "ssim": round(s, 6),
            "related": s >= ssim_threshold,
            "label_conflict": audit_label(records[a]) != audit_label(records[b]),
        })
        if log and (idx % log_every == 0 or idx == total):
            _log_progress(
                "SSIM", idx, total, started, f"{len(rows)} scored, {skipped} skipped"
            )
    rows.sort(key=lambda r: (not r["related"], -r["ssim"]))
    if log and skipped:
        print(f"  SSIM: skipped {skipped} pair(s) (load/SSIM error)", file=sys.stderr)
    return rows, skipped
