"""Parallel SSIM scoring with per-worker shard CSVs (bounded memory).

Each worker writes every scored row to its own shard file; the parent merges
the shards and keeps only the accepted (``related``) rows in memory, so the
full candidate table never has to fit in RAM.
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..hashing import compute_ssim
from ..io_utils import load_gray
from ..model import ImageRecord
from .schema import NEAR_FIELDS, _write_csv
from .util import _log_progress, audit_label

_SCORE_PATHS: list[str] = []
_SCORE_RELS: list[str] = []
_SCORE_LABELS: list[str] = []


def _init_score_worker(paths: list[str], rels: list[str], labels: list[str]) -> None:
    """ProcessPool initializer: share the per-image lookup tables once per worker."""
    global _SCORE_PATHS, _SCORE_RELS, _SCORE_LABELS
    _SCORE_PATHS = paths
    _SCORE_RELS = rels
    _SCORE_LABELS = labels


def _score_chunk(
    chunk: list[tuple[int, int, int, int]],
    ssim_threshold: float,
    shard_path: str,
) -> tuple[str, int, int, list[dict]]:
    """Score one chunk of pairs and write every row to *shard_path*.

    Returns ``(shard_path, rows_written, related_written, related_rows)`` so the
    parent never has to hold the full candidate table in memory.
    """
    rows: list[dict] = []
    related: list[dict] = []
    for a, b, d_ph, d_dh in chunk:
        try:
            s = compute_ssim(load_gray(_SCORE_PATHS[a]), load_gray(_SCORE_PATHS[b]))
        except Exception:  # noqa: BLE001 - skip unreadable pair
            continue
        row = {
            "image_id_a": a,
            "image_id_b": b,
            "image_path_a": _SCORE_RELS[a],
            "image_path_b": _SCORE_RELS[b],
            "label_a": _SCORE_LABELS[a],
            "label_b": _SCORE_LABELS[b],
            "phash_distance": d_ph,
            "dhash_distance": d_dh,
            "ssim": round(s, 6),
            "related": s >= ssim_threshold,
            "label_conflict": _SCORE_LABELS[a] != _SCORE_LABELS[b],
        }
        rows.append(row)
        if row["related"]:
            related.append(row)
    _write_csv(Path(shard_path), NEAR_FIELDS, rows)
    return shard_path, len(rows), len(related), related


def score_near_pairs_streaming(
    records: list[ImageRecord],
    pairs: list[tuple[int, int, int, int]],
    ssim_threshold: float,
    workers: int,
    near_csv_path: Path,
    shard_dir: Path,
    log: bool = True,
) -> tuple[list[dict], int]:
    """Parallel SSIM scoring that streams all rows to *near_csv_path*.

    Every worker writes its own shard CSV; the parent merges the shards and keeps
    only the accepted (``related``) rows in memory.  Returns
    ``(related_rows_sorted_by_ssim, total_candidate_count)``.
    """
    total = len(pairs)
    started = time.time()
    near_csv_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if log:
        print(
            f"  SSIM: verifying {total} candidate pair(s) with {workers} worker(s)...",
            file=sys.stderr,
        )
    if total == 0:
        _write_csv(near_csv_path, NEAR_FIELDS, [])
        return [], 0

    paths = [str(r.path) for r in records]
    rels = [r.rel for r in records]
    labels = [audit_label(r) for r in records]

    n_chunks = max(1, workers * 4)
    chunk_size = max(1, (total + n_chunks - 1) // n_chunks)
    chunks = [pairs[i:i + chunk_size] for i in range(0, total, chunk_size)]
    related: list[dict] = []
    shard_paths: list[str] = []
    scored = 0
    n_jobs = len(chunks)
    log_every = max(1, n_jobs // 20)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_score_worker,
        initargs=(paths, rels, labels),
    ) as ex:
        futures = [
            ex.submit(
                _score_chunk,
                chunk,
                ssim_threshold,
                str(shard_dir / f"near_shard_{i:06d}.csv"),
            )
            for i, chunk in enumerate(chunks)
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                shard_path, n_rows, _n_related, rel_rows = fut.result()
            except Exception as exc:  # noqa: BLE001 - keep other workers going
                if log:
                    print(f"  SSIM: worker failed: {exc}", file=sys.stderr)
                continue
            shard_paths.append(shard_path)
            related.extend(rel_rows)
            scored += n_rows
            if log and (done % log_every == 0 or done == n_jobs):
                _log_progress("SSIM", done, n_jobs, started, f"{scored} scored")

    with open(near_csv_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=NEAR_FIELDS)
        writer.writeheader()
        for shard_path in shard_paths:
            shard = Path(shard_path)
            if not shard.exists():
                continue
            with open(shard, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    writer.writerow(row)
            shard.unlink(missing_ok=True)
    try:
        shard_dir.rmdir()
    except OSError:
        pass
    if log:
        print(f"  SSIM: merged {len(shard_paths)} shard(s) -> {near_csv_path}", file=sys.stderr)

    related.sort(key=lambda r: -r["ssim"])
    return related, total
