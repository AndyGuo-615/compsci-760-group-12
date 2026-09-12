"""Shared helpers for the single-dataset audit pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..model import ImageRecord

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(values: np.ndarray) -> np.ndarray:
    """Population count of a uint64 array (bitwise_count, with a table fallback)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return _POPCOUNT_TABLE[values.view(np.uint8)].sum(axis=-1)


def _hash_int(h) -> int:
    """Return the integer value of an imagehash object (via its hex string)."""
    return int(str(h), 16)


def _log_progress(
    label: str, done: int, total: int, started: float, extra: str = ""
) -> None:
    """Print a one-line progress log to stderr (count, %, elapsed, ETA).

    ``started`` is a ``time.time()`` timestamp; ETA is extrapolated from the
    average rate so far.  ``extra`` is appended verbatim (e.g. pair counts).
    """
    elapsed = time.time() - started
    frac = done / total if total else 1.0
    eta = elapsed / frac - elapsed if frac > 0 else 0.0
    line = (
        f"  {label}: {done}/{total} ({100.0 * frac:5.1f}%) "
        f"elapsed {elapsed:6.1f}s eta {eta:6.1f}s"
    )
    if extra:
        line += f" - {extra}"
    print(line, file=sys.stderr)


def audit_label(rec: ImageRecord) -> str:
    """Label = first path component under the dataset root (matches Andy's rule)."""
    parts = Path(rec.rel).parts
    return parts[0] if len(parts) > 1 else rec.cls
