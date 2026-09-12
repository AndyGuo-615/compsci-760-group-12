"""Perceptual hashing and image similarity.

Ref: Proposal Section 7.5 (near-duplicate detection, image-similarity measures);
Survey Section VI.A (exact and near-duplicate detection); Zauner; Wang et al.
"""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image

from .io_utils import sha256_file
from .model import ImageRecord

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    SSIM_BACKEND = "scikit-image (structural_similarity)"
except ImportError:  # scikit-image is optional
    _skimage_ssim = None
    SSIM_BACKEND = "numpy fallback (normalized cross-correlation)"


def compute_hashes(rec: ImageRecord) -> None:
    """Populate file metadata, SHA-256 and perceptual hashes.

    Ref: Proposal Section 7.1 (manifest), Section 7.4 (exact duplicates) and
    Section 7.5 (near-duplicate detection).
    """
    rec.file_format = rec.path.suffix.lower().lstrip(".")
    try:
        rec.file_size = rec.path.stat().st_size
    except OSError:
        rec.file_size = 0
    try:
        rec.sha256 = sha256_file(rec.path)
    except OSError as exc:
        rec.error = f"SHA-256 failed: {exc}"
        return
    try:
        with Image.open(rec.path) as im:
            rec.width, rec.height = im.size
            im = im.convert("L")
            rec.phash = imagehash.phash(im)
            rec.dhash = imagehash.dhash(im)
            rec.ahash = imagehash.average_hash(im)
            rec.readable = True
    except Exception as exc:  # noqa: BLE001 - report any decode failure
        rec.error = f"perceptual hash failed: {exc}"


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (SSIM), with a numpy cross-correlation fallback."""
    if _skimage_ssim is not None:
        return float(_skimage_ssim(a, b, data_range=255.0))
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((a * b).sum() / denom)


def hamming(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    """Hamming distance between two imagehash objects."""
    return int(h1 - h2)
