"""File discovery, record creation and loading helpers.

Ref: Proposal Section 6.3 (total image count, file names) and Section 7.1
(data manifest).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import HASH_CHUNK_SIZE, IMAGE_EXTS, SSIM_SIZE
from .model import ImageRecord


def find_images(root: Path) -> list[Path]:
    """Recursively collect every image file under *root*, sorted by path."""
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                found.append(Path(dirpath) / fn)
    return sorted(found)


def make_record(path: Path, root: Path) -> ImageRecord:
    """Build an ImageRecord (relative path, filename, class = parent folder)."""
    return ImageRecord(
        path=path,
        rel=str(path.relative_to(root)),
        name=path.name,
        cls=path.parent.name,
    )


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Stream a file and return its SHA-256 hex digest.

    Ref: Proposal Section 6.3 (SHA-256) and Section 7.4 (exact duplicates);
    Survey Section VI.A; NIST FIPS 180-4.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gray(path: Path, size: tuple[int, int] = SSIM_SIZE) -> np.ndarray:
    """Load an image as a fixed-size grayscale array for SSIM.

    Ref: Proposal Section 7.5 (image-similarity measures); Survey Section VI.A.
    """
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.float64)
