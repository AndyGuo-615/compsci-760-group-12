"""Shared constants and defaults."""

from __future__ import annotations

from pathlib import Path

# Config image whitelist
IMAGE_EXTS = {
    ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".webp", ".pgm", ".ppm", ".gif",
}

# Config Result output location
DEFAULT_OUTPUT = Path("dataset_comparison_report.txt")
DEFAULT_AUDIT_OUTPUT = Path("dataset_audit_report.txt")
DEFAULT_NEAR_DUP_THRESHOLD = 10
DEFAULT_DHASH_THRESHOLD = 12
DEFAULT_SSIM_THRESHOLD = 0.95
DEFAULT_TOP = 20
DEFAULT_WORKERS = 1

SSIM_SIZE = (128, 128)
HASH_CHUNK_SIZE = 1 << 20
AUDIT_HAMMING_CHUNK = 64
