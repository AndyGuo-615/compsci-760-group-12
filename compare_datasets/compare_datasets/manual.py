"""Barebone user manual printed when the CLI is run with no arguments."""

from __future__ import annotations

# Barebone user manual printed when the CLI is run with no arguments.
def print_user_manual() -> None:
    print(
        """compare_datasets.py - compare two datasets, or audit one dataset

USAGE
  # compare two datasets (Section 6.3 checks)
  python run_compare.py PATH_A PATH_B [options]

  # audit a single dataset for duplicates + grouping (Sections 7.1 / 7.4 / 7.5)
  python run_compare.py PATH_A --audit [options]

  PATH_A, PATH_B        dataset directories.  In compare mode their internal
                        structure may differ; files are matched by filename.

OPTIONS
  --audit               audit ONE dataset: manifest, exact duplicates, near
                        duplicates and union-find grouping
  --manifest-only       (audit) stop after the manifest + exact duplicates
  --export-dir DIR      (audit) directory for CSV/JSON exports
                        default: <report name>_files/
  -o, --output FILE     report file (Markdown text inside)
                        default: dataset_comparison_report.txt (compare mode)
                                 dataset_audit_report.txt (audit mode)
  --match-content       (compare) also match by content (exact SHA-256
                        sort-merge) to catch renamed / byte-identical duplicates
  --near-dup-threshold N
                        perceptual-hash (pHash) Hamming distance <= N counts as
                        a near-duplicate candidate.  default: 10
  --dhash-threshold N   (audit) dHash Hamming distance <= N to confirm a
                        candidate.  default: 12
  --ssim-threshold S    SSIM >= S confirms a related pair.  default: 0.95
  --workers N           (audit) parallel worker processes for the SSIM stage;
                        0 = all CPU cores.  default: 1
  --top N               max rows in detail tables.  default: 20
  -h, --help            show this help

COMPARE CHECKS (Section 6.3)
  1. total image count      4. SHA-256 hashes (exact)
  2. images per class       5. perceptual hashes (pHash / dHash / aHash)
  3. file names             6. image similarity (SSIM)

AUDIT CHECKS (Sections 7.1 / 7.4 / 7.5)
  1. manifest               3. near duplicates (vectorised Hamming + SSIM)
  2. exact duplicates       4. grouping (union-find)

EXAMPLES
  python run_compare.py ./datasetA ./datasetB
  python run_compare.py ./datasetA ./datasetB -o report.txt --match-content
  python run_compare.py ./dataset --audit --export-dir audit_out
  python run_compare.py ./dataset --audit --manifest-only
"""
    )
