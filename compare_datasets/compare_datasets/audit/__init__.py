"""Single-dataset audit pipeline: manifest, duplicates and grouping.

Ref: Proposal Section 7.1 (data manifest), Section 7.4 (exact duplicate
detection), Section 7.5 (near duplicate detection) and image-level grouping;
Survey Section VI.A (exact and near-duplicate detection).
"""

from .grouping import UnionFind, assign_groups
from .manifest import build_manifest_rows, exact_duplicate_rows
from .near import near_duplicate_candidates, score_near_pairs
from .parallel import score_near_pairs_streaming
from .runner import export_audit, run_audit
from .schema import (
    EXACT_FIELDS,
    GROUPED_FIELDS,
    MANIFEST_FIELDS,
    NEAR_FIELDS,
    _write_csv,
)
from .summary import build_summary
from .util import _hash_int, _log_progress, audit_label

__all__ = [
    "EXACT_FIELDS",
    "GROUPED_FIELDS",
    "MANIFEST_FIELDS",
    "NEAR_FIELDS",
    "UnionFind",
    "assign_groups",
    "audit_label",
    "build_manifest_rows",
    "build_summary",
    "exact_duplicate_rows",
    "export_audit",
    "near_duplicate_candidates",
    "run_audit",
    "score_near_pairs",
    "score_near_pairs_streaming",
]
