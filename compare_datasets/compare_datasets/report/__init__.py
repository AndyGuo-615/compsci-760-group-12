"""Markdown report assembly."""

from .audit_report import audit_conclusion, build_audit_report
from .compare_report import build_report, classify_verdict
from .formatting import md_table, pct

__all__ = [
    "audit_conclusion",
    "build_audit_report",
    "build_report",
    "classify_verdict",
    "md_table",
    "pct",
]
