"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import export_audit, run_audit
from .config import (
    DEFAULT_AUDIT_OUTPUT,
    DEFAULT_DHASH_THRESHOLD,
    DEFAULT_NEAR_DUP_THRESHOLD,
    DEFAULT_OUTPUT,
    DEFAULT_SSIM_THRESHOLD,
    DEFAULT_TOP,
    DEFAULT_WORKERS,
)
from .manual import print_user_manual
from .model import Settings
from .pipeline import run_comparison
from .report import build_audit_report, build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two image datasets, or audit one dataset for duplicates"
    )
    parser.add_argument("path_a", help="dataset directory")
    parser.add_argument(
        "path_b", nargs="?",
        help="second dataset directory (compare mode only)",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="audit a single dataset: manifest, exact/near duplicates, grouping",
    )
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="(audit) stop after the manifest and exact-duplicate detection",
    )
    parser.add_argument(
        "--export-dir", default=None,
        help="(audit) directory for CSV/JSON exports, default: <report name>_files",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="output file (Markdown text inside); default depends on mode",
    )
    parser.add_argument(
        "--near-dup-threshold", type=int, default=DEFAULT_NEAR_DUP_THRESHOLD,
        help="pHash Hamming distance <= this is a near-duplicate candidate (default 10)",
    )
    parser.add_argument(
        "--dhash-threshold", type=int, default=DEFAULT_DHASH_THRESHOLD,
        help="(audit) dHash Hamming distance <= this confirms a candidate (default 12)",
    )
    parser.add_argument(
        "--ssim-threshold", type=float, default=DEFAULT_SSIM_THRESHOLD,
        help="SSIM >= this confirms a related pair (default 0.95)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="(audit) parallel worker processes for the SSIM stage; 0 = all CPU cores (default 1)",
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP,
        help="max rows listed in detail tables (default 20)",
    )
    parser.add_argument(
        "--match-content", action="store_true",
        help="(compare) also match by content (exact SHA-256 sort-merge), catching renamed duplicates",
    )
    return parser


def _write_report(out_path: Path, report: str) -> None:
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}", file=sys.stderr)


def _run_audit(args: argparse.Namespace) -> int:
    if args.path_b:
        sys.exit("--audit takes a single dataset; remove the second path.")

    root = Path(args.path_a).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    settings = Settings(
        match_content=False,
        near_dup_threshold=args.near_dup_threshold,
        dhash_threshold=args.dhash_threshold,
        ssim_threshold=args.ssim_threshold,
        top=args.top,
        workers=args.workers,
    )

    out_path = Path(args.output) if args.output else DEFAULT_AUDIT_OUTPUT
    export_dir = (
        Path(args.export_dir)
        if args.export_dir
        else out_path.with_name(out_path.stem + "_files")
    ).expanduser()

    result = run_audit(root, settings, manifest_only=args.manifest_only, export_dir=export_dir)
    report = build_audit_report(result, settings)
    _write_report(out_path, report)
    export_audit(result, export_dir)
    print(f"Exports written to {export_dir.resolve()}", file=sys.stderr)
    if result.summary.get("near_failed_chunks"):
        print(
            f"WARNING: {result.summary['near_failed_chunks']} SSIM worker chunk(s) failed; "
            "some candidate pairs were not scored (see the report).",
            file=sys.stderr,
        )
        return 1
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    if not args.path_b:
        sys.exit("Compare mode needs two dataset paths (or pass --audit for one dataset).")

    root_a = Path(args.path_a).expanduser().resolve()
    root_b = Path(args.path_b).expanduser().resolve()
    for root in (root_a, root_b):
        if not root.is_dir():
            sys.exit(f"Not a directory: {root}")

    settings = Settings(
        match_content=args.match_content,
        near_dup_threshold=args.near_dup_threshold,
        dhash_threshold=args.dhash_threshold,
        ssim_threshold=args.ssim_threshold,
        top=args.top,
    )
    comp = run_comparison(root_a, root_b, settings)
    report = build_report(comp, settings)

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    _write_report(out_path, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        print_user_manual()
        return 0

    args = build_parser().parse_args(argv)
    if args.audit:
        return _run_audit(args)
    return _run_compare(args)
