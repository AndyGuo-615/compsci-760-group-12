"""Single-dataset audit report.

Ref: Proposal Sections 7.1 (manifest), 7.4 (exact duplicates) and 7.5 (near
duplicates) plus image-level grouping.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from ..hashing import SSIM_BACKEND
from ..model import AuditResult, Settings
from .formatting import md_table


def audit_conclusion(summary: dict) -> str:
    if summary["exact_duplicate_groups"] == 0 and summary["accepted_near_duplicate_pairs"] == 0:
        return "No exact or accepted near-duplicate images were found."
    parts = []
    if summary["exact_duplicate_groups"]:
        parts.append(f"{summary['exact_duplicate_groups']} exact duplicate group(s)")
    if summary["accepted_near_duplicate_pairs"]:
        parts.append(f"{summary['accepted_near_duplicate_pairs']} accepted near-duplicate pair(s)")
    msg = "Found " + " and ".join(parts) + "."
    if summary["cross_label_groups"]:
        msg += (
            f" **{summary['cross_label_groups']} group(s) span more than one class**"
            " — check for label leakage."
        )
    return msg


def build_audit_report(result: AuditResult, settings: Settings) -> str:
    manifest = result.manifest_rows
    exact = result.exact_rows
    near = result.near_rows
    grouped = result.grouped_rows
    s = result.summary
    top = settings.top

    L: list[str] = []
    L.append("# Dataset Audit Report")
    L.append("")
    L.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- **Dataset:** `{result.root}`")
    L.append("- **Mode:** single-dataset audit (Proposal §7.1 / §7.4 / §7.5 + grouping)")
    L.append(f"- **SSIM backend:** {SSIM_BACKEND}")
    L.append(f"- **pHash threshold:** Hamming <= {settings.near_dup_threshold}")
    L.append(f"- **dHash threshold:** Hamming <= {settings.dhash_threshold}")
    L.append(f"- **SSIM threshold:** {settings.ssim_threshold}")
    L.append(f"- **Runtime:** {result.runtime:.1f} s")
    L.append("")
    L.append("---")
    L.append("")

    L.append("## 1. Manifest (Proposal §7.1)")
    L.append("")
    L.append(f"- Total images: **{s['total_images']}**")
    L.append(f"- Readable: **{s['readable_images']}**; unreadable: **{s['unreadable_images']}**")
    L.append("")
    L += md_table(["Class", "Images"], [[k, v] for k, v in s["class_counts"].items()])
    L.append("")
    formats = Counter(r["file_format"] for r in manifest)
    L.append("Formats: " + ", ".join(f"{k} ({v})" for k, v in sorted(formats.items())))
    L.append("")

    L.append("## 2. Exact Duplicates (Proposal §7.4)")
    L.append("")
    L.append(f"- Exact duplicate groups: **{s['exact_duplicate_groups']}**")
    L.append(f"- Images in exact duplicate groups: **{s['images_in_exact_duplicate_groups']}**")
    L.append(f"- Exact groups spanning >1 class: **{s['exact_cross_label_groups']}**")
    L.append("")
    if exact:
        by_group: dict[str, list[dict]] = defaultdict(list)
        for row in exact:
            by_group[row["exact_group"]].append(row)
        rows = []
        for gid, items in list(by_group.items())[:top]:
            rows.append([
                gid,
                items[0]["group_size"],
                items[0]["label_count"],
                "yes" if items[0]["label_conflict"] else "no",
                ", ".join(i["image_path"] for i in items[:5]),
            ])
        L.append(f"### Exact duplicate groups (first {len(rows)} of {s['exact_duplicate_groups']})")
        L.append("")
        L += md_table(["Group", "Size", "Classes", "Conflict", "Files (first 5)"], rows)
        L.append("")

    related = [r for r in near if r["related"]]
    L.append("## 3. Near Duplicates (Proposal §7.5)")
    L.append("")
    L.append(f"- Hash-level candidate pairs: **{s['near_candidate_pairs']}**")
    L.append(f"- Accepted pairs (SSIM >= {settings.ssim_threshold}): **{s['accepted_near_duplicate_pairs']}**")
    L.append(f"- Accepted pairs spanning >1 class: **{s['accepted_near_cross_label_pairs']}**")
    if s.get("near_pairs_skipped"):
        L.append(
            "- Pairs not scored (load/SSIM error or failed worker chunk): "
            f"**{s['near_pairs_skipped']}**"
        )
    if s.get("near_failed_chunks"):
        L.append(f"- Worker chunks failed: **{s['near_failed_chunks']}**")
    L.append("")
    if near:
        rows = [[
            r["label_a"], r["label_b"], r["image_path_a"], r["image_path_b"],
            r["phash_distance"], r["dhash_distance"], f"{r['ssim']:.4f}",
            "yes" if r["related"] else "no",
        ] for r in near[:top]]
        L.append(f"### Candidate pairs, highest SSIM first (first {len(rows)} of {s['near_candidate_pairs']})")
        L.append("")
        L += md_table(
            ["Class A", "Class B", "File A", "File B", "pHash", "dHash", "SSIM", "Related"],
            rows,
        )
        L.append("")

    L.append("## 4. Groups (union-find)")
    L.append("")
    L.append(f"- Total groups: **{s['final_groups']}**")
    L.append(f"- Non-singleton groups: **{s['non_singleton_groups']}**")
    L.append(f"- Groups spanning >1 class: **{s['cross_label_groups']}**")
    L.append("")
    if grouped:
        rows = []
        seen: set[str] = set()
        for row in grouped:
            if row["group_size"] > 1 and row["group_id"] not in seen:
                seen.add(row["group_id"])
                rows.append([
                    row["group_id"], row["group_size"], row["group_label_count"],
                    row["group_reason"], "yes" if row["label_conflict"] else "no",
                ])
            if len(rows) >= top:
                break
        if rows:
            L.append(f"### Non-singleton groups (first {len(rows)})")
            L.append("")
            L += md_table(["Group", "Size", "Classes", "Reason", "Conflict"], rows)
            L.append("")

    L.append("## 5. Conclusion")
    L.append("")
    L.append(audit_conclusion(s))
    L.append("")
    errors = [r for r in manifest if r["error"]]
    if errors:
        L.append("### Files that could not be processed")
        L.append("")
        L.append("```")
        for r in errors[:top]:
            L.append(f"{r['image_path']}  ->  {r['error']}")
        L.append("```")
        L.append("")

    return "\n".join(L) + "\n"
