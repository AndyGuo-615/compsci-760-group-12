"""Two-dataset comparison report (Proposal Section 6.3 checks).

Ref: Proposal Section 6.3 (Krishna GitHub dataset comparison); Survey
Section VI.A (exact and near-duplicate detection).
"""

from __future__ import annotations

from datetime import datetime

from ..hashing import SSIM_BACKEND
from ..model import Comparison, ImageRecord, Settings
from .formatting import md_table, pct


def classify_verdict(
    n_a: int,
    n_b: int,
    only_a: list[ImageRecord],
    only_b: list[ImageRecord],
    sha_mismatches: int,
) -> str:
    if sha_mismatches:
        return (
            "The two datasets overlap but are **not byte-identical** "
            f"({sha_mismatches} matched file(s) differ by SHA-256). Treat with care."
        )
    if not only_a and not only_b:
        return "The two datasets contain **the same files, byte-identical** (identical sets)."
    if not only_a and only_b:
        return (
            "**Dataset A is a strict byte-identical subset of Dataset B.** "
            f"B = A + {len(only_b)} extra file(s) (all A files match exactly)."
        )
    if only_a and not only_b:
        return (
            "**Dataset B is a strict byte-identical subset of Dataset A.** "
            f"A = B + {len(only_a)} extra file(s) (all B files match exactly)."
        )
    return (
        "The two datasets **partially overlap**: "
        f"{len(only_a)} file(s) only in A, {len(only_b)} file(s) only in B, "
        "and the shared files are byte-identical."
    )


def build_report(comp: Comparison, settings: Settings) -> str:
    records_a = comp.records_a
    records_b = comp.records_b
    matched = comp.matched
    only_a = comp.only_a
    only_b = comp.only_b
    ambiguous = comp.ambiguous
    sha_mismatch_rows = comp.sha_mismatch_rows
    phash_rows = comp.phash_rows
    dhash_dists = comp.dhash_dists
    ahash_dists = comp.ahash_dists
    phash_missing = comp.phash_missing
    ssim_rows = comp.ssim_rows
    ssim_errors = comp.ssim_errors
    counts_a = comp.counts_a
    counts_b = comp.counts_b
    all_classes = comp.all_classes
    errors = comp.errors
    top = settings.top

    n_ph = len(phash_rows)
    ph_identical = sum(1 for d, _, _ in phash_rows if d == 0)
    ph_near = sum(1 for d, _, _ in phash_rows if 0 < d <= settings.near_dup_threshold)
    ph_far = n_ph - ph_identical - ph_near
    ph_distances = [d for d, _, _ in phash_rows]
    ph_mean = sum(ph_distances) / n_ph if n_ph else 0.0
    ph_max = max(ph_distances) if ph_distances else 0

    n_ssim = len(ssim_rows)
    ssim_vals = [s for s, _, _ in ssim_rows]
    ssim_mean = sum(ssim_vals) / n_ssim if n_ssim else 0.0
    ssim_min = min(ssim_vals) if ssim_vals else 0.0
    ssim_max = max(ssim_vals) if ssim_vals else 0.0
    ssim_below = sum(1 for s in ssim_vals if s < settings.ssim_threshold)

    diff = len(records_b) - len(records_a)

    L: list[str] = []
    L.append("# Dataset Comparison Report")
    L.append("")
    L.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- **Dataset A:** `{comp.root_a}`")
    L.append(f"- **Dataset B:** `{comp.root_b}`")
    L.append("- **Match key:** file basename (class = immediate parent folder name)")
    L.append(f"- **Content match (SHA-256 sort-merge):** {'enabled' if settings.match_content else 'disabled'}")
    L.append(f"- **SSIM backend:** {SSIM_BACKEND}")
    L.append(f"- **Near-duplicate threshold:** perceptual-hash Hamming distance <= {settings.near_dup_threshold}")
    L.append(f"- **SSIM threshold:** {settings.ssim_threshold}")
    L.append(f"- **Runtime:** {comp.runtime:.1f} s")
    L.append("")
    L.append("---")
    L.append("")

    L.append("## 1. Overview")
    L.append("")
    L += md_table(
        ["Metric", "Dataset A", "Dataset B"],
        [
            ["Total image files", len(records_a), len(records_b)],
            ["Distinct class folders", len(counts_a), len(counts_b)],
            ["Matched by filename", len(matched), len(matched)],
            ["Only in A", len(only_a), "—"],
            ["Only in B", "—", len(only_b)],
            ["Ambiguous filenames (disambiguated by class)", len(ambiguous), len(ambiguous)],
            ["Files with read/hash errors", sum(1 for r in records_a if r.error), sum(1 for r in records_b if r.error)],
        ],
    )
    L.append("")

    L.append("## 2. Total Image Count")
    L.append("")
    L.append(f"- Dataset A: **{len(records_a)}** images")
    L.append(f"- Dataset B: **{len(records_b)}** images")
    L.append(f"- Difference (B - A): **{diff:+d}** images")
    L.append("")

    L.append("## 3. Images per Class")
    L.append("")
    rows = []
    for c in all_classes:
        ca, cb = counts_a.get(c, 0), counts_b.get(c, 0)
        rows.append([c, ca, cb, f"{cb - ca:+d}" if (ca or cb) else "0"])
    rows.append(["**Total**", f"**{len(records_a)}**", f"**{len(records_b)}**", f"**{diff:+d}**"])
    L += md_table(["Class", "A", "B", "B - A"], rows)
    L.append("")

    L.append("## 4. File Names")
    L.append("")
    L.append(f"- Matched by filename: **{len(matched)}**")
    L.append(f"- Only in A: **{len(only_a)}**")
    L.append(f"- Only in B: **{len(only_b)}**")
    L.append("")
    if only_a:
        L.append(f"### Files only in A (first {min(top, len(only_a))} of {len(only_a)})")
        L.append("")
        L.append("```")
        for r in only_a[:top]:
            L.append(r.rel)
        if len(only_a) > top:
            L.append(f"... and {len(only_a) - top} more")
        L.append("```")
        L.append("")
    if only_b:
        L.append(f"### Files only in B (first {min(top, len(only_b))} of {len(only_b)})")
        L.append("")
        L.append("```")
        for r in only_b[:top]:
            L.append(r.rel)
        if len(only_b) > top:
            L.append(f"... and {len(only_b) - top} more")
        L.append("```")
        L.append("")
    if ambiguous:
        L.append(f"### Ambiguous filenames ({len(ambiguous)})")
        L.append("")
        L.append("```")
        for name in ambiguous[:top]:
            L.append(name)
        L.append("```")
        L.append("")

    L.append("## 5. SHA-256 Hashes")
    L.append("")
    L.append(f"- Matched pairs compared: **{len(matched)}**")
    L.append(f"- Byte-identical (same SHA-256): **{len(matched) - len(sha_mismatch_rows)}** ({pct(len(matched) - len(sha_mismatch_rows), len(matched))})")
    L.append(f"- Different SHA-256: **{len(sha_mismatch_rows)}**")
    L.append("")
    if sha_mismatch_rows:
        L.append(f"### Mismatched files (first {min(top, len(sha_mismatch_rows))})")
        L.append("")
        L += md_table(["Class", "File", "A SHA-256 (16)", "B SHA-256 (16)"], sha_mismatch_rows[:top])
        L.append("")

    L.append("## 6. Perceptual Hashes")
    L.append("")
    L.append(f"- Pairs compared: **{n_ph}**")
    L.append(f"- Identical pHash (distance 0): **{ph_identical}** ({pct(ph_identical, n_ph)})")
    L.append(f"- Near-duplicate (1 <= distance <= {settings.near_dup_threshold}): **{ph_near}** ({pct(ph_near, n_ph)})")
    L.append(f"- Different (> {settings.near_dup_threshold}): **{ph_far}** ({pct(ph_far, n_ph)})")
    L.append(f"- pHash distance: mean **{ph_mean:.2f}**, max **{ph_max}**")
    if dhash_dists:
        L.append(f"- dHash distance: mean **{sum(dhash_dists) / len(dhash_dists):.2f}**, max **{max(dhash_dists)}**")
    if ahash_dists:
        L.append(f"- aHash distance: mean **{sum(ahash_dists) / len(ahash_dists):.2f}**, max **{max(ahash_dists)}**")
    if phash_missing:
        L.append(f"- Pairs skipped (missing perceptual hash): **{phash_missing}**")
    L.append("")
    if phash_rows:
        L.append(f"### Largest pHash distances (first {min(top, len(phash_rows))})")
        L.append("")
        L += md_table(
            ["Class", "File", "pHash distance"],
            [[a.cls, a.name, d] for d, a, _ in phash_rows[:top]],
        )
        L.append("")

    L.append("## 7. Image Similarity (SSIM)")
    L.append("")
    L.append(f"- Pairs compared: **{n_ssim}**")
    L.append(f"- SSIM: mean **{ssim_mean:.4f}**, min **{ssim_min:.4f}**, max **{ssim_max:.4f}**")
    L.append(f"- Pairs with SSIM < {settings.ssim_threshold}: **{ssim_below}** ({pct(ssim_below, n_ssim)})")
    if ssim_errors:
        L.append(f"- Pairs skipped (load/SSIM error): **{ssim_errors}**")
    L.append("")
    if ssim_rows:
        L.append(f"### Lowest SSIM (first {min(top, len(ssim_rows))})")
        L.append("")
        L += md_table(
            ["Class", "File", "SSIM"],
            [[a.cls, a.name, f"{s:.4f}"] for s, a, _ in ssim_rows[:top]],
        )
        L.append("")

    L.append("## 8. Content Matches (exact SHA-256, sort-merge)")
    L.append("")
    if settings.match_content:
        content_groups = comp.content_groups
        n_shared = len(content_groups)
        n_a_content = sum(len(g[1]) for g in content_groups)
        n_b_content = sum(len(g[2]) for g in content_groups)
        renamed_groups = sum(
            1
            for _h, a_list, b_list in content_groups
            if not (
                len(a_list) == 1
                and len(b_list) == 1
                and a_list[0].name == b_list[0].name
            )
        )
        L.append("- Method: sort both datasets' SHA-256 lists and merge once; finds **exact** content matches regardless of filename.")
        L.append("- Catches **renamed / byte-identical duplicates** that the name-based pass (Section 4) misses.")
        L.append("- Near-duplicates are **not** detected here (see Section 6 for perceptual hashes).")
        L.append("")
        L.append(f"- Shared SHA-256 values: **{n_shared}**")
        L.append(f"- Files involved: A **{n_a_content}**, B **{n_b_content}**")
        L.append(f"- Shared groups whose filename differs: **{renamed_groups}**")
        L.append("")
        if content_groups:
            L.append(f"### Shared content (first {min(top, n_shared)} of {n_shared})")
            L.append("")
            rows = []
            for h, a_list, b_list in content_groups[:top]:
                rows.append([
                    h[:16],
                    ", ".join(r.rel for r in a_list),
                    ", ".join(r.rel for r in b_list),
                ])
            L += md_table(["SHA-256 (16)", "A file(s)", "B file(s)"], rows)
            L.append("")
    else:
        L.append("_Not run. Pass `--match-content` to match files by exact content (SHA-256) and catch renamed duplicates._")
        L.append("")

    L.append("## 9. Conclusion")
    L.append("")
    L.append(classify_verdict(len(records_a), len(records_b), only_a, only_b, len(sha_mismatch_rows)))
    L.append("")
    if errors:
        L.append("### Files that could not be processed")
        L.append("")
        L.append("```")
        for r in errors[:top]:
            L.append(f"{r.rel}  ->  {r.error}")
        L.append("```")
        L.append("")

    return "\n".join(L) + "\n"
