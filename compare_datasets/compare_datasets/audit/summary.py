"""Summary JSON construction for the audit."""

from __future__ import annotations

from collections import Counter

from ..model import ImageRecord


def build_summary(
    records: list[ImageRecord],
    manifest_rows: list[dict],
    exact_rows: list[dict],
    near_rows: list[dict],
    grouped_rows: list[dict],
    settings: dict,
    runtime: float,
    near_candidate_count: int | None = None,
) -> dict:
    related = [r for r in near_rows if r["related"]]
    return {
        "settings": settings,
        "total_images": len(records),
        "readable_images": sum(1 for r in records if r.readable),
        "unreadable_images": sum(1 for r in records if not r.readable),
        "class_counts": dict(sorted(Counter(r["blood_group"] for r in manifest_rows).items())),
        "exact_duplicate_groups": len({r["exact_group"] for r in exact_rows}),
        "images_in_exact_duplicate_groups": len(exact_rows),
        "exact_cross_label_groups": len({r["exact_group"] for r in exact_rows if r["label_conflict"]}),
        "near_candidate_pairs": len(near_rows) if near_candidate_count is None else near_candidate_count,
        "accepted_near_duplicate_pairs": len(related),
        "accepted_near_cross_label_pairs": sum(1 for r in related if r["label_conflict"]),
        "final_groups": len({r["group_id"] for r in grouped_rows}),
        "non_singleton_groups": len({r["group_id"] for r in grouped_rows if r["group_size"] > 1}),
        "cross_label_groups": len({r["group_id"] for r in grouped_rows if r["label_conflict"]}),
        "runtime_seconds": round(runtime, 1),
    }
