#!/usr/bin/env python3
"""Rebuild Protocol B's main_subject_groups.csv from two audit manifests.

Inputs are the manifest.csv files produced by compare_datasets audit mode for:
1. the 5,837-image blood-group dataset; and
2. SOCOFing/Real (6,000 images).

Only exact SHA-256 matches are used.  If one hash maps to several SOCOFing
subjects, every candidate is retained and those subjects are joined into one
connected component.  This reproduces the conservative subject grouping used
for Protocol B without silently choosing one candidate subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


SOCOFING_RE = re.compile(
    r"^(?P<subject>\d+)__(?P<gender>[MF])_"
    r"(?P<hand>Left|Right)_(?P<finger>.+?)_finger\.BMP$",
    re.IGNORECASE,
)

OUTPUT_FIELDS = [
    "filepath", "label", "group_id", "subject_ids", "socofing_files", "sha256"
]


class UnionFind:
    def __init__(self, items: set[int]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: list[dict[str, str]], columns: set[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = columns.difference(rows[0])
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def parse_socofing_name(image_path: str) -> dict[str, str | int]:
    filename = PurePosixPath(image_path.replace("\\", "/")).name
    match = SOCOFING_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Unrecognised SOCOFing Real filename: {image_path}")
    values = match.groupdict()
    return {
        "subject_id": int(values["subject"]),
        "gender": values["gender"].upper(),
        "hand": values["hand"].title(),
        "finger": values["finger"].lower(),
        "filename": filename,
    }


def prefixed_path(prefix: str, image_path: str) -> str:
    relative = image_path.replace("\\", "/").lstrip("/")
    clean_prefix = prefix.replace("\\", "/").strip("/")
    return str(PurePosixPath(clean_prefix, relative)) if clean_prefix else relative


def build(
    main_rows: list[dict[str, str]],
    socofing_rows: list[dict[str, str]],
    dataset_prefix: str,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    require_columns(main_rows, {"image_path", "blood_group", "sha256"}, "main manifest")
    require_columns(socofing_rows, {"image_path", "sha256"}, "SOCOFing manifest")

    paths = [row["image_path"] for row in main_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("main manifest contains duplicate image_path values")

    by_hash: dict[str, list[dict]] = defaultdict(list)
    subject_metadata: dict[int, dict[str, str]] = {}
    for row in socofing_rows:
        digest = row["sha256"].strip().lower()
        if not digest:
            continue
        meta = parse_socofing_name(row["image_path"])
        subject_id = int(meta["subject_id"])
        subject_metadata.setdefault(subject_id, {
            "gender": str(meta["gender"]),
        })
        by_hash[digest].append(meta)

    all_subjects = {int(meta["subject_id"]) for rows in by_hash.values() for meta in rows}
    union_find = UnionFind(all_subjects)
    matches_by_image: list[list[dict]] = []

    for row in main_rows:
        digest = row["sha256"].strip().lower()
        unique = {
            (int(meta["subject_id"]), str(meta["filename"])): meta
            for meta in by_hash.get(digest, [])
        }
        matches = sorted(unique.values(), key=lambda meta: (
            int(meta["subject_id"]), str(meta["filename"])
        ))
        matches_by_image.append(matches)
        subject_ids = sorted({int(meta["subject_id"]) for meta in matches})
        for other in subject_ids[1:]:
            union_find.union(subject_ids[0], other)

    components: dict[int, set[int]] = defaultdict(set)
    for subject in sorted(all_subjects):
        components[union_find.find(subject)].add(subject)
    ordered_components = sorted(components.values(), key=min)
    subject_group: dict[int, str] = {}
    for number, subjects in enumerate(ordered_components, 1):
        group_id = f"SUBJECT_GROUP_{number:04d}"
        for subject in subjects:
            subject_group[subject] = group_id

    output: list[dict] = []
    links: list[dict] = []
    unmatched = 0
    ambiguous = 0
    for row, matches in zip(main_rows, matches_by_image):
        subject_ids = sorted({int(meta["subject_id"]) for meta in matches})
        if not subject_ids:
            unmatched += 1
            group_id = f"UNMATCHED_{unmatched:04d}"
        else:
            group_id = subject_group[subject_ids[0]]
            ambiguous += len(subject_ids) > 1

        output.append({
            "filepath": prefixed_path(dataset_prefix, row["image_path"]),
            "label": row["blood_group"],
            "group_id": group_id,
            "subject_ids": ";".join(map(str, subject_ids)),
            "socofing_files": ";".join(str(meta["filename"]) for meta in matches),
            "sha256": row["sha256"].strip().lower(),
        })
        for meta in matches:
            links.append({
                "main_image_path": row["image_path"],
                "blood_group": row["blood_group"],
                "sha256": row["sha256"].strip().lower(),
                "subject_id": int(meta["subject_id"]),
                "gender": meta["gender"],
                "hand": meta["hand"],
                "finger": meta["finger"],
                "socofing_file": meta["filename"],
            })

    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in output:
        group_rows[row["group_id"]].append(row)

    person_rows: list[dict] = []
    for group_id in sorted(group_rows):
        rows = group_rows[group_id]
        subjects = sorted({
            int(value)
            for row in rows
            for value in row["subject_ids"].split(";")
            if value
        })
        labels = sorted({row["label"] for row in rows})
        person_rows.append({
            "group_id": group_id,
            "subject_ids": ";".join(map(str, subjects)),
            "subject_count": len(subjects),
            "image_count": len(rows),
            "blood_groups": ";".join(labels),
            "blood_group_count": len(labels),
            "label_conflict": len(labels) > 1,
        })

    matched_subjects = sorted({
        int(value)
        for row in output
        for value in row["subject_ids"].split(";")
        if value
    })
    hands = Counter(str(link["hand"]) for link in links)
    fingers = Counter(str(link["finger"]) for link in links)
    summary = {
        "main_images": len(main_rows),
        "socofing_real_images": len(socofing_rows),
        "matched_images": len(main_rows) - unmatched,
        "unmatched_images": unmatched,
        "ambiguous_multi_subject_images": ambiguous,
        "subject_link_rows": len(links),
        "matched_subject_count": len(matched_subjects),
        "subject_id_min": min(matched_subjects) if matched_subjects else None,
        "subject_id_max": max(matched_subjects) if matched_subjects else None,
        "missing_subject_ids_in_range": (
            sorted(set(range(min(matched_subjects), max(matched_subjects) + 1)) - set(matched_subjects))
            if matched_subjects else []
        ),
        "final_groups": len(group_rows),
        "non_singleton_groups": sum(len(rows) > 1 for rows in group_rows.values()),
        "groups_with_multiple_subject_ids": sum(row["subject_count"] > 1 for row in person_rows),
        "groups_with_multiple_blood_groups": sum(bool(row["label_conflict"]) for row in person_rows),
        "largest_group_size": max((len(rows) for rows in group_rows.values()), default=0),
        "hand_link_counts": dict(sorted(hands.items())),
        "finger_link_counts": dict(sorted(fingers.items())),
    }
    return output, links, person_rows, summary


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild main_subject_groups.csv by SHA-256 matching to SOCOFing Real."
    )
    parser.add_argument("--main-manifest", type=Path, required=True)
    parser.add_argument("--socofing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("subject_manifest_out"))
    parser.add_argument("--dataset-prefix", default="data/datasets")
    parser.add_argument(
        "--allow-unmatched", action="store_true",
        help="Write unmatched images instead of failing the reproducibility check.",
    )
    args = parser.parse_args()

    output, links, people, summary = build(
        read_csv(args.main_manifest),
        read_csv(args.socofing_manifest),
        args.dataset_prefix,
    )
    if summary["unmatched_images"] and not args.allow_unmatched:
        raise RuntimeError(
            f"{summary['unmatched_images']} main images did not match SOCOFing Real; "
            "rerun with --allow-unmatched only if this is expected."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "main_subject_groups.csv", OUTPUT_FIELDS, output)
    write_csv(args.output_dir / "subject_match_links.csv", [
        "main_image_path", "blood_group", "sha256", "subject_id", "gender",
        "hand", "finger", "socofing_file",
    ], links)
    write_csv(args.output_dir / "person_consistency_audit.csv", [
        "group_id", "subject_ids", "subject_count", "image_count",
        "blood_groups", "blood_group_count", "label_conflict",
    ], people)
    with (args.output_dir / "subject_manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Outputs written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
