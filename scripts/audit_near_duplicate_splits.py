#!/usr/bin/env python3
"""Audit accepted near-duplicate pairs against Protocol A/B split CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path, PurePosixPath


TRUE_VALUES = {"1", "true", "yes", "y"}
SPLIT_RE = re.compile(r"protocol_([ab])_seed(\d+)\.csv$", re.IGNORECASE)


def normalise_path(value: str) -> str:
    """Normalise audit and split paths to class/basename form."""
    path = value.strip().replace("\\", "/")
    parts = [part for part in PurePosixPath(path).parts if part not in {"", "."}]
    if len(parts) < 2:
        return "/".join(parts).casefold()
    return "/".join(parts[-2:]).casefold()


def read_accepted_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path_a", "image_path_b", "related"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Near-pair CSV missing columns: {sorted(missing)}")
        return [
            row
            for row in reader
            if row["related"].strip().casefold() in TRUE_VALUES
        ]


def read_split(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filepath", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split CSV {path.name} missing columns: {sorted(missing)}")
        for row in reader:
            key = normalise_path(row["filepath"])
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError(f"Duplicate path with conflicting split: {key}")
            mapping[key] = row["split"].strip().casefold()
    return mapping


def audit_pair(pair: dict[str, str], mapping: dict[str, str]) -> dict[str, str]:
    key_a = normalise_path(pair["image_path_a"])
    key_b = normalise_path(pair["image_path_b"])
    split_a = mapping.get(key_a, "")
    split_b = mapping.get(key_b, "")
    if not split_a or not split_b:
        status = "unresolved"
    elif split_a == split_b:
        status = "within_split"
    else:
        status = "cross_split"
    return {
        "image_path_a": pair["image_path_a"],
        "image_path_b": pair["image_path_b"],
        "phash_distance": pair.get("phash_distance", ""),
        "dhash_distance": pair.get("dhash_distance", ""),
        "ssim": pair.get("ssim", ""),
        "split_a": split_a,
        "split_b": split_b,
        "status": status,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check accepted near-duplicate pairs across Protocol A/B splits."
    )
    parser.add_argument("--near-pairs", required=True, type=Path)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pairs = read_accepted_pairs(args.near_pairs)
    split_files = sorted(args.splits_dir.glob("protocol_*_seed*.csv"))
    if not split_files:
        raise FileNotFoundError(f"No protocol split CSV files in {args.splits_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for split_file in split_files:
        match = SPLIT_RE.match(split_file.name)
        if not match:
            continue
        protocol = match.group(1).upper()
        seed = int(match.group(2))
        mapping = read_split(split_file)
        rows = [audit_pair(pair, mapping) for pair in pairs]
        for row in rows:
            details.append({"protocol": protocol, "seed": seed, **row})

        cross = sum(row["status"] == "cross_split" for row in rows)
        within = sum(row["status"] == "within_split" for row in rows)
        unresolved = sum(row["status"] == "unresolved" for row in rows)
        summaries.append({
            "protocol": protocol,
            "seed": seed,
            "accepted_pairs": len(pairs),
            "within_split_pairs": within,
            "cross_split_pairs": cross,
            "unresolved_pairs": unresolved,
            "pass": cross == 0 and unresolved == 0,
        })

    write_csv(
        args.output_dir / "near_duplicate_cross_split_summary.csv",
        summaries,
        ["protocol", "seed", "accepted_pairs", "within_split_pairs",
         "cross_split_pairs", "unresolved_pairs", "pass"],
    )
    write_csv(
        args.output_dir / "near_duplicate_cross_split_details.csv",
        details,
        ["protocol", "seed", "image_path_a", "image_path_b",
         "phash_distance", "dhash_distance", "ssim", "split_a", "split_b", "status"],
    )
    with (args.output_dir / "near_duplicate_cross_split_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"accepted_pairs": len(pairs), "runs": summaries}, handle, indent=2)
        handle.write("\n")

    print(f"Accepted near-duplicate pairs: {len(pairs)}")
    for row in summaries:
        print(
            f"Protocol {row['protocol']} seed {row['seed']}: "
            f"cross={row['cross_split_pairs']}, "
            f"within={row['within_split_pairs']}, "
            f"unresolved={row['unresolved_pairs']}, pass={row['pass']}"
        )
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
