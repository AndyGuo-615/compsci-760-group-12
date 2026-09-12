"""Filename-based and content-based matching.

Ref: Proposal Section 6.3 (file names, SHA-256) and Section 7.4 (exact
duplicate detection); Survey Section VI.A.
"""

from __future__ import annotations

from collections.abc import Callable

from .model import ImageRecord


def match_records(
    records_a: list[ImageRecord], records_b: list[ImageRecord]
) -> tuple[
    list[tuple[ImageRecord, ImageRecord]],
    list[ImageRecord],
    list[ImageRecord],
    list[str],
]:
    """Match the two datasets by filename, disambiguating by class folder."""
    idx_a: dict[str, list[ImageRecord]] = {}
    idx_b: dict[str, list[ImageRecord]] = {}
    for r in records_a:
        idx_a.setdefault(r.name, []).append(r)
    for r in records_b:
        idx_b.setdefault(r.name, []).append(r)

    matched: list[tuple[ImageRecord, ImageRecord]] = []
    only_a: list[ImageRecord] = []
    only_b: list[ImageRecord] = []
    ambiguous: list[str] = []

    for name, a_list in idx_a.items():
        b_list = idx_b.get(name, [])
        if not b_list:
            only_a.extend(a_list)
            continue
        if len(a_list) == 1 and len(b_list) == 1:
            matched.append((a_list[0], b_list[0]))
            continue

        a_by_cls = {r.cls: r for r in a_list}
        b_by_cls = {r.cls: r for r in b_list}
        common = set(a_by_cls) & set(b_by_cls)
        for c in sorted(common):
            matched.append((a_by_cls[c], b_by_cls[c]))
        only_a.extend(a_by_cls[c] for c in a_by_cls if c not in common)
        only_b.extend(b_by_cls[c] for c in b_by_cls if c not in common)
        ambiguous.append(name)

    for name, b_list in idx_b.items():
        if name not in idx_a:
            only_b.extend(b_list)

    matched.sort(key=lambda pair: (pair[0].cls, pair[0].name))
    return matched, only_a, only_b, ambiguous


def match_by_content(
    records_a: list[ImageRecord],
    records_b: list[ImageRecord],
    key: Callable[[ImageRecord], str] = lambda r: r.sha256,
) -> list[tuple[str, list[ImageRecord], list[ImageRecord]]]:
    """Sort-merge match by content hash (exact matches only).

    Sorts both datasets by the same key and walks the two sorted lists once.
    Because it keys on content rather than filename, it finds renamed /
    byte-identical duplicates that the name-based pass misses. Exact equality
    only: near-duplicates need perceptual-hash Hamming distance.

    Returns ``(hash, [A records], [B records])`` groups for every hash present
    in both datasets.
    """
    a_sorted = sorted((r for r in records_a if key(r)), key=key)
    b_sorted = sorted((r for r in records_b if key(r)), key=key)

    groups: list[tuple[str, list[ImageRecord], list[ImageRecord]]] = []
    i = j = 0
    while i < len(a_sorted) and j < len(b_sorted):
        ka, kb = key(a_sorted[i]), key(b_sorted[j])
        if ka == kb:
            i2 = i
            while i2 < len(a_sorted) and key(a_sorted[i2]) == ka:
                i2 += 1
            j2 = j
            while j2 < len(b_sorted) and key(b_sorted[j2]) == kb:
                j2 += 1
            groups.append((ka, a_sorted[i:i2], b_sorted[j:j2]))
            i, j = i2, j2
        elif ka < kb:
            i += 1
        else:
            j += 1
    return groups
