"""Union-find grouping over exact and accepted near-duplicate pairs.

Ref: Proposal Section 8.2 (Protocol B leakage-controlled group split); the
grouping is image-level, not the NBIS fingerprint-level grouping of Section 7.6.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..model import ImageRecord
from .util import audit_label


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def assign_groups(
    records: list[ImageRecord],
    exact_rows: list[dict],
    near_rows: list[dict],
) -> list[dict]:
    uf = UnionFind(len(records))
    reasons: dict[int, set[str]] = defaultdict(set)

    by_exact: dict[str, list[int]] = defaultdict(list)
    for row in exact_rows:
        by_exact[row["exact_group"]].append(row["image_id"])
    for ids in by_exact.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
        for i in ids:
            reasons[i].add("exact_duplicate")

    for row in near_rows:
        if not row["related"]:
            continue
        a, b = row["image_id_a"], row["image_id_b"]
        uf.union(a, b)
        reasons[a].add("near_duplicate")
        reasons[b].add("near_duplicate")

    roots = [uf.find(i) for i in range(len(records))]
    members: dict[int, list[int]] = defaultdict(list)
    for i, root in enumerate(roots):
        members[root].append(i)
    ordered = sorted(members, key=lambda r: min(members[r]))
    names = {root: f"G{i:05d}" for i, root in enumerate(ordered, 1)}

    size = Counter(names[roots[i]] for i in range(len(records)))
    label_count: dict[str, set[str]] = defaultdict(set)
    for i, r in enumerate(records):
        label_count[names[roots[i]]].add(audit_label(r))

    rows: list[dict] = []
    for i, r in enumerate(records):
        gid = names[roots[i]]
        rows.append({
            "image_id": i,
            "image_path": r.rel,
            "blood_group": audit_label(r),
            "group_id": gid,
            "group_reason": ";".join(sorted(reasons.get(i, {"singleton"}))),
            "group_size": size[gid],
            "group_label_count": len(label_count[gid]),
            "label_conflict": len(label_count[gid]) > 1,
        })
    return rows
