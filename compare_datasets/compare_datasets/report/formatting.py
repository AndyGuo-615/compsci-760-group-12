"""Small Markdown formatting helpers."""

from __future__ import annotations


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"
