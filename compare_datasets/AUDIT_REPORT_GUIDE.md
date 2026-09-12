# Interpreting the Dataset Audit Report
author： Xia Yang
UPI: yxia728
Date: 11/Sep/2026

A first-time reader's guide to `dataset_audit_report.md`, produced by the audit mode of this tool:

```bash
python run_compare.py <DATASET_DIR> --audit -o dataset_audit_report.md
```

## What this tool does

This command line tool is for machine-learning data pre-processing. Specifically, the script creates a **data
manifest** that records one row per image — its path, class label, width, height, file format,
file size, SHA-256 hash and perceptual hashes (pHash/dHash) — and then uses those records to:

1. **detect exact duplicates** — images with identical SHA-256 bytes (Proposal §7.4);
2. **detect near duplicates** — perceptual-hash candidates confirmed by SSIM, catching resized,
   recompressed or re-saved copies (Proposal §7.5);
3. **group related images** — merging both into `group_id` clusters so they stay together when the
   data is split into training / validation / test sets (Proposal §8.2);
4. **scale across CPU cores** — the near-duplicate stage (the slow one) runs on multiple worker
   processes with `--workers N` (`0` = all CPU cores), streaming results to disk so memory stays
   bounded even for tens of thousands of images. it provides much faster processing vs python default single core

The result is a Markdown audit report plus CSV/JSON exports (`manifest.csv`,
`exact_duplicates.csv`, `near_duplicate_pairs.csv`, `grouped_manifest.csv`, `summary.json`),
which together give every image a `group_id` for leakage-controlled evaluation.

It maps every section of the report to the relevant part of the
**COMPSCI 760 Project Proposal** so you may see *why* each number is there and *what decision it
informs*. Read the Proposal §7.1–§7.7 alongside this guide.



---

## 0. Read the report in 60 seconds

| Question you should ask | Look at | Proposal link |
|---|---|---|
| Is the dataset intact and balanced? |  Manifest | §7.1, §7.2 |
| Are there byte-identical copies? |  Exact Duplicates | §7.4 |
| Are there re-encoded / resized copies? |  Near Duplicates | §7.5 |
| Which images belong together? |  Groups | §7.4–§7.6 |
| What is the one-line verdict? |  Conclusion | §8.2 |

The single most important number for the project is **Groups** — those group IDs are what stop related images leaking across train / validation / test under **Protocol B** (Proposal §8.2).

---

## 1. What this report is, and what it is not

- It is a **single-dataset audit**: it looks for duplication *inside one folder*.
  (To compare two datasets, run the tool without `--audit`.)
- It operates at the **image level**: it finds images that look the same. It does **not** decide
  whether two different images come from the same finger or subject — that is the NBIS
  fingerprint-matching problem of Proposal **§7.6**, validated separately on SOCOFing (**§7.7**).
- "**Class**" in this report means **the first folder under the dataset root** (the label folder).
  For the main Sravani dataset that is the blood group (`A+`, `AB-`, …); for SOCOFing it is
  `Real` / `Altered`. Keep this in mind — it changes how you read "cross-class".

---

## 2. The report header

```
- Generated:   <timestamp>
- Dataset:     <path you audited>
- Mode:        single-dataset audit (Proposal §7.1 / §7.4 / §7.5 + grouping)
- SSIM backend: scikit-image (structural_similarity)   ← or the numpy fallback
- pHash threshold: Hamming <= 10
- dHash threshold: Hamming <= 12
- SSIM threshold:  0.95
- Runtime:     <seconds>
```

These are the **knobs** that produced every number below. If you change a threshold, the
counts in §3 and §4 change; the numbers are only comparable between runs that use the same
settings (also recorded in `summary.json` → `settings`).

- **pHash / dHash thresholds** — how close two perceptual hashes must be to become a *candidate*
  pair (Hamming distance). Lower = stricter.
- **SSIM threshold** — how structurally similar two images must be to *confirm* a candidate pair.
- **Runtime** — the near-duplicate stage dominates. See "Speed" below.

---

## 3. Section 1 — Manifest (Proposal §7.1)

**Proposal §7.1** asks for a manifest with one row per image (`image_path, blood_group, width,
height, file_format, file_size, sha256, …`), plus a `group_id` added after duplicate analysis.

The report shows the summary; the full table is exported to `manifest.csv`.

| Field in report | Meaning | Why it matters (Proposal) |
|---|---|---|
| Total images | every readable image found | §7.2 structure check |
| Readable / unreadable | decoding successes / failures | §7.2 damaged-file check |
| Class counts | images per label folder | §7.2 class balance |
| Formats | file extensions seen | §7.2 format check |

**How to use it:** an unbalanced class table (e.g. `A+` far smaller than `A-`) is expected in the
main dataset and motivates **balanced accuracy / macro F1** in Proposal §11 rather than plain
accuracy. Any unreadable files must be reported, not silently dropped.

---

## 4. Section 2 — Exact Duplicates (Proposal §7.4)

**Proposal §7.4** computes a SHA-256 for every images; identical hashes are exact copies and
**must receive the same group ID** and must never be split across train / validation / test.

The report gives:

- number of exact-duplicate **groups** (a group is 2+ files with the same SHA-256),
- total images inside those groups,
- how many groups span more than one class.

The table lists each `EXACT_xxxxx` group with its size, class count, and sample file paths.
Full detail: `exact_duplicates.csv`.

**How to read it:** a group of size 3 means three files are byte-identical. If the group spans
more than one class, two different labels share literally the same bytes — either a labelling
error or leakage. **`Conflict = yes`** is the flag to investigate.

> In a clean dataset you would expect few or none. In SOCOFing, 11 groups exist because the
> same fingerprint was accidentally filed under different subject IDs — see the worked example.

---

## 5. Section 3 — Near Duplicates (Proposal §7.5)

**Proposal §7.5** covers related images that are *not* byte-identical because they were resized,
rotated, cropped, recompressed, brightened, or re-saved in another format. The Proposal lists
perceptual hashing, SSIM, image embeddings and fingerprint matching as possible methods; this
tool uses **perceptual hashing (pHash + dHash) to generate candidates, then SSIM to confirm**.

Two stages are reported:

1. **Hash-level candidate pairs** — pairs whose pHash *and* dHash are within the thresholds.
   This is a cheap filter and is deliberately over-inclusive.
2. **Accepted pairs (SSIM ≥ threshold)** — candidates that also pass the structural check.
   These are the ones treated as "related".

The table lists the highest-SSIM pairs with columns:

| Column | Meaning |
|---|---|
| Class A / Class B | label folders of the two files |
| File A / File B | relative paths |
| pHash / dHash | Hamming distance (0 = identical perceptual hash) |
| SSIM | structural similarity in [0, 1]; 1.0 = identical structure |
| Related | `yes` if SSIM ≥ threshold |

**How to read it:** start at the top. A pair with `pHash 0 / dHash 0 / SSIM 1.0000` is
effectively the same image in two files. Pairs that are `Related = yes` across two classes are
candidates for label errors or leakage. Full list: `near_duplicate_pairs.csv` (every candidate,
not just the accepted ones).

> Thresholds matter. A loose pHash threshold (e.g. 10) produces a very large candidate list that
> includes *different fingers*. A strict threshold (e.g. 4) keeps only near-certain matches.
> Tightening the threshold is usually the single biggest speed-up.

---

## 6. Section 4 — Groups (union-find)

This is the **deliverable for Protocol B** (Proposal §8.2). The tool merges all exact-duplicate
groups and all accepted near-duplicate pairs into equivalence classes with a union-find, giving
each image a `group_id`.

- **Total groups** — one per distinct image plus one per merged cluster.
- **Non-singleton groups** — clusters with 2+ images (the interesting ones).
- **Groups spanning >1 class** — clusters containing images from different label folders.

The table shows the largest non-singleton groups with their `Reason` (`exact` or
`near_duplicate`) and `Conflict` flag. Full mapping: `grouped_manifest.csv`.

**How to use it:** under **Protocol B**, split *by `group_id`*, never by image. All images in a
group go to the same side of the split. That is how this report prevents the leakage that
Protocol A (random image split, §8.1) would allow.

---

Finished !