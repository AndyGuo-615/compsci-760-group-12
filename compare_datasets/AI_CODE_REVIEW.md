# Code Review by AI — compare_datasets (may not accurate)

- **Package:** `/var/home/peter/760_Group_Project/compare_datasets/`
- **Scope:** all Python modules (package + entry point), 1,810 LOC
- **Date:** 2026-09-11

## Severity legend

| Level | Meaning |
|---|---|
| **Critical** | Wrong results or data loss in normal use |
| **High** | Breaks or becomes unusable at realistic scale |
| **Medium** | Incorrect/misleading output, or silent failure on some paths |
| **Low** | Maintainability or minor-correctness nits |

## Module map (post-split)

| Module | Lines | Responsibility |
|---|---|---|
| `run_compare.py` | 7 | entry point → `compare_datasets.cli:main` |
| `compare_datasets/__init__.py` | 5 | version |
| `compare_datasets/config.py` | 22 | constants and defaults |
| `compare_datasets/model.py` | 76 | `ImageRecord`, `Settings`, `Comparison`, `AuditResult` |
| `compare_datasets/io_utils.py` | 60 | discovery, records, SHA-256, grayscale loading |
| `compare_datasets/hashing.py` | 67 | perceptual hashes + SSIM (numpy fallback) |
| `compare_datasets/matching.py` | 96 | filename matching + content sort-merge |
| `compare_datasets/pipeline.py` | 117 | compare-mode scan/hash/match pipeline |
| `compare_datasets/manual.py` | 57 | barebone user manual |
| `compare_datasets/cli.py` | 149 | argparse + dispatch |
| `compare_datasets/audit/util.py` | 51 | popcount, hash-int, progress log, label |
| `compare_datasets/audit/schema.py` | 31 | CSV column schemas + writer |
| `compare_datasets/audit/manifest.py` | 58 | §7.1 manifest + §7.4 exact duplicates |
| `compare_datasets/audit/near.py` | 129 | §7.5 near-dup search + serial SSIM scoring |
| `compare_datasets/audit/parallel.py` | 159 | parallel SSIM scoring (shard CSVs) |
| `compare_datasets/audit/grouping.py` | 87 | union-find groups |
| `compare_datasets/audit/summary.py` | 37 | summary JSON |
| `compare_datasets/audit/runner.py` | 117 | orchestration + CSV/JSON export |
| `compare_datasets/report/formatting.py` | 15 | table / percent helpers |
| `compare_datasets/report/compare_report.py` | 273 | two-dataset report (§6.3) |
| `compare_datasets/report/audit_report.py` | 151 | single-dataset audit report |

The split was a **pure structural move**: the pre/post regression run produced
byte-identical reports, CSVs and JSON across serial, parallel, `--manifest-only`,
compare and compare `--match-content` (20 artifacts, 0 differences).

## Summary

| ID | Severity | Where | One-line |
|---|---|---|---|
| M1 | Medium | `audit/runner.py:60-73` | `near_candidate_pairs` counts scored rows in serial but raw candidates in parallel |
| M2 | High | `audit/near.py:88` | default serial path keeps every scored row in RAM |
| M3 | High | `audit/near.py:26` | candidate list fully materialised (~10M tuples at 110k) |
| M4 | Medium | `audit/parallel.py:132` | failed worker chunk is logged and skipped; its rows vanish |
| L1 | Low | `pyproject.toml:7` vs `__init__.py:5` | version mismatch 0.1.0 vs 0.2.0 |
| L2 | Low | `audit/schema.py:9,13,22` | `blood_group` column actually holds the first path component |
| L3 | Low | `audit/util.py:48` | `audit_label` falls back to the root folder name |
| L4 | Low | `config.py` vs `model.py:33-37` | defaults duplicated in two places |
| L5 | Low | `io_utils.py:53,59` | resize to 128×128 ignores aspect ratio |
| L6 | Low | `report/compare_report.py:16` | `classify_verdict` ignores content-only matches |
| L7 | Low | `audit/near.py` vs `audit/parallel.py` | SSIM row dict duplicated between the serial and parallel scorers |

No **Critical** findings.

---

## M1 — `near_candidate_pairs` is mode-dependent

- **Location:** `compare_datasets/audit/runner.py:60-73`
- **Description:** The serial branch sets `near_candidate_count = len(near_rows)`
  (only the rows that were *scored and readable*), while the parallel branch takes
  the count returned by `score_near_pairs_streaming` (the *raw* candidate total).
  The audit report labels this number "Hash-level candidate pairs" in both cases.
- **Impact:** The headline candidate count changes depending on `--workers`, so
  two runs of the same dataset disagree. Users comparing serial vs parallel output
  will see different numbers and may mistrust the report.
- **Fix:** Count the candidates once, from `len(pairs)` (available in `run_audit`
  before either scorer runs), and pass that to `build_summary` in both branches.

## M2 — serial SSIM scoring holds the whole table in memory

- **Location:** `compare_datasets/audit/near.py:88` (`score_near_pairs`), called at
  `audit/runner.py:72` on the default `--workers 1` path.
- **Description:** `score_near_pairs` appends a dict per candidate pair and returns
  them all. At SOCOFing scale that is ~884,896 dicts (~hundreds of MB); the
  parallel path was deliberately built to stream instead, but it is not the default.
- **Impact:** The default invocation is the memory-risky one — the opposite of what
  a user would expect from the "safe" single-process mode.
- **Fix:** Route the serial path through the same streaming/shard writer (or a
  generator + `_write_csv`), so only accepted rows are retained.

## M3 — candidate list is fully materialised

- **Location:** `compare_datasets/audit/near.py:26` (`near_duplicate_candidates`),
  accumulator at `:52`.
- **Description:** The vectorised Hamming search builds the complete
  `list[tuple[int, int, int, int]]` before returning. At 110k images and pHash ≤ 10
  that is ~10M tuples (~1.5–2 GB).
- **Impact:** Memory blow-up / OOM before SSIM even starts on large inputs.
- **Fix:** Make the function a generator that yields per-chunk pair lists, and have
  both scorers consume chunks incrementally.

## M4 — failed worker chunk is silently dropped

- **Location:** `compare_datasets/audit/parallel.py:132`
- **Description:** On a worker exception the code prints `SSIM: worker failed: ...`
  and `continue`s, so that chunk's rows never reach the merged CSV or the summary.
- **Impact:** Silent data loss; the report under-counts without a non-zero exit or a
  structured warning.
- **Fix:** Accumulate a `failed_pairs` counter, surface it in the summary/report, and
  exit non-zero (or re-raise) when a chunk fails.

## L1 — version mismatch

- **Location:** `pyproject.toml:7` (`0.1.0`) vs `compare_datasets/__init__.py:5` (`0.2.0`)
- **Impact:** Installed package metadata disagrees with `__version__`.
- **Fix:** Single-source the version (e.g. read `__version__` in `pyproject.toml`).

## L2 — `blood_group` column name is misleading

- **Location:** `compare_datasets/audit/schema.py:9,13,22`; written in
  `audit/manifest.py:21,51` and `audit/grouping.py:80`.
- **Description:** The value is the first path component (`Real`/`Altered` for
  SOCOFing, the class folder for Sravani), not a blood group.
- **Impact:** Report/CSV readers misread the column; the tool's generic "check for
  label leakage" conclusion is a false alarm for SOCOFing.
- **Fix:** Rename to `label` (or make the column name configurable) and adjust the
  report wording.

## L3 — `audit_label` collapses direct-child files

- **Location:** `compare_datasets/audit/util.py:48`
- **Description:** Falls back to `rec.cls` (the root folder name) when the image sits
  directly under the dataset root, so all such files share one label.
- **Impact:** Wrong grouping/label-conflict stats for flat datasets.
- **Fix:** Fall back to `"__root__"` (or the filename) and document the behaviour.

## L4 — defaults duplicated

- **Location:** `compare_datasets/config.py` and `compare_datasets/model.py:33-37`
- **Description:** The same numeric defaults live in both the `Settings` dataclass and
  the `config` constants.
- **Impact:** Drift risk when one is changed.
- **Fix:** Have `Settings` default to the `config` constants.

## L5 — non-aspect-preserving resize

- **Location:** `compare_datasets/io_utils.py:53,59`
- **Description:** `load_gray` resizes to a fixed 128×128, distorting non-square
  images before SSIM.
- **Impact:** SSIM on resized pairs is slightly off for non-square inputs.
- **Fix:** Letterbox/pad to a square, or make the size configurable.

## L6 — `classify_verdict` ignores content-only matches

- **Location:** `compare_datasets/report/compare_report.py:16`
- **Description:** The compare-mode verdict is computed from name matching only, so a
  renamed byte-identical pair found via `--match-content` does not change the verdict.
- **Impact:** Section 9 can contradict Section 8.
- **Fix:** Feed the content-match result into `classify_verdict`.

## L7 — duplicated SSIM row construction

- **Location:** `compare_datasets/audit/near.py` (`score_near_pairs`) vs
  `compare_datasets/audit/parallel.py` (`_score_chunk`).
- **Description:** Both build the identical `NEAR_FIELDS` row dict; only the source of
  the path/label lookups differs.
- **Impact:** Two copies to keep in sync when the schema changes.
- **Fix:** Extract a `_pair_row(...)` helper shared by both scorers.

---

## Coverage gaps vs the Project Proposal

| Proposal section | Status in this tool |
|---|---|
| §7.1 Data manifest | **Partial** — has id/path/label/size/format/dimensions/SHA-256/pHash/dHash; missing brightness, contrast, image_quality |
| §7.2 Basic dataset audit | **Partial** — counts/formats covered; no resolution/quality report |
| §7.3 Low-cost controls | **Absent** (metadata-only baseline, label permutation) |
| §7.4 Exact duplicate detection | **Implemented** (`audit/manifest.py`) |
| §7.5 Near duplicate detection | **Implemented** (`audit/near.py`, `audit/parallel.py`) |
| §7.6 NBIS fingerprint matching & grouping | **Absent** — grouping here is image-level, not finger-level |
| §7.7 Validate grouping on SOCOFing | **Absent** — no precision/recall/F1 or Adjusted Rand Index |
| §8.2 Protocol B group split | **Not implemented** — `group_id` is produced, but no split code |
| §10 Controls / §11 Metrics | **Out of scope** for this tool |

## Recommended fix order

1. **M2 + M3** (memory at scale — same root cause, do together)
2. **M1** (mode-dependent headline number)
3. **M4** (silent data loss)
4. **L7** (de-duplicate the row builder while touching the scorers)
5. **L1** (version single-sourcing)
6. L2–L6 as opportunistic cleanups
