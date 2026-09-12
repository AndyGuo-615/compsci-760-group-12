# compare_datasets

Compare two image-dataset directories (which may have **different internal structure**) across the
checks from the COMPSCI 760 proposal, **Section 6.3**:

1. Total image count
2. Number of images in each class
3. File names
4. SHA-256 hashes
5. Perceptual hashes (pHash / dHash / aHash)
6. Image-similarity measures (SSIM)

Files are matched by **basename** (filename), because the two directory trees may place the same
files at different paths. The "class" of an image is the name of its immediate parent directory
(e.g. `A+`, `AB-`). Results are written as a Markdown report inside a `.txt` file.

## Layout

```
compare_datasets/
├── run_compare.py            # entry point
├── requirements.txt
├── pyproject.toml
├── README.md
├── AUDIT_REPORT_GUIDE.md     # how to read the audit report (maps to Proposal §7)
├── CODE_REVIEW.md            # code review: findings + Proposal coverage gaps
└── compare_datasets/
    ├── __init__.py
    ├── config.py             # constants and defaults
    ├── model.py              # ImageRecord, Settings, Comparison, AuditResult
    ├── io_utils.py           # discovery, records, SHA-256, grayscale loading
    ├── hashing.py            # perceptual hashes + SSIM (with numpy fallback)
    ├── matching.py           # filename matching + content sort-merge matching
    ├── pipeline.py           # scan / hash / match pipeline
    ├── manual.py             # barebone user manual
    ├── cli.py                # argparse + main
    ├── audit/                # single-dataset audit (Proposal §7.1 / §7.4 / §7.5)
    │   ├── util.py           # popcount, hash-int, progress log, labels
    │   ├── schema.py         # CSV column schemas + writer
    │   ├── manifest.py       # §7.1 manifest + §7.4 exact duplicates
    │   ├── near.py           # §7.5 near-dup search + serial SSIM scoring
    │   ├── parallel.py       # parallel SSIM scoring (per-worker shard CSVs)
    │   ├── grouping.py       # union-find groups
    │   ├── summary.py        # summary JSON
    │   └── runner.py         # orchestration + CSV/JSON export
    └── report/               # Markdown report assembly
        ├── formatting.py     # table / percent helpers
        ├── compare_report.py # two-dataset report (Proposal §6.3)
        └── audit_report.py   # single-dataset audit report
```

## Install

```bash
pip install -r requirements.txt
# or, to get the `compare-datasets` command:
pip install -e .
```

`scikit-image` is used for SSIM. If it is not installed, the tool falls back to a normalized
cross-correlation measure and says so in the report.

## Usage

```bash
# from inside the compare_datasets/ folder
python run_compare.py PATH_A PATH_B
python run_compare.py PATH_A PATH_B -o report.txt
python run_compare.py PATH_A PATH_B --match-content
python run_compare.py PATH_A PATH_B --near-dup-threshold 10 --ssim-threshold 0.95
```

Running with no arguments prints a short user manual.

## Options

| Option | Default | Meaning |
|---|---|---|
| `PATH_A`, `PATH_B` | — | the two dataset directories to compare |
| `-o`, `--output` | `dataset_comparison_report.txt` | report file (Markdown text inside) |
| `--match-content` | off | also match by content (exact SHA-256, sort-merge) to catch renamed duplicates |
| `--near-dup-threshold` | `10` | perceptual-hash Hamming distance ≤ N counts as near-duplicate |
| `--ssim-threshold` | `0.95` | SSIM below this counts as "not similar" |
| `--workers` | `1` | (audit) parallel processes for the SSIM stage; `0` = all CPU cores |
| `--top` | `20` | max rows in detail tables |

## Report sections

1. Overview
2. Total Image Count
3. Images per Class
4. File Names
5. SHA-256 Hashes
6. Perceptual Hashes
7. Image Similarity (SSIM)
8. Content Matches (exact SHA-256, sort-merge) — only with `--match-content`
9. Conclusion

## Audit mode (`--audit`)

Audit **one** dataset for internal duplication (Proposal §7.1 / §7.4 / §7.5):

```bash
python run_compare.py PATH --audit
python run_compare.py PATH --audit --manifest-only
python run_compare.py PATH --audit --export-dir audit_out
python run_compare.py PATH --audit --workers 8          # parallel SSIM stage
python run_compare.py PATH --audit --workers 0 --near-dup-threshold 4
```

It writes a Markdown report plus CSV/JSON exports:

| File | Contents |
|---|---|
| `manifest.csv` | §7.1 manifest (id, path, class, size, format, dimensions, SHA-256, pHash, dHash) |
| `exact_duplicates.csv` | §7.4 byte-identical groups |
| `near_duplicate_pairs.csv` | §7.5 candidate pairs with pHash/dHash distance and SSIM |
| `grouped_manifest.csv` | union-find groups over exact + accepted near pairs |
| `summary.json` | counts and settings |

New to the report? See **[AUDIT_REPORT_GUIDE.md](AUDIT_REPORT_GUIDE.md)** — a section-by-section
guide that maps each report part to the Project Proposal (§7.1–§7.7) and explains how to act on it.

The near-duplicate search is vectorised with NumPy (bounded memory) instead of a Python
double loop, so it is usable on datasets of tens of thousands of images. It is still an
all-pairs Hamming search; `--manifest-only` skips it entirely (recommended for very large sets).

The SSIM verification stage dominates the runtime (it loads two images per candidate pair).
`--workers N` parallelises it across processes and streams every row to
`near_duplicate_pairs.csv` via per-worker shard files, so memory stays bounded even when the
candidate list is large. Because the number of candidate pairs grows with the square of the
image count, tightening `--near-dup-threshold` is usually a bigger win than adding workers.

## Limitations

- Matching (compare mode) is **filename-based**; a duplicate saved under a different name is only
  caught with `--match-content` (exact SHA-256 only, not near-duplicates).
- "Class" must be the immediate parent folder of the image (audit mode uses the first path
  component under the dataset root as the label).
- Duplicate basenames are disambiguated by class folder.
