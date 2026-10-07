# compsci-760-group-12 "Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?"

## Data audit + Pre-processing python code

Usages please refer to: [compare_datasets](compare_datasets/README.md)

## Contribution

This branch contains the data audit and pre-processing work of the project.

**Xiayang-Peter (Xia Yang, UPI yxia728)** — tool implementation and documentation:

- **`compare_datasets` tool** (entry script `compare_datasets/run_compare.py`): compares two image-dataset directories (which may have different internal structure) across the COMPSCI 760 Proposal §6.3 checks — total image count, images per class, file names, SHA-256 hashes, perceptual hashes (pHash / dHash / aHash) and SSIM image similarity — and writes a Markdown comparison report. Files are matched by filename, with an optional content mode (sort-merge on exact SHA-256) to catch renamed duplicates.
- **Single-dataset audit mode** (`--audit`, Proposal §7.1 / §7.4 / §7.5): builds a per-image manifest (path, class, size, format, dimensions, SHA-256, pHash, dHash), detects exact duplicates (byte-identical SHA-256) and near-duplicates (perceptual-hash candidates confirmed by SSIM, catching resized / recompressed copies), and merges both into union-find `group_id` clusters so related images stay together in leakage-controlled train/validation/test splits (feeds Protocol B, Proposal §8.2).
- **Scalability work**: all-pairs Hamming search vectorised with NumPy (bounded memory, no Python double loop); the near-duplicate SSIM stage parallelised across worker processes (`--workers N`, `0` = all cores) with results streamed to per-worker shard CSVs so memory stays bounded for tens of thousands of images.
- **Documentation**: per-dataset overviews of the three candidate datasets (see *External resources used* below), plus [compare_datasets/README.md](compare_datasets/README.md) (full usage) and [AUDIT_REPORT_GUIDE.md](compare_datasets/AUDIT_REPORT_GUIDE.md) (a section-by-section guide that maps the audit report to Proposal §7.1–§7.7).

**Andy Guo** — original script, review and reliability:

- Wrote the original variant of the script.
- Did more on checking and review than on writing the final code, because *Lookers-on see most of the game*; most of the reliability and QOL fixes are of his suggestion — near-duplicate scoring no longer silently drops candidate pairs (skipped pairs and failed worker chunks are counted and reported in the audit report and summary JSON, and a failed chunk raises a warning with a non-zero exit code).
- Reviewed the other sections of the project as well — see each section's README file: [Control Experiments](https://github.com/AndyGuo-615/compsci-760-group-12/blob/litong-control-experiments/README.md), [Protocol A: ResNet-18 Training Pipeline](https://github.com/AndyGuo-615/compsci-760-group-12/blob/xiting-resnet18-pipeline/README.md), [Protocol B: Leakage-Controlled Group Split](https://github.com/AndyGuo-615/compsci-760-group-12/blob/zoe-protocol-b/README.md).


Replication: all outputs of this component are produced by the entry script `compare_datasets/run_compare.py` — compare mode `python run_compare.py PATH_A PATH_B` or audit mode `python run_compare.py <DATASET_DIR> --audit`; run from inside the `compare_datasets/` folder, see [compare_datasets/README.md](compare_datasets/README.md) for all options.

## External resources used

**Third-party Python libraries** (declared in `compare_datasets/requirements.txt`):

- [Pillow](https://python-pillow.org/) — image loading and grayscale conversion.
- [NumPy](https://numpy.org/) (>= 2.0) — vectorised Hamming search for near-duplicate candidates; fallback similarity measure when scikit-image is unavailable.
- [imagehash](https://github.com/JohannesBuchner/imagehash) — perceptual hashing (pHash / dHash / aHash).
- [scikit-image](https://scikit-image.org/) — SSIM (`skimage.metrics.structural_similarity`); optional, the tool falls back to a built-in normalized cross-correlation and says so in the report.

**Third-party datasets** (surveyed and audited; one overview document written for each):

- Krishna GitHub dataset — krishna111809 (2024), *fingerprint-based-blood-group-detection*: https://github.com/krishna111809/fingerprint-based-blood-group-detection — see [DataSet_1_Overview](DataSet_1_Overview_Krishna_GitHub_Fingerprint_Blood_Group.md).
- Sravani Kaggle dataset — Nanubala Sravani, *Fingerprint Blood Group Classification Dataset*: https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset — see [DataSet_2_Overview](DataSet_2_Overview_Sravani_Fingerprint_Blood_Group_Classification.md).
- SOCOFing (Sokoto Coventry Fingerprint Dataset) — Shehu, Ruiz-Garcia, Palade & James (2018), arXiv:1807.10609: https://www.kaggle.com/datasets/ruizgara/socofing — see [DataSet_3_Overview](DataSet_3_Overview_Sokoto_Coventry_Fingerprint_Dataset.md).

**Course materials and literature**:

- COMPSCI 760 Project Proposal — §6.3 (dataset comparison checks), §7.1 (manifest), §7.4 (exact duplicates), §7.5 (near-duplicate detection), §8.2 (leakage-controlled grouping).
- COMPSCI 760 Literature Survey (`12_LiteratureSurvey.pdf`), Section VI.A (exact and near-duplicate detection); cited: C. Zauner (2010), *Implementation and benchmarking of perceptual image hash functions*, Master's thesis, Univ. of Applied Sciences Hagenberg; Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli (2004), *Image quality assessment: From error visibility to structural similarity*, IEEE Transactions on Image Processing 13(4):600–612.

**AI tools used during development**:

- [DeepSeek API](https://platform.deepseek.com/) — the LLM API used for code assistance during development.
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — the DeepSeek model used (served via the `deepseek-flash` model name).
- [opencode](https://opencode.ai/) — terminal-based AI coding agent.
- [Oh My OpenCode](https://github.com/code-yeongyu/oh-my-opencode) — multi-agent orchestration harness for opencode.
- AI-assisted code review notes are kept at [compare_datasets/AI_CODE_REVIEW.md](compare_datasets/AI_CODE_REVIEW.md); its findings guided the subsequent reliability fixes.
