# Dataset Reference — Sravani Fingerprint Blood Group Classification Dataset

> **Role in project:** MAIN dataset (Proposal §6.1)
> **Verified:** 2026-09-11 against the local archive and Kaggle's official metadata API.

---

## 1. Summary (Dataset Metadata)

| Field | Value |
|---|---|
| Title | Fingerprint Blood Group Classification Dataset |
| Uploader | Nanubala Sravani (Kaggle username `sravani2006`) |
| Source | https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset |
| Upload / last update | 2026-07-06 (version 1, "Initial release") |
| Local archive | `Nanubala Sravani - Fingerprint Blood Group Classification Dataset.zip` (59 MB compressed) |
| License | **Unknown** (see §2) |
| Files | 5,837 `.BMP` images |
| Classes | 8 (A+, A−, AB+, AB−, B+, B−, O+, O−) |
| Image format | BMP, predominantly 96 × 103 px, 32-bit |
| Uncompressed size | ~228.3 MiB (239,423,330 bytes) |

---

## 2. License

**Kaggle license field: `Unknown`** — confirmed via Kaggle's dataset metadata API (`licenseName: "Unknown"`, `hasLicenseName: true`). No standard license is assigned and no license URL is shown.

> [!warning] No license is assigned to this dataset, and the uploader does not claim ownership.
> The dataset description states it is *"a private research copy of a fingerprint image dataset that was originally obtained from Kaggle"* and that it is *"uploaded solely for use in my private Kaggle notebooks and is not claimed as original work or ownership."*

**Implications for the project:**
- There is **no explicit permission to redistribute or publish** the images.
- Usage rights depend on the license of the (unnamed) original source dataset.
- Treat as **research use only**; do not redistribute. In the report, cite the Kaggle page and state the license as **"Unknown"**.
- Because the uploader is not the owner, the provenance chain is incomplete (see §4).

---

## 3. Metadata (verified from the local archive)

### Class distribution

| Class | Images |
|---|---:|
| A+ | 402 |
| A− | 1,009 |
| AB+ | 708 |
| AB− | 761 |
| B+ | 652 |
| B− | 741 |
| O+ | 852 |
| O− | 712 |
| **Total** | **5,837** |

### Image properties

| Property | Value |
|---|---|
| Format | BMP, 32-bit (RGB + alpha) |
| Dimensions | 96 × 103 px (majority) |
| File size | 39,690 B (majority — 5,793 images) |
| Outliers | 44 images are 241 × 298 px, 24-bit, 215,890 B |
| Naming | `cluster_<classIndex>_<n>.BMP` (e.g. `cluster_0_2609.BMP`) |
| Folder structure | `datasets/<class>/<file>` |
| Non-image files | none |

### Class index → blood group mapping

> [!note] The cluster index is **not alphabetical**. Verified from the class folders:
> `cluster_0` = A+ · `cluster_1` = A− · `cluster_2` = B+ · `cluster_3` = B− · `cluster_4` = AB+ · `cluster_5` = AB− · `cluster_6` = O+ · `cluster_7` = O−

---

## 4. Provenance & uploader description (verbatim)

> "This dataset is a private research copy of a fingerprint image dataset that was originally obtained from Kaggle and is organized into eight blood group classes (A+, A-, AB+, AB-, B+, B-, O+, O-). It is intended for academic research and machine learning experiments in fingerprint-based blood group classification, including model training, evaluation, baseline comparisons, ablation studies, and statistical validation. This dataset is uploaded solely for use in my private Kaggle notebooks and is not claimed as original work or ownership."

- **Derived dataset:** yes — a copy of an unnamed Kaggle dataset.
- **Original source:** not named by the uploader.
- **Label provenance:** not documented (how the blood-group labels were obtained is not stated).
- **No DOI / citation** is provided on the Kaggle page.

---

## 5. Contribution to our project (Proposal §6.1)

This is the **main dataset** for the blood-group classification experiments. It is used for:

- Training, validating and testing the blood-group model
- Random image splitting (Protocol A)
- Leakage-controlled group splitting (Protocol B)
- Metadata-only baseline experiments
- Image-masking (ridge-only / background-only) experiments

The exact image count, dimensions, format and class sizes were listed in the proposal as "to be confirmed after download" — they are now confirmed above.

---

## 6. Critical notes for the report

1. **Not independent from the Krishna GitHub dataset — it is a byte-identical subset.** All 5,837 files are present in the Krishna111809 GitHub dataset with the **same class/filename AND the same SHA-256 hash** (0 mismatches, 0 Sravani-only files; verified 2026-09-11). The Krishna dataset has 163 additional files, all A+ (`cluster_0_*`). This dataset is therefore exactly the Krishna dataset minus 163 A+ images. See `Dataset - Krishna GitHub Fingerprint Blood Group.md`.
2. **Counts differ from Phadke et al. (2025).** The literature survey cites Phadke's dataset as A− 1,009 / A+ 565 (6,000 images total). Those counts match the **Krishna** dataset, not this one (A+ 402, 5,837 total). The dataset behind the reported 88% accuracy is therefore the 6,000-image version, not this Kaggle copy.
3. **License is Unknown** and the uploader disclaims ownership — a reproducibility and ethics caveat that should be stated explicitly.

---

## 7. Citation

> Nanubala Sravani. (2026, July 6). *Fingerprint Blood Group Classification Dataset*. Kaggle. https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset
