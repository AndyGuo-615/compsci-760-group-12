# Dataset Reference — SOCOFing (Sokoto Coventry Fingerprint Dataset)

> **Role in project:** CONTROL dataset (Proposal §6.2; literature survey reference [19])
> **Verified:** 2026-09-11 against the local archive and official sources.

---

## 1. Summary (Dataset Metadata)

| Field | Value |
|---|---|
| Title | Sokoto Coventry Fingerprint Dataset (SOCOFing) |
| Authors | Yahaya Isah Shehu, Ariel Ruiz-Garcia, Vasile Palade, Anne James |
| Year | 2018 |
| Paper | arXiv:1807.10609 — https://arxiv.org/abs/1807.10609 |
| DOI | https://doi.org/10.48550/arXiv.1807.10609 |
| Source | https://www.kaggle.com/datasets/ruizgara/socofing |
| Local archive | `Sokoto Coventry Fingerprint Dataset.zip` (878 MB compressed) |
| License | **Custom / "Other (specified in description)"** — noncommercial research only (see §2) |
| Real images | 6,000 (600 subjects × 10 fingers) |
| Altered images | 49,270 |
| Image format | BMP, 96 × 103 px |

---

## 2. License

The operative terms are stated in the paper (Shehu et al., 2018, §3 "Usage"):

> "This dataset is provided AS IS for noncommercial not-for-profit research purposes only. Any publications arising from the use of this software, including but not limited to academic journal and conference publications, technical reports and manuals, must cite this document and the following work: [ICANN 2018 paper]."

> [!warning] License ambiguity — flag in the report
> The arXiv **preprint document** is licensed **CC BY-NC-SA 4.0**. That license applies to the *paper*, not explicitly to the *data*. Some secondary sources describe the dataset itself as CC BY-NC-SA 4.0, but the primary sources do not state this. The defensible description is: **custom noncommercial, not-for-profit research-only terms**.

**Implications for the project:** our use is academic, noncommercial research — should within the permitted scope. We must cite both original works below.

### Required citations

1. Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A. (2018). *Sokoto Coventry Fingerprint Dataset*. arXiv:1807.10609.
2. Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A. (2018). *Detection of Fingerprint Alterations Using Deep Convolutional Neural Networks*. ICANN 2018, LNCS 11139, 51–60. https://doi.org/10.1007/978-3-030-01418-6_6

---

## 3. Metadata (Sokoto Coventry Fingerprint Dataset.zip)

### Contents

| Subset | Images | Notes |
|---|---:|---|
| Real | 6,000 | 600 subjects × 10 fingers |
| Altered-Easy | 17,931 | synthetic alterations |
| Altered-Medium | 17,067 | synthetic alterations |
| Altered-Hard | 14,272 | synthetic alterations |
| **Total (incl. Altered)** | **55,270** | paper states 55,273 (3-image difference) |

### Image properties (File name interperation)

| Property | Real | Altered |
|---|---|---|
| Format | BMP, 32-bit | BMP, 8-bit (256 colours) |
| Dimensions | 96 × 103 px (5,956); 241 × 298 px (44) | 96 × 103 px (48,881); larger (389) |
| File size | 39,690 B (96×103); 215,890 B (241×298) | 10,966 B (96×103) |
| Naming | `<id>__<M/F>_<Hand>_<finger>_finger.BMP` | adds alteration suffix (e.g. `_CR`, `_Obl`) |
| Ground-truth labels | subject ID, gender, hand, finger — encoded in filename | same |

Example filename: `100__M_Left_index_finger.BMP`

> Archive duplication

> The downloaded archive contains the dataset **twice**: a root `SOCOFing/{Real,Altered}` **and** an identical nested `socofing/SOCOFing/{Real,Altered}` (same 6,000 Real filenames; a sampled large file matched by SHA-256). Keep only one copy when extracting.

---

## 4. Provenance

- Collected in Sokoto, Nigeria, from 600 African subjects (all 18 years or older).
- Institutions: Coventry University (Faculty of Engineering, Environment and Computing) and Nottingham Trent University (Faculty of Science and Technology).
- Captured at 500 dpi; scanners: Hamster Plus (HSDU03PTM) and SecuGen SDU03PTM.
- Altered images generated with the STRANGE toolbox.
- No explicit IRB/consent statement appears in the paper (only that subjects were 18+).

---

## 5. Contribution (@ Proposal Chapter 6.2)

SOCOFing is the **control dataset**. It has **no blood-group labels**, so it is **not** used to measure blood-group accuracy. It is used to:

- Test fingerprint matching and subject grouping
- Compare predicted groups against the real subject labels (ground truth)
- Compare random image splitting with subject-based splitting
- Test whether the leakage-detection / grouping method works correctly

This underpins the validation of the grouping method (Proposal §7.7) before that method is applied to the main dataset.

---
