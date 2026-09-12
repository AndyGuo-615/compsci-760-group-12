# Dataset Reference — Krishna111809 GitHub Fingerprint Blood Group Dataset

> **Role in project:** Optional backup / external comparison (Proposal §6.3)
> **Verified:** 2026-09-11 against the local archive and the GitHub repository.

---

## 1. Summary (Dataset Metadata)

| Field | Value |
|---|---|
| Title | fingerprint-based-blood-group-detection |
| Owner | Krishna Murthi (GitHub `krishna111809`) |
| Source | https://github.com/krishna111809/fingerprint-based-blood-group-detection |
| Local archive | `Krishna Murthi - fingerprint-based-blood-group-detection-main.zip` (161 MB) |
| License | **MIT** (see §2) |
| Files | 6,000 `.BMP` images |
| Classes | 8 (A+, A−, AB+, AB−, B+, B−, O+, O−) |
| Image format | BMP, 96 × 103 px, 32-bit |
| Last push | 2024-11-12  |

---

## 2. License

Under **MIT License** — `Copyright (c) 2024 Krishna Murthi`. Verified from the `LICENSE` file inside the archive.

MIT permits use, copying, modification, and redistribution provided the copyright notice and permission notice are retained. There is **no dataset-specific license or provenance** in the repository; the MIT license covers the repository contents (including the images as bundled).

> The blood-group images may originate from a third party, so the MIT grant may not be authoritative for the image data itself. This better noted down in the project paper

---

## 3. Metadata (verified from the local archive)

### Class distribution

| Class | Images |
|---|---:|
| A+ | 565 |
| A− | 1,009 |
| AB+ | 708 |
| AB− | 761 |
| B+ | 652 |
| B− | 741 |
| O+ | 852 |
| O− | 712 |
| **Total** | **6,000** |

### Image properties

| Property | Value |
|---|---|
| Format | BMP, 32-bit |
| Dimensions | 96 × 103 px (majority); 241 × 298 px (44) |
| File size | 39,690 B (96×103); 215,890 B (241×298) |
| Naming | `cluster_<classIndex>_<n>.BMP` |
| Folder structure | `dataset/dataset_blood_group/<class>/<file>` |

### Class index → blood group mapping

> [!note] The cluster index is **not alphabetical**. Verified from the class folders:
> `cluster_0` = A+ · `cluster_1` = A− · `cluster_2` = B+ · `cluster_3` = B− · `cluster_4` = AB+ · `cluster_5` = AB− · `cluster_6` = O+ · `cluster_7` = O−

### Repository structure

```
fingerprint-based-blood-group-detection-main/
├── code/                          Alexnet, Lenet, Resnet34, Vgg16 Jupyter notebooks
├── dataset/dataset_blood_group/   6,000 images in 8 class folders
├── graphs/                        accuracy / loss / val_* plots per model
├── performance metrix/            comparison bar charts
├── sample dataset/                sample_data.jpg
├── test/                          model_blood_group_detection_resnet.h5 (~98 MB) + test.ipynb
├── requirements.txt
├── README.md
└── LICENSE                        (MIT)
```

---

## 4. README discrepancies

- README claims "~6,000–7,000" images; actual = exactly **6,000**.
- README says PyTorch; `requirements.txt` pins **TensorFlow 2.4.1** (plus numpy 1.19.5, pandas 1.2.4, scikit-learn 0.24.2).
- README folder path typo `datasetbloodgroup` vs actual `dataset_blood_group`.

---

## 5. Contribution to our project (@Proposal Chapter 6.3)

The proposal lists this as an **optional backup / possible external test dataset**, not used in the main experiments. Before any use, it was to be compared with the main Sravani dataset (image counts, class counts, filenames, SHA-256, perceptual hashes, similarity).

> [!danger] That comparison is already conclusive: this is NOT an independent dataset.
> - **All 5,837 Sravani files are byte-identical members of this repository's dataset** — same class/filename **and** same SHA-256 hash (0 mismatches, 0 Sravani-only files; verified 2026-09-11).
> - This repository has 6,000 files; the **163 extra are all A+** (`cluster_0_*`).
> - Same `cluster_<idx>_<n>.BMP` naming and identical counts in the other 7 classes.
>
> Per the proposal, it must therefore be treated as **another version of the main dataset, not an independent source**, and must **not** be combined directly with the Sravani dataset. Its main remaining value is as the **6,000-image superset** corresponding to the counts reported by Phadke et al. (2025).

---

## 6. Citation

> krishna111809. (2024). *fingerprint-based-blood-group-detection* [Computer software]. GitHub. https://github.com/krishna111809/fingerprint-based-blood-group-detection
