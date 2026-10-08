# Protocol B: Leakage-Controlled Fingerprint Classification

This branch implements **Protocol B** for the COMPSCI 760 Group 12 project:

> **Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?**

Protocol B replaces Protocol A's random image-level split with a group-based split designed to prevent images linked to the same person, exact duplicates, and accepted near-duplicates from crossing the training, validation, and test partitions. The model architecture and training settings are kept aligned with Protocol A so that the split strategy is the main experimental difference.

## Main findings

- All 5,837 blood-group images were linked to SOCOFing identities by exact SHA-256 matching.
- The 600 SOCOFing subject IDs were conservatively merged into 587 leakage-control groups where ambiguous identity links existed.
- Five group-based splits were generated with seeds 111, 222, 333, 444, and 555.
- No leakage-control group crossed partitions in any Protocol B split.
- An independent near-duplicate audit accepted three image pairs. Protocol A had 1–2 cross-partition pairs per seed, whereas Protocol B had zero for all five seeds.

### Protocol B results

| Seed | Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|
| 111 | 0.8729 | 0.8695 | 0.8759 |
| 222 | 0.8853 | 0.8893 | 0.8873 |
| 333 | 0.8830 | 0.8874 | 0.8835 |
| 444 | 0.8623 | 0.8554 | 0.8624 |
| 555 | 0.8648 | 0.8667 | 0.8675 |
| **Mean** | **0.8737** | **0.8737** | **0.8753** |
| **Sample SD** | **0.0104** | **0.0144** | **0.0105** |

These results measure prediction of the labels supplied with the dataset. They are **not evidence that fingerprints can determine a person's real blood group**. The recovered subject groups include conflicting blood-group labels, which raises concerns about the validity of the dataset labels.

## Contribution and code provenance

### Zoe Wang's contribution

Zoe Wang was responsible for Protocol B and contributed the following work:

- designed and implemented the leakage-controlled group-splitting procedure;
- linked the main dataset to SOCOFing subject identities and created the reproducible subject-group manifest workflow;
- generated and validated five Protocol B splits using seeds 111, 222, 333, 444, and 555;
- ran the five Protocol B ResNet-18 experiments and summarized accuracy, balanced accuracy, macro F1, per-class recall, and confusion matrices;
- implemented the independent cross-split near-duplicate audit for both Protocol A and Protocol B;
- documented the workflow, provenance, limitations, and reproduction commands.

The main files created for this contribution are:

```text
scripts/build_main_subject_groups.py
scripts/create_group_split.py
scripts/run_protocol_b.py
scripts/audit_near_duplicate_splits.py
data/manifests/main_subject_groups.csv
data/manifests/subject_match_links.csv
data/manifests/person_consistency_audit.csv
data/manifests/subject_manifest_summary.json
splits/protocol_b_seed*.csv
results/protocol_b_seed*.txt
results/near_duplicate_split_audit/
```

### Reused team code

To make the comparison fair, Protocol B reuses the common ResNet-18 training pipeline developed for **Protocol A by Xiting Li**:

```text
src/dataset.py
src/model.py
src/transforms.py
src/train.py
src/evaluate.py
```

The Protocol A split files and `scripts/create_random_split.py` / `scripts/run_protocol_a.py` are retained for comparison. The image-audit workflow used to produce the source manifests and independently detected near-duplicate pairs was developed in **Xia Yang's data-processing work**. Protocol B adds the scripts that convert those audit outputs into subject groups and test whether accepted near-duplicate pairs cross partitions.

### Online resources and third-party software

No online code was copied and presented as original work. The project uses these public resources and libraries:

- [Fingerprint-Based Blood Group Dataset on Kaggle](https://www.kaggle.com/datasets/rajumavinmar/fingerprint-based-blood-group-dataset): the eight-class dataset evaluated in this project. The repository does not redistribute the images.
- [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing) and its [dataset paper](https://arxiv.org/abs/1807.10609): the `Real` images and filename metadata were used to recover subject identities through exact image-content matching. The repository does not redistribute SOCOFing images.
- [PyTorch](https://pytorch.org/) and [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): model implementation and ImageNet-pretrained weights. The final classifier layer was replaced with an eight-class output layer.
- [scikit-learn](https://scikit-learn.org/): balanced accuracy, macro F1, recall, and confusion-matrix calculations.
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [Pillow](https://python-pillow.org/): numerical operations, CSV processing, and image loading.

The SHA-256 matching, conservative connected-group construction, group assignment objective, validation checks, and cross-split audit logic in this branch were implemented for this project.

## Repository structure

```text
data/
  datasets/                         # downloaded images; not committed
  manifests/                        # subject links and leakage-control groups
results/
  near_duplicate_split_audit/       # Protocol A/B leakage-audit results
  protocol_b_seed*.txt              # saved training and evaluation output
scripts/
  build_main_subject_groups.py      # rebuild subject-group manifest
  create_group_split.py             # generate five Protocol B splits
  run_protocol_b.py                 # main experiment reproduction script
  audit_near_duplicate_splits.py    # audit accepted near-duplicates
splits/
  protocol_a_seed*.csv
  protocol_b_seed*.csv
src/                                # shared model/training implementation
```

## Requirements

Python 3.10 or later is recommended. Install the required packages in a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision pandas numpy scikit-learn pillow
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision pandas numpy scikit-learn pillow
```

## Dataset setup

Download the fingerprint blood-group dataset separately and place the eight class folders under `data/datasets/`:

```text
data/datasets/
├── A+
├── A-
├── AB+
├── AB-
├── B+
├── B-
├── O+
└── O-
```

The expected class counts are:

| Class | Images |
|---|---:|
| A+ | 402 |
| A- | 1,009 |
| AB+ | 708 |
| AB- | 761 |
| B+ | 652 |
| B- | 741 |
| O+ | 852 |
| O- | 712 |
| **Total** | **5,837** |

Verify the installation and dataset layout:

```bash
python scripts/check_dataset.py
```

## Reproducing the reported Protocol B results

The main result-replication script is:

```text
scripts/run_protocol_b.py
```

The committed split CSV files can be used directly. Run the following command once for each seed:

```bash
python scripts/run_protocol_b.py --seed 111 --batch-size 32 --epochs 10 --lr 0.0001 --optimizer adam --patience 3 --pretrained
python scripts/run_protocol_b.py --seed 222 --batch-size 32 --epochs 10 --lr 0.0001 --optimizer adam --patience 3 --pretrained
python scripts/run_protocol_b.py --seed 333 --batch-size 32 --epochs 10 --lr 0.0001 --optimizer adam --patience 3 --pretrained
python scripts/run_protocol_b.py --seed 444 --batch-size 32 --epochs 10 --lr 0.0001 --optimizer adam --patience 3 --pretrained
python scripts/run_protocol_b.py --seed 555 --batch-size 32 --epochs 10 --lr 0.0001 --optimizer adam --patience 3 --pretrained
```

The formal experiments used:

- image size: 224 × 224;
- ImageNet-pretrained ResNet-18;
- batch size: 32;
- Adam optimizer;
- learning rate: 0.0001;
- maximum epochs: 10;
- early-stopping patience: 3;
- cross-entropy loss;
- no class weights and no data augmentation.

The best checkpoint for each run is written to `checkpoints/`. Checkpoints and dataset images are intentionally excluded from Git.

### Regenerating the Protocol B splits

With the dataset in `data/datasets/` and the committed `data/manifests/main_subject_groups.csv` present, run:

```bash
python scripts/create_group_split.py
```

This script regenerates all five `splits/protocol_b_seed*.csv` files and fails if a filepath is duplicated, an image is unassigned, a partition is empty, or a group crosses partitions.

## Rebuilding the subject-group manifest

The manifest-replication script is:

```text
scripts/build_main_subject_groups.py
```

It requires two `manifest.csv` files produced by the data-audit pipeline: one for the 5,837-image blood-group dataset and one for the 6,000 SOCOFing `Real` images. It joins images only when their SHA-256 hashes match exactly and conservatively merges ambiguous subject links.

```bash
python scripts/build_main_subject_groups.py \
  --main-manifest path/to/main_audit/manifest.csv \
  --socofing-manifest path/to/socofing_real_audit/manifest.csv \
  --output-dir data/manifests \
  --dataset-prefix data/datasets
```

Expected key summary values are:

- 5,837 matched main-dataset images;
- 6,000 SOCOFing Real images;
- 600 matched subject IDs;
- 5,882 subject-link rows;
- 26 images with multiple candidate subject IDs;
- 587 final leakage-control groups;
- 0 unmatched images.

## Reproducing the near-duplicate leakage audit

First use the data-audit pipeline to create a complete `near_duplicate_pairs.csv` for the 5,837-image dataset with these fixed thresholds:

- pHash distance ≤ 10;
- dHash distance ≤ 12;
- SSIM ≥ 0.95.

The completed audit produced 21,205 candidate pairs and three accepted near-duplicate pairs. Then run:

```bash
python scripts/audit_near_duplicate_splits.py \
  --near-pairs path/to/main_full_audit/near_duplicate_pairs.csv \
  --splits-dir splits \
  --output-dir results/near_duplicate_split_audit
```

Expected cross-split counts are:

| Protocol | Seed 111 | Seed 222 | Seed 333 | Seed 444 | Seed 555 |
|---|---:|---:|---:|---:|---:|
| A | 2 | 1 | 1 | 2 | 2 |
| B | 0 | 0 | 0 | 0 | 0 |

The audit also expects zero unresolved paths. Detailed pair-level records and the summary files are committed under `results/near_duplicate_split_audit/`.

## Reproducibility notes

- The seeds control split generation and `torch.manual_seed` in the training entry script.
- Exact numerical reproduction can still vary slightly across PyTorch versions, GPU models, CUDA/cuDNN versions, and nondeterministic accelerator operations.
- The repository contains the split files and text outputs used for the reported table, allowing the experimental inputs and reported metrics to be inspected without retraining.
- There are no files named `Untitled`, and the project README files contain usage instructions rather than empty placeholders.

## Ethical and scientific limitation

This repository evaluates a dataset and a leakage-control methodology. It does not provide a clinically validated blood-group test. Blood type must be determined using accepted laboratory methods; model outputs from this project must not be used for medical decisions.
