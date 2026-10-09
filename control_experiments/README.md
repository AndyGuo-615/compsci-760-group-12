# Control Experiments
This repository contains the control experiments for a fingerprint-based blood-group classification study. The main experiment trains a ResNet-18 classifier to predict ABO/Rh blood group (8 classes: A+, A-, AB+, AB-, B+, B-, O+, O-) from fingerprint images using Protocol B subject-disjoint train/validation/test splits across five seeds (111, 222, 333, 444, 555).

The control experiments test whether the main model's performance can be explained by artifacts other than genuine fingerprint ridge information, for example subject/acquisition metadata, low-level image properties, background texture, or label leakage from the splitting procedure.

# Environment
The scripts were written for Python 3.9+. Install the required packages:

bash
pip install numpy pandas scikit-learn pillow matplotlib scikit-image
pip install torch torchvision        # needed for run_label_permutation.py and run_ridge_vs_background.py
Optional:

bash
pip install opencv-python            # optional; used for a Laplacian sharpness proxy in run_metadata_control.py
run_label_permutation.py and run_ridge_vs_background.py import modules from the project's src/ package (dataset.py, model.py, transforms.py, train.py, evaluate.py). These are resolved automatically via PROJECT_ROOT = Path(__file__).resolve().parents[2].

# Prerequisites: Split Files
All control experiments (except the preprocessing and QC scripts) expect Protocol B split files at:

control_experiments/splits/protocol_b_seed{111,222,333,444,555}.csv
Required columns are filepath, label, and split (train / validation / test) for every script. Additional columns are required by some scripts: group_id for the label permutation and metadata control experiments, and subject_ids plus socofing_files for the metadata baseline, metadata control, and exploratory check. If any required column is missing, the script raises a ValueError naming the missing column.

# How to Reproduce the Results
Run the scripts from the repository root (or from anywhere, since all paths are resolved relative to the script file). The commands below assume you are in the repository root.

Full pipeline (recommended order):

bash
# 0. (Optional) Audit the split metadata
python control_experiments/scripts/exploratory_check.py

# 1. Majority-class baseline
python control_experiments/scripts/run_majority_baseline.py

# 2. Metadata-only baselines
python control_experiments/scripts/run_metadata_baseline.py    # subject_id + gender + finger
python control_experiments/scripts/run_metadata_control.py     # image/file metadata features

# 3. Label permutation control (needs src/ + torch)
python control_experiments/scripts/run_label_permutation.py

# 4. Ridge vs background control (needs preprocessing first)
python control_experiments/scripts/preprocess_ridge_background.py
python control_experiments/scripts/check_mask_coverage.py      # QC: coverage histogram
python control_experiments/scripts/mask_coverage.py            # QC: full report + examples
python control_experiments/scripts/sample_images_ridge.py      # visual sanity check
python control_experiments/scripts/run_ridge_vs_background.py
Which script replicates which result:

run_majority_baseline.py — always predict the training-set majority class. Output: results/majority_baseline.csv.

run_metadata_baseline.py — Random Forest on subject/source metadata (subject_id_num, gender, finger_position). Output: results/metadata_baseline.csv.

run_metadata_control.py — Random Forest on image/file metadata (width, height, file_size, brightness, contrast, background_intensity, image_quality). Outputs: metadata_image/image_metadata_features.csv, metadata_image/metadata_image_control.csv, metadata_image/metadata_feature_importance.csv, metadata_image/metadata_image_control_summary.csv.

run_label_permutation.py — within-split image-level label permutation plus full ResNet-18 retraining. Outputs: splits_permuted/*.csv, checkpoints_label_permutation/*.pt, results/label_permutation.csv.

run_ridge_vs_background.py — ResNet-18 trained on ridge-only and background-only images. Outputs: checkpoints_ridge_vs_background/*.pt, results/ridge_vs_background.csv.

exploratory_check.py — hand / finger / subject-ID metadata audit. Outputs: metadata_audit/*.csv.

preprocess_ridge_background.py — generate ridge-only and background-only images. Outputs: data/datasets_processed/{ridge_only,background_only}/....

check_mask_coverage.py and mask_coverage.py — segmentation QC (coverage stats, fallbacks, examples). Outputs: results/mask_coverage*.csv, results/mask_qc*.csv, results/qc_examples/.

sample_images_ridge.py — random 50-image visual sanity check of segmentation. Outputs: results/random_segmentation_check/.

# Script Details
Majority-class baseline — run_majority_baseline.py
Computes the "no-information" reference by always predicting the most frequent training label. It loops over Protocols A and B and the seeds 111, 222, 333, 444, 555, reporting accuracy, balanced accuracy, and macro F1. Output: results/majority_baseline.csv.

Metadata-only baseline — run_metadata_baseline.py
Trains a RandomForestClassifier(n_estimators=100, max_depth=10) on subject/source metadata only (subject_id_num, gender, finger_position parsed from socofing_files), without any image content. Runs on Protocol B only and requires the subject_ids and socofing_files columns. Output: results/metadata_baseline.csv.

Image/file metadata control — run_metadata_control.py
Trains a Random Forest on seven low-level image/file properties only: width, height, file_size, brightness, contrast, background_intensity, image_quality. The full fingerprint image is never shown to the classifier. image_quality is the variance of Laplacian, with a gradient-variance fallback if OpenCV is unavailable. Metadata features are cached at metadata_image/image_metadata_features.csv; delete this file to force re-extraction. Outputs go to metadata_image/.

Label permutation control — run_label_permutation.py
The strongest negative control. It keeps the original Protocol B train/validation/test splits and all group IDs but randomly permutes labels within each split, preserving the label distribution per split. The full ResNet-18 pipeline is then retrained from scratch on the permuted labels. Expected result is performance close to chance for the 8-class problem; if accuracy is substantially above chance, the split/evaluation procedure itself is leaking information. Requires the src/ package and torch/torchvision. Settings: batch size 32, lr 1e-4, 10 epochs, Adam, pretrained ResNet-18, early-stopping patience 3. Device selection is CUDA, then MPS, then CPU. It appends to results/label_permutation.csv and de-duplicates across runs.

Ridge vs background — preprocessing and training
preprocess_ridge_background.py segments every fingerprint image into ridge-only and background-only versions using: grayscale conversion, Gaussian blur (sigma 1.0), Otsu threshold (ridges darker than background), morphological closing and opening with disk(2), keeping all connected components of at least 100 pixels, dilating the mask outward with disk(6) to absorb boundary ridges, then keeping pixels inside the mask for ridge-only and pixels outside the mask (blurred with sigma 4.0) for background-only. Outputs go to data/datasets_processed/ridge_only/<blood_group>/<filename> and data/datasets_processed/background_only/<blood_group>/<filename>.

run_ridge_vs_background.py trains two ResNet-18 models per seed, one on ridge-only images and one on background-only images, using the same Protocol B splits and the same settings as run_label_permutation.py. Outputs are checkpoints_ridge_vs_background/protocol_b_seed{seed}_{condition}_best.pt and results/ridge_vs_background.csv, with per-condition mean and standard deviation printed at the end. If the background-only model performs well above chance, the main model may be exploiting background texture or acquisition artifacts rather than fingerprint ridges.

Segmentation QC — check_mask_coverage.py, mask_coverage.py, sample_images_ridge.py
These scripts verify that the ridge/background segmentation is reliable before trusting the ridge-vs-background results. check_mask_coverage.py produces a full QC report covering coverage distribution, boundary contact ratio, connected components, fallback reasons, and per-class coverage, saving a coverage histogram and up to 20 example visualizations per suspicious category. mask_coverage.py produces a simpler coverage analysis with mean, median, min, and max coverage, fallback rate, the percentage of images with coverage above 90%, equal to 100%, or below 10%, and per-class coverage stats. sample_images_ridge.py randomly samples 50 images (seed 111) and produces a 4-column grid showing original, mask, ridge-only, and background-only versions for visual inspection. A coverage of exactly 1.0 (100%) or very high coverage usually indicates the segmentation fallback (np.ones_like(gray, dtype=bool)) was triggered, and those images should be treated as unreliable.

Metadata audit — exploratory_check.py
A purely descriptive audit of the Protocol B splits with no model training. It checks whether hand, finger type, or subject ID are confounded with blood-group labels, and writes per-seed CSV tables to control_experiments/metadata_audit/.

# Interpreting the Controls
A clean result looks like this. The majority baseline should sit near 12.5% balanced accuracy (chance for 8 classes). The metadata-only control (subject, gender, finger) should be near chance, meaning labels are not predictable from source metadata. The image/file metadata control should be near chance, meaning labels are not predictable from acquisition artefacts. The label permutation control should be near chance, meaning the training and evaluation pipeline does not leak information. In the ridge vs background comparison, ridge-only should substantially outperform background-only, meaning the main model uses ridge patterns rather than background texture. If any control performs substantially above chance, the corresponding confound should be addressed before drawing conclusions from the main experiment.

# Notes
All scripts use PROJECT_ROOT = Path(__file__).resolve().parents[N], so they can be run from any working directory. Seeds are fixed at 111, 222, 333, 444, 555 throughout for consistency with the main experiment. run_label_permutation.py and run_ridge_vs_background.py append to their result CSVs and de-duplicate on (experiment, protocol, [condition,] seed) with keep="last", so re-running a subset of seeds is safe. run_metadata_control.py caches the extracted image metadata; delete metadata_image/image_metadata_features.csv if you change the image set or want a clean re-extraction.