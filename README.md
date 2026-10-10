# Protocol A: ResNet-18 Training Pipeline

This branch contains the implementation and experimental results for **Protocol A: Random Image Split** in the COMPSCI 760 Group 12 project.

Protocol A uses stratified random image splitting to evaluate an ImageNet-pretrained ResNet-18 model on the eight blood-group labels provided by the fingerprint dataset. The training pipeline was also reused for Protocol B, supporting a controlled comparison between the two data-splitting strategies.

---

## Contribution and Code Provenance

### Xiting Li's Contribution

Xiting Li was responsible for Protocol A and contributed the following work:

1. **Developed and tested a reusable ResNet-18 training and evaluation pipeline**, including image preprocessing, model configuration, training, early stopping, and performance evaluation. The pipeline was subsequently reused by Protocol B to support consistent model and training settings across the two protocols.
2. **Implemented stratified random image splitting**, dividing the dataset into training, validation, and test sets in a 70/15/15 ratio and generating split files for five predefined seeds (111, 222, 333, 444, and 555).
3. **Conducted hyperparameter screening**, testing six combinations of three learning rates and two batch sizes and selecting the final configuration based on validation performance.
4. **Trained and evaluated Protocol A across five seeds** using the selected configuration to assess performance under different random image splits.
5. **Calculated and summarised evaluation metrics**. The code computes Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, and Macro F1. Accuracy, Balanced Accuracy, and Macro F1 were selected as the three main reporting metrics. Per-class recall and confusion matrices were also examined.
6. **Prepared project documentation** covering the experimental setup, code structure, results, usage instructions, and steps for reproducing the experiments.

### Code Provenance and AI Assistance

The Protocol A scripts and reusable training pipeline were **developed, tested, and debugged by Xiting Li**, with assistance from **ChatGPT** during code development. No code was directly copied from external GitHub repositories or online tutorials.

The implementation uses third-party Python libraries and pretrained model weights, which are acknowledged below.

---

## External Resources

Protocol A uses a publicly available dataset, third-party Python libraries, and pretrained model weights. The main resources and their purposes are listed below.

### 1. Dataset

- **[Fingerprint Blood Group Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset)**, published by Nanubala Sravani: the data source used for Protocol A. The experiments used 5,837 fingerprint images organised under eight dataset-provided blood-group labels for model training and evaluation.

The original fingerprint images are not included in this GitHub repository and must be downloaded separately. See **Dataset Setup** for the required folder structure.

### 2. Third-party Libraries and Pretrained Model

- **[PyTorch](https://pytorch.org/)**: used for model training and evaluation, including the loss function, optimizer, backpropagation, and model checkpoint saving.
- **[torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)**: provides the ResNet-18 architecture and ImageNet-pretrained weights; torchvision transforms are also used for image preprocessing. The original classification layer was replaced with an eight-class output layer.
- **[scikit-learn](https://scikit-learn.org/)**: used for stratified random splitting and the calculation of Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, and confusion matrices.
- **[pandas](https://pandas.pydata.org/)**: used to read, process, and save data-split CSV files.
- **[Pillow](https://pillow.readthedocs.io/)**: used to load fingerprint images and convert them to RGB format.

These libraries and pretrained weights were used through their public interfaces. No code was directly copied from external repositories or online tutorials.

---

## Requirements

The Protocol A experiments were conducted on **macOS using a CPU**. The main software environment was:

- Python 3.9.23
- PyTorch 2.2.2
- torchvision 0.17.2
- Device: CPU

The code also uses pandas, scikit-learn, and Pillow for data processing, splitting, image loading, and evaluation.

The main dependencies can be installed with:

```bash
pip install torch torchvision pandas scikit-learn pillow
```

This command installs the required packages, but their installed versions may differ from those used in the original experiments.

The training script automatically selects an available compute device in the following order: CUDA, MPS, then CPU. Runtime may vary depending on the device used.

---

## Code Structure

This branch contains code for Protocol A data splitting, ResNet-18 training, evaluation, and result summarisation. The main repository structure is:

```text
├── data/
│   └── README.md
│
├── results/
│   ├── hyperparameter_search.csv
│   ├── protocol_a_seed*.txt
│   └── protocol_a_summary.csv
│
├── scripts/
│   ├── check_dataset.py
│   ├── create_random_split.py
│   ├── run_protocol_a.py
│   └── summarize_protocol_a.py
│
├── splits/
│   └── protocol_a_seed*.csv
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
└── README.md
```

### 1. `scripts/`: Experiment Scripts

- `check_dataset.py`: checks whether the dataset folders exist and counts images in each blood-group category.
- `create_random_split.py`: generates stratified random split files for Protocol A using the five predefined seeds.
- `run_protocol_a.py`: trains and evaluates ResNet-18 for one seed. **This is the main script for reproducing the Protocol A experiments.**
- `summarize_protocol_a.py`: aggregates the results across five seeds and calculates the mean, sample standard deviation, and 95% confidence interval for each recorded metric.

### 2. `src/`: Reusable Training and Evaluation Pipeline

- `dataset.py`: loads fingerprint images and class labels according to a split CSV file.
- `model.py`: builds ResNet-18 and replaces the final classification layer with an eight-class output layer.
- `transforms.py`: defines image resizing, tensor conversion, and ImageNet normalization.
- `train.py`: implements training, validation, early stopping, and best-model checkpoint saving.
- `evaluate.py`: calculates Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, per-class recall, and confusion matrices.

These modules form the shared training and evaluation pipeline subsequently reused by Protocol B. Protocol B retains the core model and training implementation, with adjustments to the reported evaluation outputs.

### 3. `splits/`: Data Split Files

Contains the five Protocol A CSV files corresponding to seeds 111, 222, 333, 444, and 555. Each file records the image path, class label, and assigned partition (training, validation, or test).

### 4. `results/`: Experiment Outputs

- `hyperparameter_search.csv`: records the validation results for the six learning-rate and batch-size combinations.
- `protocol_a_seed*.txt`: contains training and validation logs, test metrics, per-class recall, and confusion matrices for the five seeds.
- `protocol_a_summary.csv`: stores the per-seed metrics and their mean, sample standard deviation, and 95% confidence intervals.

### 5. `data/`: Dataset Documentation

- `README.md`: describes the dataset source and the required local folder structure.

The original fingerprint images and generated best-model checkpoints are not committed to GitHub. Images must be downloaded separately, and checkpoints can be generated by rerunning the training script.

---

## Dataset Setup

Download the **[Fingerprint Blood Group Classification Dataset](https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset)** and place the eight blood-group folders under `data/datasets/` in the repository root:

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

The experiments used the following image counts:

| Blood group | Images |
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

After preparing the dataset, run the following command from the repository root:

```bash
python scripts/check_dataset.py
```

This script checks whether the eight class folders exist and reports the number of BMP images in each folder and in total. Compare the printed counts with the table above to check the expected dataset structure and size.

---

## Experimental Setup

### 1. Random Image Splitting

Protocol A uses a **stratified random image split** based on the dataset-provided eight blood-group labels. The 5,837 fingerprint images are divided into training, validation, and test sets using a 70/15/15 ratio:

| Partition | Target ratio | Images |
|---|---:|---:|
| Training | 70% | 4,085 |
| Validation | 15% | 876 |
| Test | 15% | 876 |
| **Total** | **100%** | **5,837** |

Stratification keeps the class proportions relatively consistent across the three partitions. Images are split independently without grouping by subject identity, so subject overlap between partitions is possible.

Five predefined random seeds were used: **111, 222, 333, 444, and 555**. Each seed has a separate split CSV file in `splits/`.

### 2. Hyperparameter Selection

Before the final five-seed evaluation, learning rate and batch size were screened using the following values:

- Learning rates: 0.0001, 0.0003, 0.001
- Batch sizes: 16, 32

This gave **3 × 2 = 6 configurations**.

To keep the comparisons consistent and limit computational cost, all six configurations were evaluated using the **seed 111 split**. Validation loss and validation accuracy were compared to select the final configuration. The test set was not used for hyperparameter selection.

Other main settings were kept fixed during screening: ImageNet-pretrained ResNet-18, Adam optimizer, a maximum of 10 epochs, and early-stopping patience of 3.

The selected configuration was:

- **Learning rate: 0.0001**
- **Batch size: 32**

The six screening results are recorded in `results/hyperparameter_search.csv`.

### 3. Final Model Training

The selected configuration was used to train and evaluate Protocol A separately on each of the five seeded splits.

| Parameter | Final setting |
|---|---|
| Model | ImageNet-pretrained ResNet-18 |
| Input size | 224 × 224 |
| Output classes | 8 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Maximum epochs | 10 |
| Early-stopping patience | 3 |

Before being passed to the model, each image is converted to RGB and resized to 224 × 224.

Early stopping is based on validation loss. Training stops when validation loss fails to improve for three consecutive epochs. The model checkpoint with the lowest validation loss is saved and then loaded for the final test evaluation.

### 4. Model Evaluation

Protocol A uses three main evaluation metrics:

- **Accuracy**: the proportion of correctly classified test images.
- **Balanced Accuracy**: the average recall across classes, giving each class equal weight.
- **Macro F1**: the average F1 score across classes, giving each class equal weight while considering both precision and recall.

The code also calculates Macro Precision and Macro Recall and outputs per-class recall and a confusion matrix for further analysis.

The test results across the five seeds were summarised using the **mean, sample standard deviation (SD), and 95% confidence interval (CI)** to describe performance and variation across random splits.

---

## Reproducing Protocol A Results

The following steps describe how to regenerate the Protocol A splits, train and evaluate the model, and summarise the results across five seeds.

Before running these commands, prepare the dataset as described in **Dataset Setup** and run them from the repository root.

### 1. Generate Data Splits

Five split CSV files are already provided in `splits/` and can be used directly. To regenerate them, run:

```bash
python scripts/create_random_split.py
```

The script produces stratified 70/15/15 splits for seeds 111, 222, 333, 444, and 555.

### 2. Run Protocol A

The main reproduction script is **`scripts/run_protocol_a.py`**. It trains and evaluates the model for one seed, applies early stopping, and saves the best checkpoint.

For example, to run seed 111:

```bash
python -u scripts/run_protocol_a.py \
  --seed 111 \
  --batch-size 32 \
  --epochs 10 \
  --lr 0.0001 \
  --optimizer adam \
  --patience 3 \
  --pretrained \
  | tee results/protocol_a_seed111.txt
```

This command uses the final training configuration and saves the terminal output to `results/protocol_a_seed111.txt`.

To reproduce the other runs, change `--seed` to 222, 333, 444, or 555 and update the corresponding output filename.

**Note:** `tee` overwrites an existing file of the same name. Back up the committed experiment logs before rerunning the command if you want to preserve them.

The best model checkpoint for each seed is saved under `checkpoints/`. This directory is created automatically by the training script and is not included in the GitHub repository.

### 3. Summarise the Five Runs

Once all five runs have completed and their output logs have been saved, run:

```bash
python scripts/summarize_protocol_a.py
```

This script reads the five `results/protocol_a_seed*.txt` files and calculates the per-seed metrics, mean, sample standard deviation, and 95% confidence intervals. The resulting summary is saved as:

```text
results/protocol_a_summary.csv
```

The branch already contains the original five training and evaluation logs and the summary CSV, so the reported results can also be inspected without retraining.

---

## Results

Protocol A was trained and evaluated using the final configuration across five predefined seeds (111, 222, 333, 444, and 555).

Only the **three main evaluation metrics — Accuracy, Balanced Accuracy, and Macro F1 —** are shown in the results tables below.

### 1. Results by Seed

| Seed | Accuracy | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 111 | 0.8961 | 0.8995 | 0.8993 |
| 222 | 0.8938 | 0.8978 | 0.8988 |
| 333 | 0.9030 | 0.9036 | 0.9037 |
| 444 | 0.8813 | 0.8775 | 0.8849 |
| 555 | 0.8950 | 0.8959 | 0.8973 |

### 2. Summary Across Five Seeds

| Metric | Mean | SD | 95% CI |
|---|---:|---:|---:|
| Accuracy | 0.8938 | 0.0079 | [0.8841, 0.9036] |
| Balanced Accuracy | 0.8949 | 0.0101 | [0.8823, 0.9074] |
| Macro F1 | 0.8968 | 0.0071 | [0.8880, 0.9056] |

Across the five seeds, the mean Accuracy was **89.38%**, mean Balanced Accuracy was **89.49%**, and mean Macro F1 was **89.68%**.

An examination of per-class recall across the five seeds did not identify a single class with consistently poor recall, although some variation between seeds was observed.

These results reflect the model's ability to predict the **labels provided by the dataset**. They do not establish that fingerprint images can reliably predict a person's true blood group.
