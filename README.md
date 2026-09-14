# compsci-760-group-12

Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?

# Protocol A: ResNet-18 Training Pipeline

This branch contains the ResNet-18 training framework and experimental results for **Protocol A: Random Image Split**.

The framework is designed to be reusable for Protocol B so that the main difference between Protocol A and Protocol B is the data split.

---

## Protocol A Setup

Protocol A uses:

- 8-class fingerprint blood group classification
- ResNet-18
- Stratified 70/15/15 train-validation-test split
- Five predefined random seeds:
  - 111
  - 222
  - 333
  - 444
  - 555
- Early stopping based on validation loss
- Best model checkpoint saving
- ImageNet-pretrained ResNet-18
- ImageNet normalization
- CrossEntropyLoss

Evaluation metrics include:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class Recall
- Confusion Matrix

---

## Dataset Setup

The fingerprint dataset is not stored in GitHub.

Download the **Fingerprint Blood Group Classification Dataset** and place the eight blood-group folders under:

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

The expected total number of images is:

```text
5837
```

The dataset structure and image counts can be checked using:

```bash
python scripts/check_dataset.py
```

---

## Protocol A Splits

Protocol A uses a stratified random image split:

- 70% training
- 15% validation
- 15% testing

Five predefined split files are used:

```text
splits/protocol_a_seed111.csv
splits/protocol_a_seed222.csv
splits/protocol_a_seed333.csv
splits/protocol_a_seed444.csv
splits/protocol_a_seed555.csv
```

Each split contains:

```text
Training:   4085 images
Validation: 876 images
Testing:    876 images
```

The split files can be regenerated using:

```bash
python scripts/create_random_split.py
```

---

## Final Training Configuration

The learning rate and batch size were selected based on validation performance. The remaining training settings were fixed in advance.

The final Protocol A configuration is:

```text
Model:          ResNet-18
Input size:     224 × 224
Pretrained:     True
Optimizer:      Adam
Learning rate:  0.0001
Batch size:     32
Max epochs:     10
Patience:       3
Loss:           CrossEntropyLoss
```

No class weights or data augmentation were used in the Protocol A experiments.

The same training configuration should be used for Protocol B.

---

## Run Protocol A

Example for seed 111:

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

The seed can be changed to:

```text
111
222
333
444
555
```

---

## Protocol A Results

Protocol A was evaluated using five predefined random seeds with the same final training configuration.

For this multiclass classification task, Balanced Accuracy is equivalent to Macro Recall, so only Balanced Accuracy is shown in the summary tables to avoid duplication.

### Results by Seed

| Seed | Accuracy | Balanced Accuracy | Macro Precision | Macro F1 |
|---:|---:|---:|---:|---:|
| 111 | 0.8961 | 0.8995 | 0.9025 | 0.8993 |
| 222 | 0.8938 | 0.8978 | 0.9040 | 0.8988 |
| 333 | 0.9030 | 0.9036 | 0.9072 | 0.9037 |
| 444 | 0.8813 | 0.8775 | 0.8975 | 0.8849 |
| 555 | 0.8950 | 0.8959 | 0.9016 | 0.8973 |

### Summary Across Five Seeds

| Metric | Mean | SD | 95% CI |
|---|---:|---:|---:|
| Accuracy | 0.8938 | 0.0079 | [0.8841, 0.9036] |
| Balanced Accuracy | 0.8949 | 0.0101 | [0.8823, 0.9074] |
| Macro Precision | 0.9026 | 0.0035 | [0.8982, 0.9070] |
| Macro F1 | 0.8968 | 0.0071 | [0.8880, 0.9056] |

Across the five runs, no class showed consistently poor recall, although some class-wise variation was observed across seeds.

---

## Results Files

The experiment outputs are stored under:

```text
results/

├── hyperparameter_search.csv
├── protocol_a_seed111.txt
├── protocol_a_seed222.txt
├── protocol_a_seed333.txt
├── protocol_a_seed444.txt
├── protocol_a_seed555.txt
└── protocol_a_summary.csv
```

`hyperparameter_search.csv` records the validation results used to select the final learning rate and batch size.

Each `protocol_a_seedXXX.txt` file contains the complete output for one Protocol A run, including the training process, validation results, test metrics, per-class recall, and confusion matrix.

`protocol_a_summary.csv` contains the five-seed summary including mean, standard deviation, and 95% confidence intervals.

---

## Code Structure

The reusable model and training components are located in:

```text
src/

├── dataset.py
├── model.py
├── transforms.py
├── train.py
└── evaluate.py
```

Their roles are:

- `dataset.py`  
  Loads fingerprint images and class labels from a split CSV.

- `model.py`  
  Builds the ResNet-18 model for eight-class classification.

- `transforms.py`  
  Defines image preprocessing and ImageNet normalization.

- `train.py`  
  Contains the training loop, validation loop, early stopping, and checkpoint saving.

- `evaluate.py`  
  Calculates Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, Per-class Recall, and the Confusion Matrix.

The main Protocol A scripts are:

```text
scripts/run_protocol_a.py
scripts/summarize_protocol_a.py
```

`run_protocol_a.py` trains and evaluates one Protocol A seed.

`summarize_protocol_a.py` reads the saved results from all five seeds and calculates the mean, standard deviation, and 95% confidence interval for the main evaluation metrics.

---

## Reuse by Protocol B

Protocol B should reuse the same common training framework:

```text
src/dataset.py
src/model.py
src/transforms.py
src/train.py
src/evaluate.py
```

Protocol B should use the same:

- ResNet-18 architecture
- image preprocessing
- pretrained setting
- optimizer
- learning rate
- batch size
- maximum epochs
- early stopping rule
- loss function
- evaluation metrics

The main change should be the data split:

```text
Protocol A
Random image split

Protocol B
Leakage-controlled group split
```

This keeps the comparison between Protocol A and Protocol B as fair as possible.

---

## Reuse for Control Experiments

The trained ResNet-18 model and evaluation functions can also be reused for later control and interpretation experiments.

For example:

- Control experiments can reuse the model and evaluation functions.
- Grad-CAM experiments can use the trained ResNet-18 checkpoint.
- Other experiments can reuse the common training and evaluation components where appropriate.

The main reusable components are:

```text
src/model.py
src/train.py
src/evaluate.py
```

The dataset still needs to be downloaded separately and placed under:

```text
data/datasets/
```