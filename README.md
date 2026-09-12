# compsci-760-group-12

Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?

# Protocol A: ResNet-18 Training Pipeline

This branch contains the ResNet-18 training framework for **Protocol A: Random Image Split**.

The framework is designed to be reusable for Protocol B so that the main difference between Protocol A and Protocol B is the data split.

---

## Current Pipeline

The current pipeline includes:

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
- Configurable optimizer:
  - Adam
  - SGD
- Optional ImageNet-pretrained ResNet-18
- ImageNet normalization when pretrained weights are used
- CrossEntropyLoss
- No class weights
- No data augmentation
- Evaluation metrics:
  - Accuracy
  - Balanced Accuracy
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

Five predefined split files are available:

```text
splits/protocol_a_seed111.csv
splits/protocol_a_seed222.csv
splits/protocol_a_seed333.csv
splits/protocol_a_seed444.csv
splits/protocol_a_seed555.csv
```

Each split contains approximately:

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

## Run a Single Seed

Example:

```bash
python scripts/run_protocol_a.py \
  --seed 111 \
  --batch-size 32 \
  --epochs 10 \
  --lr 0.0001 \
  --optimizer adam \
  --patience 3 \
  --pretrained
```

The values above are **example settings only** and are not necessarily the final experimental hyperparameters.

---

## Main Arguments

### `--seed`

Selects one of the predefined Protocol A splits.

Example:

```text
111
```

Available seeds:

```text
111
222
333
444
555
```

### `--batch-size`

Number of images processed in one training batch.

Example:

```text
32
```

### `--epochs`

Maximum number of training epochs.

Example:

```text
10
```

Early stopping may stop training before the maximum number of epochs is reached.

### `--lr`

Learning rate used by the optimizer.

Example:

```text
0.0001
```

### `--optimizer`

Selects the optimizer.

Available options:

```text
adam
sgd
```

Example:

```text
--optimizer adam
```

### `--patience`

Number of consecutive epochs without improvement in validation loss before early stopping is triggered.

Example:

```text
3
```

### `--pretrained`

Optional flag.

If included:

```text
Pretrained = True
```

The model uses ImageNet-pretrained ResNet-18 weights and the corresponding ImageNet normalization.

If omitted:

```text
Pretrained = False
```

ResNet-18 is trained without pretrained ImageNet weights.

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
  Defines image preprocessing and optional ImageNet normalization.

- `train.py`  
  Contains the training loop, validation loop, early stopping, and checkpoint saving.

- `evaluate.py`  
  Calculates Accuracy, Balanced Accuracy, Macro F1, Per-class Recall, and the Confusion Matrix.

The Protocol A entry script is:

```text
scripts/run_protocol_a.py
```

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
- class-weight setting
- augmentation setting
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

The trained ResNet-18 model and evaluation functions can be reused for later control and interpretation experiments.

For example:

- Control experiments can reuse the model and evaluation functions.
- Grad-CAM experiments can use the trained ResNet-18 checkpoint.
- Other experiments can reuse the common training and evaluation components where appropriate.

The main reusable components are located in:

```text
src/model.py
src/train.py
src/evaluate.py
```

The dataset still needs to be downloaded separately and placed under:

```text
data/datasets/
```

---

## Current Status

The main Protocol A training framework is implemented and working.

The following settings are still being reviewed before the full experiments:

- final image size / preprocessing
- pretrained setting
- optimizer
- learning rate
- batch size
- maximum epochs
- early-stopping patience

The final settings should be fixed before the formal experiments and then kept identical between Protocol A and Protocol B.

After the final configuration is confirmed, Protocol A will be run using all five predefined seeds and the results will be summarised across runs.