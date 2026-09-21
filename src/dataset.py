from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Fixed class order for all experiments
CLASS_NAMES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


class FingerprintDataset(Dataset):
    """
    Dataset for fingerprint blood-group classification.

    The split CSV must contain:
        filepath, label, split
    """

    def __init__(
        self,
        split_csv,
        split,
        transform=None,
        project_root=None
    ):
        # Project root
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]

        self.project_root = Path(project_root)
        self.transform = transform

        # Read split CSV
        split_csv = Path(split_csv)

        if not split_csv.is_absolute():
            split_csv = self.project_root / split_csv

        df = pd.read_csv(split_csv)

        # Check required columns
        required_columns = {"filepath", "label", "split"}

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Split CSV must contain columns: {required_columns}"
            )

        # Check requested split
        valid_splits = {"train", "validation", "test"}

        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}"
            )

        # Keep only the requested subset
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No images found for split: {split}"
            )

        # Check labels
        unknown_labels = set(self.df["label"]) - set(CLASS_NAMES)

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found: {unknown_labels}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self.project_root / row["filepath"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Convert to RGB because ResNet-18 expects 3 channels
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]

        return image, label