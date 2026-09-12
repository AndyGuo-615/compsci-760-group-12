from pathlib import Path

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"

EXPECTED_CLASSES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

print("Dataset path:", DATASET_DIR)
print("Dataset exists:", DATASET_DIR.exists())
print()

total_images = 0

for class_name in EXPECTED_CLASSES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"{class_name}: folder not found")
        continue

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".bmp"
    ]

    image_count = len(image_files)
    total_images += image_count

    print(f"{class_name}: {image_count} images")

print()
print(f"Total images: {total_images}")