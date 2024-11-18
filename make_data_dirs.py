import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


def copy_files(files, dest):
    for file in files:
        shutil.copy(file, dest)


def create_directory_structure(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(base_dir / subdir, exist_ok=True)


def make_data_dirs():
    SOURCE_DIR = Path("dataset_txt")
    images = sorted(SOURCE_DIR.glob("*.jpg"))
    labels = sorted(SOURCE_DIR.glob("*.txt"))

    BASE_DIR = Path("Data")
    TRAIN_DIR = BASE_DIR / "train"
    TEST_DIR = BASE_DIR / "test"
    VALIDATION_DIR = BASE_DIR / "val"

    create_directory_structure(BASE_DIR, [
        "train/images", "train/labels",
        "test/images", "test/labels",
        "val/images", "val/labels"
    ])

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    copy_files(X_train, TRAIN_DIR / "images")
    copy_files(y_train, TRAIN_DIR / "labels")
    copy_files(X_test, TEST_DIR / "images")
    copy_files(y_test, TEST_DIR / "labels")
    copy_files(X_val, VALIDATION_DIR / "images")
    copy_files(y_val, VALIDATION_DIR / "labels")

    data_yaml = f"""
            train: {TRAIN_DIR / "images"}
            test: {TEST_DIR / "images"}
            val: {VALIDATION_DIR / "images"}
            
            names:
              0: drone
            """
    yaml_path = BASE_DIR / "data.yaml"
    yaml_path.write_text(data_yaml.strip())

if __name__ == "__main__":
    make_data_dirs()
