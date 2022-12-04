import csv
import os
from pathlib import Path


IMAGES_PATH = Path(os.path.join("input", "images"))
TRAIN_CSV_PATH = Path(os.path.join("input", "annotations", "annotations_train.csv"))
TEST_CSV_PATH = Path(os.path.join("input", "annotations", "annotations_test.csv"))
VAL_CSV_PATH = Path(os.path.join("input", "annotations", "annotations_val.csv"))
CLASSES = ["object"]

OUTPUT_PATH = Path("data")
YOLO_IMAGES_PATH = OUTPUT_PATH / "images"
YOLO_TRAIN_IMAGES_PATH = YOLO_IMAGES_PATH / "train"
YOLO_TEST_IMAGES_PATH = YOLO_IMAGES_PATH / "test"
YOLO_VAL_IMAGES_PATH = YOLO_IMAGES_PATH / "val"

YOLO_LABELS_PATH = OUTPUT_PATH / "labels"
YOLO_TRAIN_LABELS_PATH = YOLO_LABELS_PATH / "train"
YOLO_TEST_LABELS_PATH = YOLO_LABELS_PATH / "test"
YOLO_VAL_LABELS_PATH = YOLO_LABELS_PATH / "val"

VISUALIZATION_PATH = Path("visualization")