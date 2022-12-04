import shutil
import os
from pathlib import Path
import csv
from collections import defaultdict
from tqdm import tqdm 

# train_0.jpg,208,537,422,814,object,3024,3024
# test_0.jpg,120,2527,225,2764,object,2448,3264
# val_0.jpg,5,1429,219,1612,object,2336,4160

IMAGES_PATH = Path(os.path.join("input", "images"))
TRAIN_CSV_READER = csv.reader(open(Path(os.path.join("input", "annotations", "annotations_train.csv"))))
TEST_CSV_READER = csv.reader(open(Path(os.path.join("input", "annotations", "annotations_test.csv"))))
VAL_CSV_READER = csv.reader(open(Path(os.path.join("input", "annotations", "annotations_val.csv"))))
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

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

for path in [
        YOLO_TRAIN_IMAGES_PATH, YOLO_TRAIN_LABELS_PATH, 
        YOLO_TEST_IMAGES_PATH, YOLO_TEST_LABELS_PATH, 
        YOLO_VAL_IMAGES_PATH, YOLO_VAL_LABELS_PATH
    ]:
    os.makedirs(path)

def convert_row(row):
    image, xmin, ymin, xmax, ymax, _, width, height = row
    x = (int(xmin) + int(xmax)) / 2
    y = (int(ymin) + int(ymax)) / 2
    _width = (x - int(xmin))
    _height = (y - int(ymin)) 
    x_rel = x / int(width)
    y_rel = y / int(height)
    width_rel = _width / int(width)
    height_rel = _height / int(height)
    return image, f"0 {x_rel} {y_rel} {width_rel} {height_rel}"

train_dict = defaultdict(list)
for train_csv_row in tqdm(TRAIN_CSV_READER):
    image, yolo_str = convert_row(train_csv_row)
    train_dict[image].append(yolo_str)

test_dict = defaultdict(list)
for test_csv_row in tqdm(TEST_CSV_READER):
    image, yolo_str = convert_row(test_csv_row)
    test_dict[image].append(yolo_str)

val_dict = defaultdict(list)
for val_csv_row in tqdm(VAL_CSV_READER):
    image, yolo_str = convert_row(val_csv_row)
    val_dict[image].append(yolo_str)

for key in train_dict.keys():
    (YOLO_TRAIN_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(train_dict[key]))
    shutil.copy2(IMAGES_PATH / key, YOLO_TRAIN_IMAGES_PATH / key)

for key in test_dict.keys():
    (YOLO_TEST_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(test_dict[key]))
    shutil.copy2(IMAGES_PATH / key, YOLO_TEST_IMAGES_PATH / key)

for key in val_dict.keys():
    (YOLO_VAL_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(val_dict[key]))
    shutil.copy2(IMAGES_PATH / key, YOLO_VAL_IMAGES_PATH / key)

shutil.copy2("dataset.yaml", OUTPUT_PATH / "dataset.yaml")