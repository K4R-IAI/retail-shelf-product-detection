import shutil
import os
from pathlib import Path
import csv, yaml
from collections import defaultdict
from tqdm import tqdm
import cv2

from constants import IMAGES_PATH, OUTPUT_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH, YOLO_TEST_IMAGES_PATH, YOLO_TEST_LABELS_PATH, YOLO_TRAIN_IMAGES_PATH, YOLO_TRAIN_LABELS_PATH, YOLO_VAL_IMAGES_PATH, YOLO_VAL_LABELS_PATH 

params = yaml.safe_load(open("params.yaml"))["train"]
#

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

for path in [
        YOLO_TRAIN_IMAGES_PATH, YOLO_TRAIN_LABELS_PATH, 
        YOLO_TEST_IMAGES_PATH, YOLO_TEST_LABELS_PATH, 
        YOLO_VAL_IMAGES_PATH, YOLO_VAL_LABELS_PATH
    ]:
    os.makedirs(path)

def resize_image(image):
    height, width = image.shape[:2]
    scale_factor = height / params["image_size"]
    return cv2.resize(image, (int(width / scale_factor), 1280))

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
for train_csv_row in tqdm(csv.reader(open(TRAIN_CSV_PATH))):
    image, yolo_str = convert_row(train_csv_row)
    train_dict[image].append(yolo_str)

test_dict = defaultdict(list)
for test_csv_row in tqdm(csv.reader(open(TEST_CSV_PATH))):
    image, yolo_str = convert_row(test_csv_row)
    test_dict[image].append(yolo_str)

val_dict = defaultdict(list)
for val_csv_row in tqdm(csv.reader(open(VAL_CSV_PATH))):
    image, yolo_str = convert_row(val_csv_row)
    val_dict[image].append(yolo_str)

for key in train_dict.keys():
    (YOLO_TRAIN_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(train_dict[key]))
    image = cv2.imread(str(IMAGES_PATH / key), cv2.IMREAD_UNCHANGED)
    image = resize_image(image)
    cv2.imwrite(str(YOLO_TRAIN_IMAGES_PATH / key), image)

for key in test_dict.keys():
    (YOLO_TEST_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(test_dict[key]))
    cv2.imread(str(IMAGES_PATH / key), cv2.IMREAD_UNCHANGED)
    image = resize_image(image)
    cv2.imwrite(str(YOLO_TEST_IMAGES_PATH / key), image)

for key in val_dict.keys():
    (YOLO_VAL_LABELS_PATH / key.replace("jpg", "txt")).write_text("\n".join(val_dict[key]))
    cv2.imread(str(IMAGES_PATH / key), cv2.IMREAD_UNCHANGED)
    image = resize_image(image)
    cv2.imwrite(str(YOLO_VAL_IMAGES_PATH / key), image)


shutil.copy2("dataset.yaml", OUTPUT_PATH / "dataset.yaml")