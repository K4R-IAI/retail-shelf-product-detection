import glob
import os
from pathlib import Path
import shutil
import cv2
from constants import VISUALIZATION_PATH, YOLO_TRAIN_IMAGES_PATH, YOLO_TRAIN_LABELS_PATH

if VISUALIZATION_PATH.exists():
    shutil.rmtree(VISUALIZATION_PATH)

os.makedirs(VISUALIZATION_PATH)

N_VISUALIZATION = 10
images = sorted(YOLO_TRAIN_IMAGES_PATH.glob("*"))[:N_VISUALIZATION]
labels = sorted(YOLO_TRAIN_LABELS_PATH.glob("*"))[:N_VISUALIZATION]

def annotate_image(image, label):
    image = str(image)
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    og_height, og_width = image.shape[:2]
    bboxes = Path(label).read_text().split("\n")
    for bbox in bboxes:
        x, y, width, height = [float(n) for n in bbox.split()[1:]]
        x, width = x * og_width, width * og_width
        y, height = y * og_height, height * og_height
        x, y, width, height = int(x), int(y), int(width), int(height)
        cv2.rectangle(image, (x - width, y - height), (x + width, y + height), color=(255, 0, 0), thickness=5)
    return image

for image, label in zip(images, labels):
    print(image, label)
    annotated_image = annotate_image(image, label)
    cv2.imwrite(str(VISUALIZATION_PATH / str(image).split("/")[-1]), annotated_image)
