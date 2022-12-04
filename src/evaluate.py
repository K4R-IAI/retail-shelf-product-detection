import os, glob
import yaml
import shutil
from yolov5 import val
import cv2
import json

params = yaml.safe_load(open("params.yaml"))["train"]

if os.path.exists("evaluation"):
    shutil.rmtree("evaluation")

val.run(
    imgsz=params["image_size"],
    save_txt=True,
    conf_thres=0.7,
    save_conf=True,
    # save_json=True,
    data=os.path.join("data", "dataset.yaml"),
    weights=os.path.join("training", "weights", "best.pt"),
    project="",
    name="evaluation"
)

for file in glob.glob(os.path.join("evaluation", "labels", "*")):
    with open(file, "r") as open_file:
        new_text = []
        for region in open_file.read().split("\n"):
            region = region.split(" ")
            try:
                new_text.append(
                    " ".join([region[0], region[5], *region[1:5]])
                )
            except IndexError:
                pass
    with open(file, "w") as open_file:
        open_file.write("\n".join(new_text))

from object_detection_metrics.evaluators.coco_evaluator import get_coco_summary
from object_detection_metrics.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from object_detection_metrics.utils.enumerators import BBType
import object_detection_metrics.utils.converter as converter
from globox import AnnotationSet, COCOEvaluator
from pathlib import Path

gts = converter.yolo2bb(
        os.path.join("data", "labels", "val"), 
        os.path.join("data", "images", "val"), 
        "classes_filtered.txt",
        BBType.GROUND_TRUTH
    )

dets = converter.yolo2bb(
        os.path.join("evaluation", "labels"), 
        os.path.join("data", "images", "val"), 
        "classes_filtered.txt",
        BBType.DETECTED
    )

yolo_gts = AnnotationSet.from_yolo(
    folder = Path(os.path.join("data", "labels", "val")), 
    image_folder = Path(os.path.join("data", "images", "val")),
    image_extension = ".JPG"
)

yolo_preds = AnnotationSet.from_yolo(
    folder = Path(os.path.join("evaluation", "labels")), 
    image_folder = Path(os.path.join("data", "images", "val")), 
    image_extension = ".JPG"
)

evaluator = COCOEvaluator(yolo_gts, yolo_preds)
evaluator.show_summary()
evaluator.save_csv(Path(os.path.join("coco.csv")), verbose=True)

results = get_pascalvoc_metrics(gts, dets)
_dict = {}
for metric, res in results.items():
    if metric == 'per_class':
        for c, ap in res.items():
            _dict[c] =  ap["AP"]
    elif metric == "mAP":
        _dict["mAp"] = res
Path("pascal.json").write_text(json.dumps(_dict))