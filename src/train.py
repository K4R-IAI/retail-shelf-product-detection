import os
import shutil
import yaml
from yolov5 import train

if os.path.exists("training"):
    shutil.rmtree("training")

params = yaml.safe_load(open("params.yaml"))["train"]

train.run(
    imgsz=params["image_size"],
    epochs=params["epochs"], 
    weights=params["pretrained_weights"],
    data=os.path.join("data", "dataset.yaml"), 
    project="", 
    name="training"
)