from ultralytics.utils.downloads import download
from pathlib import Path
import yaml
import os

download("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

with open("./data.yaml") as f:
    cfg = yaml.safe_load(f.read())

os.makedirs(cfg["path"].parent, exist_ok=True)
download(
    "https://ultralytics.com/assets/coco128.zip",
    dir=Path(cfg["path"]).parent,
    unzip=True,
    delete=True,
)
