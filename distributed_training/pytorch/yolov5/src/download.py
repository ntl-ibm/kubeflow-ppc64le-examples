from ultralytics.utils.downloads import download
from pathlib import Path
import yaml
import os
import stat

download("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
os.chmod("yolov8n.pt", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
with open("./data.yaml") as f:
    cfg = yaml.safe_load(f.read())

os.makedirs(Path(cfg["path"]).parent, exist_ok=True)
download(
    "https://ultralytics.com/assets/coco128.zip",
    dir=Path(cfg["path"]).parent,
    unzip=True,
    delete=True,
)
