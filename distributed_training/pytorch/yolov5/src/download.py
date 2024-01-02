from ultralytics.utils.downloads import download
from pathlib import Path
import yaml
import os

with open("./data.yaml") as f:
    cfg = yaml.safe_load(f.read())

os.makedirs(cfg["path"], exist_ok=True)
download(
    "https://ultralytics.com/assets/coco128.zip",
    dir=Path(cfg["path"]),
    unzip=True,
    delete=True,
)