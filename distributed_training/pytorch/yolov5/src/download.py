from ultralytics.utils.downloads import download
from pathlib import Path
import yaml
import os

with open("./data.yaml") as f:
    cfg = yaml.safe_load(f.read())

# Download labels
segments = True  # segment or box labels
dir = Path(cfg["path"])  # dataset root dir

if os.path.exists(dir):
    exit(0)

url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
urls = [
    url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")
]  # labels
download(urls, dir=dir.parent)
# Download data
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
    "http://images.cocodataset.org/zips/test2017.zip",
]  # 7G, 41k images (optional)
download(urls, dir=dir / "images", threads=3)
