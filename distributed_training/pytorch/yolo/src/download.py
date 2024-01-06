# Copyright 2024 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python script that downloads assets from ultralytics
"""

from ultralytics.utils.downloads import download
from pathlib import Path
import yaml
import os
import stat

# Download Pretrained weights
download("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
os.chmod("yolov8n.pt", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

# Open configuration
with open("./data.yaml") as f:
    cfg = yaml.safe_load(f.read())

# Download training and validation data
os.makedirs(Path(cfg["path"]).parent, exist_ok=True)
download(
    "https://ultralytics.com/assets/coco128.zip",
    dir=Path(cfg["path"]).parent,
    unzip=True,
    delete=True,
)
