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
Script to build a YOLO  detection model.

This script is expected to be run in each worker of a
DDP training group.
"""
from ultralytics import YOLO
from ultralytics.utils import RANK
import yaml
import os
import json
from trainer import YoloDdpDetectTrainer


# Open the configuration file, all training parameters are set by config
with open("./data.yaml") as f:
    cfg = yaml.safe_load(f)

# Load a model, using existing weights
model = YOLO(cfg.get("model", "yolov8n.pt"))

# Train the model
# Here we use a custom trainer that extends the training class with a patch for
# https://github.com/ultralytics/ultralytics/issues/7282
results = model.train(
    data="./data.yaml",
    cfg="./train.yaml",
    trainer=YoloDdpDetectTrainer if RANK != -1 else None,
)
if RANK in (-1, 0):
    # Save metrics for the model
    # Although the script is called for all workers, this only happens for the first on.
    # TODO: THIS EXAMPLE DOES NOT INCLUDE PROPER TESTING of model predictions
    #   In the real world, we be most interested in metrics from the unseen test data.
    #   In this example, we did not download test data, and don't evaluate against unseen data.
    #   This is bad data science, but good enough to show distributed training and infrastructure.
    r = results.results_dict
    with open("result_metrics.json", "w") as outfile:
        json.dump(r, outfile)
