# Copyright 2023 IBM All Rights Reserved.
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
Model Test script

Metrics are written to the specified json file
"""
import argparse
import json

import pytorch_lightning as L
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
)
from torchvision.datasets import MNIST

from model import MNISTModel
from data import transform_image

# MNISTDataModule


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data Directory")
    parser.add_argument("--root_dir", type=str, help="Root Directory")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Location for the checkpoint of the trained model",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="output location to store the output onnx model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mnist_full = MNIST(
        args.data_dir, download=False, train=True, transform=transform_image
    )

    # MNISTDataModule(data_dir=args.data_dir, batch_size=50).train_dataloader()
    model = MNISTModel.load_from_checkpoint(args.model_ckpt)

    x, y = mnist_full[0]
    model.to_onnx(args.onnx, x)
