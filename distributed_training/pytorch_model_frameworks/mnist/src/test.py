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

from model import MNISTModel
from data import MNISTDataModule


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
        "--metrics_json",
        type=str,
        help="output location to store metrics as json",
    )

    parser.add_argument(
        "--batch_size", type=int, help="Effective batch size (across all workers)"
    )

    parser.add_argument(
        "--pytorchjob",
        dest="pytorchjob",
        action=argparse.BooleanOptionalAction,
        help="Is this part of a Kubeflow PyTorchJob?",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    environment = KubeflowEnvironment() if args.pytorchjob else LightningEnvironment()

    if isinstance(environment, KubeflowEnvironment):
        rank_zero_info("Running distributed training with a DDP environment")
    else:
        rank_zero_info("")

    if not ((args.batch_size % environment.world_size()) == 0):
        raise ValueError(
            f"The (effective) batch size {args.batch_size} must be a multiple of the number of workers ({environment.world_size()})"
        )

    # Create Data Module
    mnist = MNISTDataModule(
        data_dir=args.data_dir, batch_size=(args.batch_size // environment.world_size())
    )

    # Load model from checkpoint
    model = MNISTModel.load_from_checkpoint(args.model_ckpt)

    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        # strategy needs to be ddp, find_unused_parameters is due to
        # https://github.com/Lightning-AI/lightning/discussions/6761
        strategy="ddp_find_unused_parameters_false",
        num_nodes=environment.world_size(),
        devices=[d for d in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else -1,
        default_root_dir=args.root_dir,
        enable_progress_bar=False,
        plugins=[environment],
    )

    # test model
    trainer.test(model, mnist)

    # save metrics
    metrics = {k: float(v) for k, v in trainer.callback_metrics}
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f)
