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
This module is an example of how to train a MNIST model using pytorch lightning 
with or without DDP.

It logs metrics using a format understood by Katib hyperparameter optimization.
"""
import argparse
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchmetrics import F1Score, Accuracy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.distributed as dst
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Dict, Optional, Any
import os
import json
import time
import logging


class MNISTDataModule(L.LightningDataModule):
    """
    Data Module to load MNIST dataset from disk
    """

    def __init__(self, data_dir: str, batch_size: int):
        """
        Params:
        data_dir   - directory for the dataset
        batch_size - batch size (per worker).
                     https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#distributing-input-data
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data = {}
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # This is where the data would be downloaded if the download happened in the DataModule,
        # lightning makes sure this method is only called by one process.
        # In this example, we assume that the data has already been downloaded onto shared storage
        # and transformed/shuffled by previous pipeline steps.
        # Caution: do not set any object state here, since this is not called for
        # every device, do that in setup()
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data
        pass

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir, download=False, train=True, transform=self.transforms
            )
            self.data["train"] = torch.utils.data.Subset(mnist_full, range(55000))
            self.data["val"] = torch.utils.data.Subset(
                mnist_full, range(55000, 55000 + 5000)
            )
        elif stage == "test":
            self.data["test"] = MNIST(
                self.data_dir, download=False, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(self.data["train"], num_workers=2, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data["test"], num_workers=1, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data["val"], num_workers=0, batch_size=self.batch_size)


class MNISTModel(L.LightningModule):
    """
    MNIST classifier
    """

    def __init__(
        self,
        pDropout1: float = 0.25,
        pDropout2: float = 0.5,
        lr: float = 0.02,
        metric_log_file: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Logging
        self.metric_logger = logging.getLogger("MNISTModel")
        self.metric_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.metric_logger.addHandler(ch)
        if metric_log_file:
            fh = logging.FileHandler(metric_log_file)
            fh.setFormatter(formatter)
            self.metric_logger.addHandler(fh)

        self.lr = lr

        # Model Layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.dropout1 = torch.nn.Dropout(p=pDropout1)
        self.fc1 = torch.nn.Linear(in_features=9216, out_features=128)
        self.dropout2 = torch.nn.Dropout(p=pDropout2)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)

        # Metrics
        self.val_f1 = F1Score(task="multiclass", average="macro", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_f1 = F1Score(task="multiclass", average="macro", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        output = F.log_softmax(self.fc2(x), dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.val_f1.update(preds, y)
        self.log("val_F1", self.val_f1, on_epoch=True, prog_bar=True)
        self.val_acc.update(preds, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.log("test_F1", self.test_f1, on_epoch=True, prog_bar=True)
        self.test_acc.update(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """
        Logs the metrics for the validation set.

        This data is in a format that is understood by the
        Katib metrics collector when the collector is defined as:

        https://www.kubeflow.org/docs/components/katib/experiment/#metrics-collector

        The TEXT format is the default, so that's what is used here, even though
        JSON is a bit easier to read in the log.
        """
        self.log_metric(f"epoch {self.trainer.current_epoch}:")
        self.log_metric(f'acc={self.trainer.callback_metrics["val_acc"]}')
        self.log_metric(f'f1={self.trainer.callback_metrics["val_F1"]}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @rank_zero_only
    def save(self, model_dest: str):
        """Saves the model to the destination file"""
        torch.save(self, model_dest)

    @rank_zero_only
    def log_metric(self, metrics: str):
        """Logs to the metrics"""
        self.metric_logger.info(metrics)


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data Directory")
    parser.add_argument("--root_dir", type=str, help="Root Directory")
    parser.add_argument("--model", type=str, help="Output trained model file")
    parser.add_argument(
        "--evaluation_metrics",
        type=str,
        default=None,
        help="Metrics file from evaluation of validation and test data",
    )
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, help="Effective batch size (across all workers)"
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=0.02)
    parser.add_argument(
        "--metric_log_file",
        type=str,
        help="Output file for TEXT logs, used by Katib Metrics Collectors",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        action=argparse.BooleanOptionalAction,
        help="Enable checkpointing?",
    )
    return parser.parse_args()


@rank_zero_only
def write_test_evaluation_metrics(path: str, metrics: Dict[str, float]):
    """Writes training and test data metrics to the specified path"""
    with open(path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    torch.manual_seed(0)
    args = parse_args()

    num_workers = int(os.environ.get("WORLD_SIZE", 1))
    if not ((args.batch_size % num_workers) == 0):
        raise ValueError(
            f"The (effective) batch size {args.batch_size} must be a multiple of the number of workers ({num_workers})"
        )

    chkpt_path = os.path.join(args.root_dir, "last.ckpt")
    if args.checkpoint and os.path.exists(chkpt_path):
        rank_zero_info(
            f"Initializing training weights/hypterparameters from checkpoint {chkpt_path}"
        )
        model = MNISTModel.load_from_checkpoint(chkpt_path)

    else:
        model = MNISTModel(lr=args.lr, metric_log_file=args.metric_log_file)
        chkpt_path = None

    callbacks = []

    if args.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.root_dir, every_n_epochs=1, save_last=True, verbose=True
            )
        )

    mnist = MNISTDataModule(
        data_dir=args.data_dir, batch_size=(args.batch_size // num_workers)
    )

    trainer = L.Trainer(
        accelerator="auto",
        # strategy needs to be ddp, find_unused_parameters is due to
        # https://github.com/Lightning-AI/lightning/discussions/6761
        strategy="ddp_find_unused_parameters_false",
        num_nodes=num_workers,
        devices=[d for d in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else -1,
        max_epochs=args.max_epochs,
        default_root_dir=args.root_dir,
        enable_progress_bar=False,
        callbacks=callbacks,
        # Note: resume_from_checkpoint is depreciated and will be
        # removed in 2.0, instead use chkpt_path on trainer.fit()
        # https://pytorch-lightning.readthedocs.io/en/1.9.0/common/trainer.html#resume-from-checkpoint
        resume_from_checkpoint=chkpt_path if args.checkpoint else None,
    )

    metrics = {}
    trainer.fit(model, mnist)
    # If we load from checkpoint, and the model was previusly trained
    # for max epochs, it won't train further and there will be no
    # metrics recorded. Just use -1 in that case
    metrics["train_f1"] = (
        float(trainer.callback_metrics["val_F1"])
        if "val_F1" in trainer.callback_metrics
        else -1
    )
    metrics["train_acc"] = (
        float(trainer.callback_metrics["val_acc"])
        if "val_acc" in trainer.callback_metrics
        else -1
    )

    trainer.test(model, mnist)
    metrics["test_f1"] = float(trainer.callback_metrics["test_F1"])
    metrics["test_acc"] = float(trainer.callback_metrics["test_acc"])

    rank_zero_info(f"Training Valiation F1 = {metrics['train_f1']}")
    rank_zero_info(f"Training Valiation accuracy = {metrics['train_acc']}")
    rank_zero_info(f"Test F1 = {metrics['test_f1']}")
    rank_zero_info(f"Test accuracy = {metrics['test_acc']}")

    # Save outputs
    model.save(args.model)
    if args.evaluation_metrics:
        write_test_evaluation_metrics(args.evaluation_metrics, metrics)
