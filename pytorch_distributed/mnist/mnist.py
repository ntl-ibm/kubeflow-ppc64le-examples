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
with DDP.
"""
import argparse
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchmetrics import F1Score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.distributed as dst
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only

from typing import Optional
import logging
import sys
import os
import json


class MNISTDataModule(L.LightningDataModule):
    """
    Data Module to load MNIST dataset from disk
    """

    def __init__(self, data_dir: str, batch_size: int):
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

    def __init__(self):
        super().__init__()

        # Model Layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.fc1 = torch.nn.Linear(in_features=9216, out_features=128)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)

        # Metrics
        self.val_f1 = F1Score(task="multiclass", num_classes=10)
        self.test_f1 = F1Score(task="multiclass", num_classes=10)

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.log("test_F1", self.test_f1, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """
        Prints the F1 score for the validation set after each epoch.
        """
        rank_zero_info(
            f">>>> Finished training epoch {self.trainer.current_epoch} / {self.trainer.max_epochs} "
            + f"val_F1 = {self.trainer.callback_metrics['val_F1']}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @rank_zero_only
    def save(self, model_dest: str):
        """Saves the model to the destination file"""
        torch.save(self, model_dest)


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data Directory")
    parser.add_argument("--root_dir", type=str, help="Root Directory")
    parser.add_argument("--model", type=str, help="Output trained model file")
    parser.add_argument(
        "--kubeflow_ui_metadata", type=str, default=None, help="Ouput metrics json"
    )
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, default=64 if torch.cuda.is_available() else 16
    )
    return parser.parse_args()


@rank_zero_only
def create_kubeflow_ui_metadata(path: str, metadata: Dict[str, str]):
    """Writes training and test data metrics to the specified path

    The file is formatted such that Kubeflow can display it as a visualization
    of a component.
    """
    headings = list(metadata.keys())

    metadata = {
        "outputs": [
            {
                "type": "table",
                "storage": "inline",
                "format": "csv",
                "header": headings,
                "source": ",".join([str(metadata[heading]) for heading in headings]),
            }
        ]
    }

    with open(path, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    torch.manual_seed(0)

    args = parse_args()
    model = MNISTModel()
    mnist = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    # Initialize a trainer
    trainer = L.Trainer(
        accelerator="auto",
        strategy="ddp",
        num_nodes=os.environ["WORLD_SIZE"],
        devices=[d for d in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else -1,
        max_epochs=args.max_epochs,
        default_root_dir=args.root_dir,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    metrics = {}
    # Train the model
    trainer.fit(model, mnist)
    metrics["train_f1"] = trainer.callback_metrics["val_F1"]

    # Test the model
    trainer.test(model, mnist)
    metrics["test_f1"] = trainer.callback_metrics["test_F1"]

    rank_zero_info(f"Training Valiation F1 = {val_f1}")
    rank_zero_info(f"Test F1 = {test_f1}")

    # Save outputs (Model and metrics)
    model.save(args.model)
    if args.kubeflow_ui_metadata:
        create_kubeflow_ui_metadata(args.kubeflow_ui_metadata, metrics)
