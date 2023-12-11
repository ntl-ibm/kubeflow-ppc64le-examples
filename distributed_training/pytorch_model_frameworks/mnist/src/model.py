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
This module defines a MNIST model.
 
It logs metrics to console or file using a format understood by Katib hyperparameter optimization.
"""
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchmetrics import F1Score, Accuracy, SumMetric
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from typing import Optional
import logging


class MNISTModel(L.LightningModule):
    """
    MNIST classifier
    """

    def __init__(
        self,
        pDropout1: float = 0.25,
        pDropout2: float = 0.5,
        lr: float = 0.02,
        katib_log_file: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Logging
        self.katib_logger = logging.getLogger("MNISTModel_katib")
        self.katib_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.katib_logger.addHandler(ch)
        if katib_log_file:
            fh = logging.FileHandler(katib_log_file)
            fh.setFormatter(formatter)
            self.katib_logger.addHandler(fh)

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
        self.val_loss = SumMetric()
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
        print(
            f"train: loss = {loss} batch shape = {batch.shape} batch_idx = {batch_idx}"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # del batch_idx

        x, y = batch
        logits = self(x)

        # https://lightning.ai/docs/pytorch/stable/integrations/ipu/prepare.html#synchronize-validation-and-test-logging
        # These are all TorchMetrics so there is no concern about syncing
        loss = F.cross_entropy(logits, y)
        print(f"val: loss = {loss} batch shape = {batch.shape} batch_idx = {batch_idx}")
        self.val_loss.update(loss)
        self.log("val_loss", self.val_loss, on_epoch=True)

        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.log("val_F1", self.val_f1, on_epoch=True)
        self.val_acc.update(preds, y)
        self.log("val_acc", self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        del batch_idx

        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.log("test_F1", self.test_f1, on_epoch=True)
        self.test_acc.update(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def on_train_epoch_end(self):
        """
        Logs the metrics for the validation set.

        This data is in a format that is understood by the
        Katib metrics collector when the collector is defined as:

        https://www.kubeflow.org/docs/components/katib/experiment/#metrics-collector

        The TEXT format is the default, so that's what is used here, even though
        JSON is a bit easier to read in the log.
        """
        self.log_metric_for_katib(f"epoch {self.trainer.current_epoch}:")
        self.log_metric_for_katib(f'acc={self.trainer.callback_metrics["val_acc"]}')
        self.log_metric_for_katib(f'f1={self.trainer.callback_metrics["val_F1"]}')
        self.log_metric_for_katib(f'loss={self.trainer.callback_metrics["val_loss"]}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @rank_zero_only
    def log_metric_for_katib(self, metrics: str):
        """Logs to the metrics"""
        self.katib_logger.info(metrics)
