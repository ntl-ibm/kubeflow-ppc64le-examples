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
Data Loader and preprocessing transformations for the MNIST data set
"""

import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

transform_image = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


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
        self.transforms = transform_image

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
