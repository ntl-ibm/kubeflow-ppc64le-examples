# https://kengz.gitbook.io/blog/ml/distributed-training-with-torchelastic-on-kubernetes#pytorch-lightning
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
# https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#torch-distributed-elastic
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-a-checkpoint
# https://github.com/Lightning-AI/lightning/discussions/7186#discussioncomment-654431
# https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#optimize-multi-machine-communication
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/265#issuecomment-1085185496
import argparse
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchmetrics import F1Score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torch.distributed as dst

from pytorch_lightning.utilities import rank_zero_info
from typing import Optional
import logging
import sys
import os

log = logging.getLogger()
log.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

ROOT_DIR = "/home/jovyan/kubeflow-ppc64le-examples/pytorch_distributed/working_root"


class MNISTDataModule(L.LightningDataModule):
    """
    Data Module to load MNIST dataset

    Splits into train/validate/test sets
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
        # This is where the data should be downloaded, lightning makes sure it only
        # happens in one process.
        # In this example, we assume that the data has already been downloaded and
        # transformed by other pipeline steps.
        # Do not set any state here, since this is not called for
        # every device, do that in setup()
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data
        pass

    def setup(self, stage: str):
        if stage == "fit":
            #  Assuming that the data has already been shuffled
            #  We want each worker seeing the same train / val split
            #  so no need for random split
            mnist_full = MNIST(
                self.data_dir, download=False, train=True, transform=self.transforms
            )
            self.data["train"] = torch.utils.data.Subset(mnist_full, range(55000))
            self.data["val"] = torch.utils.data.Subset(
                mnist_full, range(55000, 55000 + 5000)
            )
            # , self.data["val"] = random_split(
            #    mnist_full, [55000, 5000]
            # )
        elif stage == "test":
            self.data["test"] = MNIST(
                self.data_dir, download=False, train=False, transform=transforms
            )

    def train_dataloader(self):
        return DataLoader(self.data["train"], num_workers=2, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data["test"], num_workers=1, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data["val"], num_workers=0, batch_size=self.batch_size)


class MNISTModel(L.LightningModule):
    """
    Minimal MNIST classifier
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
        if ("RANK" not in os.environ) or (os.environ["RANK"] == "0"):
            print(
                f"Finished epoch {self.trainer.current_epoch} / {self.trainer.max_epochs} val_F1 = {self.trainer.callback_metrics['val_F1']}"
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=f"{ROOT_DIR}/data", help="Data Directory"
    )
    parser.add_argument(
        "--root_dir", type=str, default=f"{ROOT_DIR}/root", help="Root Directory"
    )
    parser.add_argument("--model", type=str, default=f"{ROOT_DIR}/model.pt")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, default=64 if torch.cuda.is_available() else 16
    )
    return parser.parse_args()


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

    # Train the model
    trainer.fit(model, mnist)

    if os.environ["RANK"] == "0":
        torch.save(model, args.model)
