# https://kengz.gitbook.io/blog/ml/distributed-training-with-torchelastic-on-kubernetes#pytorch-lightning
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
# https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#torch-distributed-elastic
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#save-a-checkpoint
# https://github.com/Lightning-AI/lightning/discussions/7186#discussioncomment-654431
import argparse
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchmetrics import F1Score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from typing import Optional
import logging
import sys

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

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir, download=True, train=True, transform=self.transforms
            )
            self.data["train"], self.data["val"] = random_split(
                mnist_full, [55000, 5000]
            )
        elif stage == "test":
            self.data["test"] = MNIST(
                self.data_dir, download=True, train=False, transform=transforms
            )

    def train_dataloader(self):
        return DataLoader(self.data["train"], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data["val"], batch_size=self.batch_size)


class MNISTModel(L.LightningModule):
    """
    Minimal MNIST classifier
    """

    def __init__(self):
        super().__init__()

        # Model
        self.fc = torch.nn.Linear(28 * 28, 10)

        # Metrics
        self.val_f1 = F1Score(task="multiclass", num_classes=10)
        self.test_f1 = F1Score(task="multiclass", num_classes=10)

    def forward(self, x):
        flatten = x.view(x.size(0), -1)
        return torch.relu(self.fc(flatten))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.log("val_F1", self.val_f1, on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        log.info(f'val_F1 = {metrics["val_F1"]}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.log("test_F1", self.test_f1, on_epoch=True)

    def on_test_epoch_end(self):
        metrics = self.trainer.callback_metrics
        log.info(f'test_F1 = {metrics["test_F1"]}')

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
        "--batch_size", type=int, default=256 if torch.cuda.is_available() else 64
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
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs,
        default_root_dir=args.root_dir,
        enable_checkpointing=False,
        limit_train_batches=0.25,
    )

    # Train the model
    trainer.fit(model, mnist)

    torch.save(model, args.model)
