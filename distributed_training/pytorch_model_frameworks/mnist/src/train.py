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
Main entry point for training
"""
import argparse
import pytorch_lightning as L
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from typing import Dict
import os
import json

from model import MNISTModel
from data import MNISTDataModule


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
        "--tensorboard",
        type=str,
        help="Location to save logs for tensorboard",
        default=None,
    )
    parser.add_argument(
        "--early_stopping",
        dest="early_stopping",
        action=argparse.BooleanOptionalAction,
        help="Enable early stopping?",
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
    torch.manual_seed(42)
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

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    if args.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.root_dir, every_n_epochs=1, save_last=True, verbose=True
            )
        )

    logger = TensorBoardLogger(save_dir=args.tensorboard) if args.tensorboard else True

    mnist = MNISTDataModule(
        data_dir=args.data_dir, batch_size=(args.batch_size // num_workers)
    )

    trainer = L.Trainer(
        logger=logger,
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

    model.freeze()
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
