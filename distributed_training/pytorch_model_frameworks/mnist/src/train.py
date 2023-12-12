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
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
)
from typing import Dict
import os
import shutil
from pathlib import Path

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
        help="Output location for the checkpoint of the trained model",
    )
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, help="Effective batch size (across all workers)"
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=0.02)
    parser.add_argument(
        "--katib_log_file",
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
    parser.add_argument(
        "--pytorchjob",
        dest="pytorchjob",
        action=argparse.BooleanOptionalAction,
        help="Is this part of a Kubeflow PyTorchJob?",
    )

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
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

    # Create model, resuming from checkpoint if last.ckpt exists and checkpointing asked for
    possible_prior_chkpt_path = os.path.join(args.root_dir, "last.ckpt")
    if args.checkpoint and os.path.exists(possible_prior_chkpt_path):
        prior_chkpt_path = possible_prior_chkpt_path
        rank_zero_info(
            f"Initializing training weights/hypterparameters from checkpoint {prior_chkpt_path}"
        )
        model = MNISTModel.load_from_checkpoint(prior_chkpt_path)
    else:
        model = MNISTModel(lr=args.lr, katib_log_file=args.katib_log_file)
        prior_chkpt_path = None

    # Setup callbacks
    callbacks = []

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", verbose=True))

    checkpoint_cb = None
    if args.checkpoint:
        checkpoint_cb = ModelCheckpoint(
            dirpath=args.root_dir,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=True,
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_cb)

    # Logger for TensorBoard
    logger = (
        TensorBoardLogger(save_dir=args.tensorboard, name="")
        if args.tensorboard
        else True
    )

    print(logger.log_dir)

    # Trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        # strategy needs to be ddp, find_unused_parameters is due to
        # https://github.com/Lightning-AI/lightning/discussions/6761
        strategy="ddp_find_unused_parameters_false",
        num_nodes=environment.world_size(),
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
        # resume_from_checkpoint=prior_chkpt_path,
        plugins=[environment],
    )

    # Fit model
    trainer.fit(model, mnist, ckpt_path=prior_chkpt_path)
    trainer.strategy.barrier("Trainer.fit() is complete")

    # If requested, Save Checkpoint for the model at the specified location
    if args.model_ckpt and (environment.global_rank() == 0):
        rank_zero_info(f"Copy best checkpoint to {args.model_ckpt}")
        p_dirs = Path(os.path.dirname(args.model_ckpt))
        p_dirs.mkdir(parents=True, exist_ok=True)

        if checkpoint_cb:
            shutil.copyfile(checkpoint_cb.best_model_path, args.model_ckpt)
        else:
            trainer.save_checkpoint(args.model_ckpt)
