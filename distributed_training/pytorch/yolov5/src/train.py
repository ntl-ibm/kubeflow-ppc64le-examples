from ultralytics import YOLO
import yaml
import torch.distributed as dist
import torch
import os
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, TQDM, callbacks
import subprocess
from ultralytics.models import yolo
import time
import numpy as np
import math
import warnings

# from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_yolo_dataset
from datetime import datetime, timedelta
from contextlib import contextmanager
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz
from torch import nn

LOCAL_RANK = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1


class YoloDdpTrainer(yolo.detect.DetectionTrainer):
    def train(self):
        """"""
        world_size = int(os.environ["WORLD_SIZE"])

        if self.args.rect:
            LOGGER.warning(
                "WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'"
            )
            self.args.rect = False
        if self.args.batch == -1:
            LOGGER.warning(
                "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                "default 'batch=16'"
            )
            self.args.batch = 16

        self._do_train(world_size)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in ["train", "val"]

        # init dataset *.cache only once if DDP
        # Assumes dataset and cache are shared by all
        if RANK == 0:
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        dist.barrier()
        # Rank 0 built, OK to build others
        if RANK != 0:
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning(
                "WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(
            dataset, batch_size, workers, shuffle, rank
        )  # return dataloader

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(LOCAL_RANK)
        self.device = torch.device("cuda", LOCAL_RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = (
                callbacks.default_callbacks.copy()
            )  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(
                self.amp, src=0
            )  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[LOCAL_RANK]
            )

        # Check imgsz
        gs = max(
            int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32
        )  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if (
            self.batch_size == -1 and RANK == -1
        ):  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                self.model, self.args.imgsz, self.amp
            )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=RANK, mode="train"
        )
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(
                prefix="val"
            )
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(
            round(self.args.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        weight_decay = (
            self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        )  # scale weight_decay
        iterations = (
            math.ceil(
                len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)
            )
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = (
            max(round(self.args.warmup_epochs * nb), 100)
            if self.args.warmup_epochs > 0
            else -1
        )  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for "
            f"{self.args.time} hours..."
            if self.args.time
            else f"{self.epochs} epochs..."
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1,
                        int(
                            np.interp(
                                ni, xi, [1, self.args.nbs / self.batch_size]
                            ).round()
                        ),
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None
                        else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (
                            self.args.time * 3600
                        )
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(
                                broadcast_list, 0
                            )  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            mem,
                            *losses,
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )

                # Validation
                if (
                    self.args.val
                    or final_epoch
                    or self.stopper.possible_stop
                    or self.stop
                ):
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop |= self.stopper(epoch + 1, self.fitness)
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (
                        self.args.time * 3600
                    )

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (
                        epoch - self.start_epoch + 1
                    )
                    self.epochs = self.args.epochs = math.ceil(
                        self.args.time * 3600 / mean_epoch_time
                    )
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")


# dist.init_process_group(
#    backend="nccl",
#    world_size=int(os.environ["WORLD_SIZE"]),
#    rank=int(os.environ["RANK"]),
# )

with open("./data.yaml") as f:
    cfg = yaml.safe_load(f)

# Load a model
model = YOLO(cfg.get("model", "yolov8n.pt"))

# Train the model
results = model.train(
    data="./data.yaml",
    cfg="./train.yaml",
    trainer=YoloDdpTrainer,
)

print(type(results))
print(str(results))
