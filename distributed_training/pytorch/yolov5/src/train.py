from ultralytics import YOLO
import yaml
import torch.distributed as dist
import torch
import os
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
import subprocess
from ultralytics.models import yolo

# from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_yolo_dataset
from datetime import datetime, timedelta
from contextlib import contextmanager

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

    def get_dataloader(
        self, dataset_path, batch_size=16, rank=0, mode="train", local_rank=-1
    ):
        """Construct and return dataloader."""
        assert mode in ["train", "val"]

        # init dataset *.cache only once if DDP
        # assumes multi-node will share the same dataset
        if RANK == 0:
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        dist.barrier()

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
