from ultralytics import YOLO
import yaml
import torch.distributed as dist
import torch
import os
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
import subprocess
from ultralytics.models import yolo
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_yolo_dataset


class YoloDdpTrainer(yolo.detect.DetectionTrainer):
    def train(self):
        """"""
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"{world_size}, {RANK}")
        self._do_train(world_size)

    def get_dataloader(
        self, dataset_path, batch_size=16, rank=0, mode="train", local_rank=-1
    ):
        """Construct and return dataloader."""
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(
            local_rank
            if local_rank > 0
            else (int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0)
        ):  # init dataset *.cache only once if DDP
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
    device=[d for d in range(torch.cuda.device_count())],
    trainer=YoloDdpTrainer,
)

print(type(results))
print(str(results))
