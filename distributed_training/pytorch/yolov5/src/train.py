from ultralytics import YOLO
import yaml
import torch.distributed as dist
import torch
import os
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
import subprocess
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command, colorstr
from ultralytics.models import yolo


class YoloDdpTrainer(yolo.detect.DetectionTrainer):
    def train(self):
        """"""
        world_size = int(os.environ["WORLD_SIZE"])
        self._do_train(world_size)


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
