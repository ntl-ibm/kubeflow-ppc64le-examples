import argparse

from torchvision import models
from typing import Dict, Union
import torch
import numpy as np
from kserve import Model, ModelServer
import base64
from ultralytics import YOLO
from PIL import Image
import io
from werkzeug import exceptions
import os


def b64ToImage(img_data: str) -> Image:
    raw_img_data = base64.b64decode(img_data)
    return Image.open(io.BytesIO(raw_img_data))


IOU = float(os.environ.get("IOU", "0.7"))
CONF = float(os.environ.get("CONF", "0.25"))


class YoloModelPredictor(Model):
    MODEL_PATH = "/mnt/models/model.pt"

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        self.model = YOLO(self.MODEL_PATH, task="detect")
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        if len(payload["instances"]) > 1:
            raise exceptions.BadRequest("Payload must have exactly one instance")

        input_image = b64ToImage(payload["instances"][0]["image"]["b64"])

        # https://docs.ultralytics.com/reference/engine/results/
        result = self.model(
            input_image,
            iou=IOU,
            conf=CONF,
        )
        result_image = result.plot(
            conf=True, pil=True, boxes=True, labels=True, probs=True
        )

        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        return {"predictions": [base64.b64encode(buffered.getvalue())]}


if __name__ == "__main__":
    model = YoloModelPredictor("custom-model")
    ModelServer().start([model])
