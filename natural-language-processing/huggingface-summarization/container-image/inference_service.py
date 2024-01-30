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
Implements and InferenceService prdictor for the credit-risk demo.
Inferencing is performed using an ONNX model and onnx-runtime.

Models are evaluated using CPU only.

Author: ntl@us.ibm.com
"""
import argparse
import logging
import os
from typing import Dict, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import kserve
import torch
from ray import serve

NUM_GPUS = torch.cuda.device_count()
NUM_REPLICAS = 1


@serve.deployment(
    name="billsum",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": 1, "num_gpus": NUM_GPUS / NUM_REPLICAS},
)
class BillSummarizer(kserve.Model):
    MODEL_PATH = "/mnt/models/pt"

    def __init__(self, name: str):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "billsum"
        self.model_path = f"/mnt/models/{self.name}"
        self.load()
        self.ready = True

    def load(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
    ) -> Dict:
        if "instances" not in payload or len(payload["instances"]):
            raise ValueError("Instances must be a list of a single instance")

        text = payload["instances"][0]
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            inputs, max_new_tokens=headers.get("max_new_tokens", 128)
        )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

    # This arg is automatically provided by the inferenceservice
    # it contains the name of the inference service resource.
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", required=True
    )
    args, _ = parser.parse_known_args()

    kserve.ModelServer().start({"billsum": BillSummarizer})
