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

Author: ntl@us.ibm.com
"""
import argparse
import os
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import kserve
import torch
from ray import serve
from werkzeug.exceptions import BadRequest


class KServeModelForSeq2SeqLM(kserve.Model):

    def __init__(self, name, version):
        self.name = name
        self.version = version
        super().__init__(name=self.name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load()
        self.ready = True

    def load(self):
        model_path = f"/mnt/models/{self.version}/{self.name}"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
    ) -> Dict:
        if (
            "instances" not in payload
            or not isinstance(payload["instances"], list)
            or (len(payload["instances"]) != 1)
        ):
            raise BadRequest(
                description='Payload must contain an "Instances" which must be a list of a single document'
            )

        max_new_tokens = headers.get("max_new_tokens", 128) if headers else 128

        text = (
            os.environ.get("PREFIX", "")
            + payload["instances"][0]
            + os.environ.get("SUFFIX", "")
        )
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"summary": summary}


def create_ray_deployment(model_name, model_version, num_replicas):
    @serve.deployment(
        name=model_name,
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": torch.cuda.device_count() / num_replicas},
    )
    class ModelDeployment(KServeModelForSeq2SeqLM):
        def __init__(self):
            super.__init__(name=model_name, version=model_version)

    return ModelDeployment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", required=True
    )
    parser.add_argument(
        "--model_version", help="Version of the model.", type=int, required=True
    )
    parser.add_argument(
        "--num_replicas", help="number of replicas", type=int, default=1
    )

    args, _ = parser.parse_known_args()
    deployment = create_ray_deployment(
        model_name=args.model_name,
        model_version=args.model_version,
        num_replicas=args.num_replicas,
    )

    kserve.ModelServer().start({args.model_name: deployment})
