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

import kserve
import numpy as np
import onnxruntime as ort
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferRequest
from kserve.protocol.infer_type import InferOutput, InferRequest, InferResponse


class CreditRiskPredictor(kserve.Model):
    MODEL_PATH = "/mnt/models/model.onnx"
    RISK_THRESHOLD = float(os.environ["THRESHOLD"])

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()
        self.ready = True

    def load(self):
        logging.info(
            f"Starting a CPU inference session using the model {CreditRiskPredictor.MODEL_PATH}"
        )
        self.INFERENCE_SESSION = ort.InferenceSession(
            CreditRiskPredictor.MODEL_PATH, providers=["CPUExecutionProvider"]
        )

    def predict(
        self,
        payload: Union[Dict, InferRequest, ModelInferRequest],
        headers: Dict[str, str] = None,
    ) -> InferResponse:
        logging.info(f"Predict was called with a payload type of {type(payload)}")
        if isinstance(payload, ModelInferRequest):
            payload = InferRequest.from_grpc(payload)
        elif isinstance(payload, Dict):
            payload = InferRequest.from_rest(payload)

        logging.debug("Scoring Risk...")
        scores = self.INFERENCE_SESSION.run(
            [], {"input_1": payload.inputs[0].as_numpy()}
        )

        logging.debug(f"Applying a threshold of {CreditRiskPredictor.RISK_THRESHOLD}")
        result = np.array(
            [
                1 if score > CreditRiskPredictor.RISK_THRESHOLD else 0
                for score in np.array(scores).flatten()
            ],
            dtype=np.uint8,
        )

        logging.debug("Encoding response")
        output_0 = InferOutput(name="risk", shape=result.shape, datatype="UINT8")
        output_0.set_data_from_numpy(result)

        return InferResponse(
            response_id=payload.id, model_name=self.name, infer_outputs=[output_0]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

    # This arg is automatically provided by the inferenceservice
    # it contains the name of the inference service resource.
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", required=True
    )

    args, _ = parser.parse_known_args()

    model = CreditRiskPredictor(name=args.model_name)
    kserve.ModelServer().start([model])
