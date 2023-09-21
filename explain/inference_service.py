import kserve
from typing import Dict, Union
import os
import numpy as np
from ray import serve
import logging
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferRequest
import onnxruntime as ort
from kserve.protocol.infer_type import InferRequest, InferResponse, InferOutput
import argparse

logging.basicConfig()
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.INFO)

class CreditRiskPredictor(kserve.Model):
    MODEL_PATH = "/mnt/models/model.onnx"
    RISK_THRESHOLD = float(os.environ["THRESHOLD"])

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()
        self.ready = True

    def load(self):
        self.INFERENCE_SESSION = ort.InferenceSession(
            CreditRiskPredictor.MODEL_PATH, providers=["CPUExecutionProvider"]
        )

    def predict(
        self,
        payload: Union[Dict, InferRequest, ModelInferRequest],
        headers: Dict[str, str] = None,
    ) -> InferResponse:
        if isinstance(payload, ModelInferRequest):
            payload = InferRequest.from_grpc(payload)
        elif isinstance(payload, Dict):
            payload = InferRequest.from_rest(payload)

        scores = self.INFERENCE_SESSION.run([], {"input_1": payload.inputs[0].as_numpy()})

        result = np.array(
            [1 if score > CreditRiskPredictor.RISK_THRESHOLD else 0 
             for score in np.array(scores).flatten()],
            dtype=np.uint8,
        )

        output_0 = InferOutput(name="risk", shape=result.shape, datatype="UINT8")
        output_0.set_data_from_numpy(result)

        return InferResponse(
            response_id=payload.id, model_name=self.name, infer_outputs=[output_0]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

    args, _ = parser.parse_known_args()
    
    model = CreditRiskPredictor(name="credit-risk")
    kserve.ModelServer().start([model])
