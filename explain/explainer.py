from typing import Dict, Any, Optional, Union
import pandas as pd
import http
from ray import serve
import logging
import joblib
import tornado
import kserve
from kserve.protocol.infer_type import InferRequest, InferInput, InferResponse
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
import argparse
import numpy as np
import sys
import dill

class CreditRiskExplainer(kserve.Model):
    EXPLAINER_PATH = "/mnt/models/explainer.dll"
    
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.preprocessor = None
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.load()
        self.ready = True
        
    def load(self):
        with open(CreditRiskExplainer.EXPLAINER_PATH, "r") as f:
            self.explainer = dill.load(f)
            self.explainer.predictor = self._predict_fn
        
    def _predict_fn(self, arr: Union[np.ndarray, List]) -> np.ndarray:
        if isinstance(arr, List):
            X = np.array(arr).astype(np.float32)
        elif isinstance(arr, np.ndarray):
            X = arr.astype(np.float32)
        else:
            raise NotImplementedError()
            
        input_0 = InferInput(name="input_1", shape=X.shape, datatype="FP32", data=X)
        request = InferRequest(model_name=self.name, infer_inputs=[input_0], request_id=None)
        
        response = self.predict(request)
        
        if isinstance(response, ModelInferResponse):
            response = InferResponse.from_grpc(response)
        elif isinstance(response, Dict):
            raise NotImplementedError("Json input to postprocess is not supported")
        
        return response.outputs[0].as_numpy()
        
    def explain(self, payload: Union[Dict, InferRequest, ModelInferRequest], headers: Dict[str, str] = None) -> Dict:
        if isinstance(payload, ModelInferRequest):
            payload = InferRequest.from_grpc(payload)
        elif isinstance(payload, Dict):
            payload = InferRequest.from_rest(payload)
            
        X = payload.inputs[0].as_numpy()
        explanation = explainer.explain(
                        X,
                        threshold=0.95,
                        batch_size=200,
                        beam_size=20,
                        min_samples_start=10000,
        )
        
        return json.loads(explanation.to_json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    # This arg is automatically provided by the inferenceservice
    # It contains the service host name for the predictor service.
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )


    # Deterimes whether REST or GRPC will be used to communicate with the predictor
    # The default is REST/JSON format. The value provided must match the interface
    # that is exposed by the predictor.
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )
    args, _ = parser.parse_known_args()
    
    model = CreditRiskTransformer(
        name="credit-risk",
        predictor_host=args.predictor_host,
        protocol=args.protocol
    )
    kserve.ModelServer().start([model])
