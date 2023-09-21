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
import onnx


class CreditRiskTransformer(kserve.Model):
    MODEL_INPUT_NAME = "input"
    PREPROCESSOR_PATH = "/mnt/models/preprocessor.joblib"
    MODEL_PATH = "/mnt/models/model.onnx"
    
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.preprocessor = None
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.load()
        self.ready = True
        
    def load(self):
        self.preprocessor = joblib.load(CreditRiskTransformer.PREPROCESSOR_PATH)
        model = onnx.load(CreditRiskTransformer.MODEL_PATH)
        
    def preprocess(
        self, inputs: Dict, headers: Optional[Dict[str, str]] = None
    ) -> InferRequest:
        """Preprocesses the incomming json"""
        try:
            X = self.preprocessor.transform(pd.DataFrame(inputs)).astype(np.float32)
            
            input_0 = InferInput(name="input_1", shape=X.shape, datatype="FP32", data=X)
            request = InferRequest(model_name=self.name, infer_inputs=[input_0], request_id=None)
            return request
        except (ValueError, KeyError, TypeError, IOError, AttributeError) as e:
            logging.exception(e)
            raise tornado.web.HTTPError(
                status_code=http.HTTPStatus.BAD_REQUEST,
                reason="The input data was not valid!",
            )


    def postprocess(self, response: Union[Dict, ModelInferResponse, InferResponse], headers: Dict[str,str]) -> Dict[Any, Any]:
        """Convert model outputs from the prediction to the JSON response"""
        
        if isinstance(response, ModelInferResponse):
            response = InferResponse.from_grpc(response)
        elif isinstance(response, Dict):
            raise NotImplementedError("Json input to postprocess is not supported")
            
        tensor = response.outputs[0].as_numpy()
        payload = {
            "predictions": ["Risk" if score == 1 else "No Risk" for score in tensor]
        }

        return payload


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
