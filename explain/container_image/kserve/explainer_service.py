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
Implements a tabular anchor explainer for the credit-risk demo.

In KServe 10, the transformer's preprocessor does not get invoked
prior to invoking the explainer. That's why the explain function
needs to do its own pre and post processing.

The explainer WILL make many GRPC calls to the predictor as it
searches for examples 'close' to the input that change the prediction result.

Author: ntl@us.ibm.com
"""
import argparse
import http
import logging
from typing import Dict, List, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

import dill
import joblib
import kserve
import numpy as np
import pandas as pd
from fastapi import HTTPException
from alibi.explainers.anchors.anchor_tabular import AnchorTabular
from kserve.protocol.infer_type import InferInput, InferRequest, InferResponse
from kserve.protocol.grpc.grpc_predict_v2_pb2 import (
    ModelInferRequest,
    ModelInferResponse,
)
from sklearn.pipeline import Pipeline


class CreditRiskExplainer(kserve.Model):
    EXPLAINER_PATH = "/mnt/models/explainer.dll"
    PREPROCESSOR_PATH = "/mnt/models/preprocessor.joblib"
    explainer: AnchorTabular = None
    preprocessor: Pipeline = None

    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.async_executor = ThreadPoolExecutor(
            max_workers=1, initializer=self.sync_predict_init
        )
        self.load()
        self.ready = True
        logging.info(
            f"Started server with predictor {self.predictor_host}, protocol {self.protocol}"
        )

    def load(self):
        logging.info(f"Loading explainer from {CreditRiskExplainer.EXPLAINER_PATH}")
        with open(CreditRiskExplainer.EXPLAINER_PATH, "rb") as f:
            self.explainer = dill.load(f)
            self.explainer.reset_predictor(self._predict_fn)

        logging.info(
            f"Loading preprocessor from {CreditRiskExplainer.PREPROCESSOR_PATH}"
        )
        self.preprocessor = joblib.load(CreditRiskExplainer.PREPROCESSOR_PATH)

    def sync_predict_init():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def sync_predict(
        self,
        payload: Union[Dict, InferRequest, ModelInferRequest],
        headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse, ModelInferResponse]:
        loop = asyncio.get_event_loop()

        return loop.run_until_complete(self.predict(payload, headers={}), debug=True)

    def _predict_fn(self, arr: Union[np.ndarray, List]) -> np.ndarray:
        """
        Prediction function
        Parameters:
            arr: numpy array of input features
        Returns:
            numpy array of predicted class (1 or 0 for each example)

        This will make a grpc request to the prediction pod and deserialize the response.
        """
        if isinstance(arr, List):
            X = np.array(arr).astype(np.float32)
        elif isinstance(arr, np.ndarray):
            X = arr.astype(np.float32)
        else:
            raise NotImplementedError()

        logging.info("Bulding inference request")
        input_0 = InferInput(name="input_1", shape=X.shape, datatype="FP32", data=X)
        request = InferRequest(
            model_name=self.name, infer_inputs=[input_0], request_id=None
        )

        logging.info("Invoking inference request")
        future = self.async_executor.submit(self.sync_predict)
        response = future.result()

        logging.info(f"Deserializing respone of type {type(response)}")
        response = InferResponse.from_grpc(response)

        return response.outputs[0].as_numpy()

    def _explain_example(self, X: np.ndarray) -> Dict:
        """Explains a specific example.

        The explainer cannot handle explaining a batch of examples.

        Parameters:
            X - numpy array containing features from a single example
        Returns:
            Dictionary with a simplified explanation.
        """
        explanation = self.explainer.explain(
            X,
            threshold=0.95,
            batch_size=200,
            beam_size=20,
            min_samples_start=10000,
        )
        return {
            "anchor": explanation.data["anchor"],
            "precision": explanation.data["precision"],
            "coverage": explanation.data["coverage"],
        }

    def explain(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        if not isinstance(payload, Dict):
            raise HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail="The provided input must be a json dictionary",
            )

        try:
            logging.info(f"Preprocessing payload {payload}")
            X = self.preprocessor.transform(pd.DataFrame(payload)).astype(np.float32)
        except ValueError as e:
            logging.exception(e)
            raise HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=f"Invalid Request: {str(e)}",
            ) from None

        logging.info(f"Explaining preprocessed inputs (shape {X.shape})")
        return {"explanations": [self._explain_example(example) for example in X]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    # This arg is automatically provided by the inferenceservice
    # It contains the service host name for the predictor service.
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )

    # This arg is automatically provided by the inferenceservice
    # it contains the name of the inference service resource.
    parser.add_argument(
        "--model_name", help="The name that the model is served under.", required=True
    )

    # Deterimes whether REST or GRPC will be used to communicate with the predictor
    # The default is REST/JSON format. The value provided must match the interface
    # that is exposed by the predictor.
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )

    args, _ = parser.parse_known_args()

    model = CreditRiskExplainer(
        name=args.model_name, predictor_host=args.predictor_host, protocol=args.protocol
    )
    kserve.ModelServer().start([model])
