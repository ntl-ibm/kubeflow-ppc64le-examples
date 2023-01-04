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
This is an example KServe transformer for use with the monkey classification example.


The transformer accepts a single image and returns class scores for the image.

Comments for advanced Batch Processing:
The Model supports batches of images, this transformer will send a batch of size 1 to the predictor, 
and interpret a batch of size 1 responses.

If this example needs to be changed to process a batch of images using a Triton backend, the batching capabiliites
of Triton need to be considered. The `max_batch_size` configuration parameter may need to be adjusted.
See: 
https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size
https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#scheduling-and-batching

The batch configuration is controlled by a configuration file in the model repository:
https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#repository-layout

Comments for gRPC support:
The code supports either a gRPC and HTTP/REST protocol for communicating with the predictor. When the inference service is deployed,
only one of these protocols will be exposed to the transformer. HTTP/REST is the default and the easiest to use. However,
transmitting large tensors using JSON has poor performance, gRPC will result in much faster transmissions. The instructions for
exposing the gRPC endpoint of a triton predictor can be found here:
https://kserve.github.io/website/modelserving/v1beta1/triton/torchscript/#inference-with-grpc-endpoint
The transformer will attempt to use the protocol defined by the "--protocol" argument on the deployment command.

Comments on class labels:
The transformer service is deployed with an environment value that contains the mapping of class lables.
Author: ntl@us.ibm.com
"""
import argparse
import base64
import json
import logging
from http import HTTPStatus
from typing import Dict, Optional, Union, Any
import io
import os

import kserve
import numpy as np
from PIL import Image
import tornado.web
from kserve.model import PredictorProtocol
from tritonclient.grpc.service_pb2 import ModelInferRequest, ModelInferResponse
from tritonclient.grpc import InferResult, InferInput

logging.basicConfig()
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.INFO)


class MonkeyImageTransformer(kserve.Model):
    MODEL_INPUT_NAME = "image"
    MODEL_OUTPUT_NAME = "scores"
    CLASS_LABELS = json.loads(os.environ["CLASS_LABELS"])

    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        logging.info(
            f"Transformer {name} on host {predictor_host} using protocol {protocol}"
        )
        self.ready = True

    def _v2_grpc_tensor_transform(self, input_tensors) -> ModelInferRequest:
        """Converts tensors to a gRPC request payload

        This is appropriate when the predictor has exposed a V2 gRPC interface.
        """
        request = ModelInferRequest()
        request.model_name = self.name
        input_0 = InferInput(
            MonkeyImageTransformer.MODEL_INPUT_NAME, input_tensors.shape, "UINT8"
        )
        input_0.set_data_from_numpy(input_tensors)
        request.inputs.extend([input_0._get_tensor()])
        if input_0._get_content() is not None:
            request.raw_input_contents.extend([input_0._get_content()])
        return request

    def _json_tensor_transform(self, input_tensor) -> Dict[Any, Any]:
        """Converts tensors to a JSON request payload

        This payload is appropriate when the predictor has exposed a V2 REST interface.
        """
        logging.debug("Building json payload")
        payload = {
            "inputs": [
                {
                    # Input name comes from the name of the input layer in the model
                    # The tensor we pass (as JSON) will be reshaped and cast to
                    # the expected data type.
                    "name": MonkeyImageTransformer.MODEL_INPUT_NAME,
                    "shape": input_tensor.shape,
                    "datatype": "UINT8",
                    "data": input_tensor.flatten().tolist(),
                }
            ]
        }
        return payload

    def preprocess(
        self, inputs: Dict, headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict[Any, Any], ModelInferRequest]:
        """Preprocesses an image for inference

        Input format:
        { "instance" :  {
                            "image" : "<b64 jpeg image data>"
                        }
        }

        The format of the returned Dict or ModelInferRequest matches the format expected by the predictor.
        The assumption is that the inference service was deployed to use the v2 model.
        https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md
        """
        logging.debug("decoding image and expanding dimensions to a batch")
        try:
            tensor = np.expand_dims(
                np.array(
                    Image.open(
                        io.BytesIO(
                            base64.b64decode(
                                inputs["instance"]["image"].encode("utf-8")
                            )
                        )
                    ),
                    dtype="uint8",
                ),
                0,
            )
        except (ValueError, KeyError, IOError, AttributeError) as e:
            logging.exception(e)
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="The input data was not valid!",
            )

        logging.debug(f"done preprocessing.")
        if self.protocol == PredictorProtocol.GRPC_V2.value:
            return self._v2_grpc_tensor_transform(tensor)

        return self._json_tensor_transform(tensor)

    def _json_model_output_to_tensor(self, model_output: Dict[Any, Any]) -> np.array:
        """Converts the model's output (JSON) to a tensor

        This is used when the predictor has exposed the REST inteface.
        """
        return np.array(model_output["outputs"][0]["data"])

    def _v2_grpc_model_output_to_tensor(
        self, model_output: ModelInferResponse
    ) -> np.array:
        """Converts the model's output (GRPC) to a tensor.

        This is used when the predictor has exposed the gRPC interface
        """
        response = InferResult(model_output)
        # float32 is not serialized by json, so we cast here to float
        return response.as_numpy(MonkeyImageTransformer.MODEL_OUTPUT_NAME).astype(float)

    def postprocess(
        self, model_output: Union[Dict[Any, Any], ModelInferResponse]
    ) -> Dict[Any, Any]:
        """Convert model outputs from the prediction to the JSON response

        This method assumes that the inference service was deployed to use the v2 JSON model.
        The model output formats are described at:
        https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md
        """

        logging.debug("post processing....")

        tensor = (
            self._v2_grpc_model_output_to_tensor(model_output)
            if self.protocol == PredictorProtocol.GRPC_V2.value
            else self._json_model_output_to_tensor(model_output)
        )

        batch_scores = tensor.reshape(-1, len(MonkeyImageTransformer.CLASS_LABELS))

        payload = {
            "predictions": [
                {
                    MonkeyImageTransformer.CLASS_LABELS[p]: scores[p]
                    for p in np.argsort(scores)[::-1]
                }
                for scores in batch_scores
            ]
        }

        logging.debug(f"done post processing. {payload}")
        return payload


if __name__ == "__main__":

    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    # This arg is automatically provided by the inferenceservice
    # It contains the service host name for the predictor service.
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )

    # This arg is automatically provided by the inferenceservice
    # it contains the name of the inference service resource.
    # For back-ends that support serving only a single model, we would expect this to
    # be the model name.
    parser.add_argument("--model_name", help="The name that the model is served under.")

    # Deterimes whether REST or GRPC will be used to communicate with the predictor
    # The default is REST/JSON format. The value provided must match the interface
    # that is exposed by the predictor.
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v2"
    )
    args, _ = parser.parse_known_args()

    # The name parameter needs to match to the model name on the infer request.
    # Some inference service backends can only serve one model, and then the model
    # name provided as the argument will (most likely) match. But triton can serve
    # multiple models, so in that case we need to know the name of the model that
    # the transformer is associated with, as there can be potentially many transformers
    # in that case.
    model = MonkeyImageTransformer(
        "monkey-classification",
        predictor_host=args.predictor_host,
        protocol=args.protocol,
    )
    kserve.ModelServer().start([model])
