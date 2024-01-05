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

from flask import Flask, Response, request
import http
from werkzeug import exceptions
from PIL import Image
import io
import os
from ultralytics import YOLO
import http.client
from download_model import download_s3
import logging

IOU = float(os.environ.get("IOU", "0.7"))
CONF = float(os.environ.get("CONF", "0.25"))
MODEL_DIR = "./models"
MODEL_PATH = f"{MODEL_DIR}/model.pt"

app = Flask(__name__, instance_relative_config=True)
app.logger.setLevel(level=logging.INFO)


@app.route("/alive", methods=["GET"])
def alive():
    return Response(status=http.HTTPStatus.OK)


@app.route("/detect", methods=["POST"])
def detect():
    content_type = request.headers.get("Content-Type")
    if not (content_type and content_type.lower() == "image/jpeg"):
        raise exceptions.UnsupportedMediaType()

    image = Image.open(io.BytesIO(request.data))

    app.logger.info(f"Loading model {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    results = model(
        image,
        iou=IOU,
        conf=CONF,
    )

    app.logger.info("Plotting detected objects")
    result_image = results[0].plot(
        conf=True, pil=True, boxes=True, labels=True, probs=True
    )

    app.logger.info("Return response")
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    return Response(
        buffered.getvalue(),
        status=http.client.OK,
        headers={"Content-Type": "image/jpeg"},
    )


if __name__ == "__main__":
    download_s3(os.environ["STORAGE_URI"], MODEL_DIR)
    app.run(debug=False, host="0.0.0.0", port=8080)
