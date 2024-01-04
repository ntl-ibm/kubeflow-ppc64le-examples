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

IOU = float(os.environ.get("IOU", "0.7"))
CONF = float(os.environ.get("CONF", "0.25"))

app = Flask(__name__, instance_relative_config=True)


@app.route("/alive", methods=["GET"])
def alive():
    return Response(status=http.HTTPStatus.OK)


@app.route("/detect", methods=["POST"])
def detect():
    content_type = request.headers.get("Content-Type")
    if not (content_type and content_type.lower() == "image/jpeg"):
        raise exceptions.UnsupportedMediaType()

    image = Image.open(io.BytesIO(request.data))

    model = YOLO("/mnt/models/model.pt", task="detect")

    results = model(
        image,
        iou=IOU,
        conf=CONF,
    )

    # predicted_image = results.render()[0]
    result_image = results[0].plot(
        conf=True, pil=True, boxes=True, labels=True, probs=True
    )

    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    return Response(
        buffered.getvalue(),
        status=http.client.OK,
        headers={"Content-Type": "image/jpeg"},
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
