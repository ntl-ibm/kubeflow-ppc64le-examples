# Copyright 2024 IBM All Rights Reserved.
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
This module implements a simple flask server that accepts a JPEG image and 
returns the JPEG with bounding boxes of detected objects.

Author: ntl@us.ibm.com
"""
from flask import Flask, Response, request, send_file
import http
from werkzeug import exceptions
from PIL import Image
import io
import os
from ultralytics import YOLO
import http.client
from download_model import download_s3
import logging
import sys

# Configuration comes from the environment
IOU = float(os.environ.get("IOU", "0.7"))
CONF = float(os.environ.get("CONF", "0.25"))
FONT_SIZE = float(os.environ["FONT_SIZE"]) if "FONT_SIZE" in os.environ else None
LINE_WIDTH = float(os.environ["LINE_WIDTH"]) if "LINE_WIDTH" in os.environ else None
MODEL_DIR = "./models"
MODEL_PATH = f"{MODEL_DIR}/model.pt"


app = Flask(__name__, instance_relative_config=True)

# Setup logging to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(level=logging.INFO)

app.logger.setLevel(level=logging.INFO)


@app.route("/alive", methods=["GET"])
def alive():
    """Health check API used for readiness probes"""
    return Response(status=http.HTTPStatus.OK)


@app.route("/detect", methods=["POST"])
def detect():
    """Detection API

    Input: JPEG image
    OUTPUT: JPEG image with detections drawn
    """
    app.logger.info("Reading input image")
    content_type = request.headers.get("Content-Type")
    if not (content_type and content_type.lower() == "image/jpeg"):
        raise exceptions.UnsupportedMediaType()

    try:
        image = Image.open(io.BytesIO(request.data))
    except Exception:
        raise exceptions.BadRequest("Unable to read JPEG image")

    # Model is loaded in this method every time because documentation says the model
    # should not be shared across threads:
    # https://docs.ultralytics.com/guides/yolo-thread-safe-inference/#thread-safe-example
    app.logger.info(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    app.logger.info(f"Starting inference")
    results = model(
        image,
        iou=IOU,
        conf=CONF,
        font_size=FONT_SIZE,
        line_width=LINE_WIDTH,
    )

    # Plotting example taken from
    # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
    app.logger.info("Plotting detected objects")
    result_image_array = results[0].plot()  # plot a BGR numpy array of predictions
    result_image = Image.fromarray(result_image_array[..., ::-1])

    app.logger.info("Returning response as jpeg")
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    return send_file(io.BytesIO(buffered.getvalue()), mimetype="image/jpeg")


if __name__ == "__main__":
    # Download model from S3 to container storage
    download_s3(os.environ["STORAGE_URI"], MODEL_DIR)

    # The default behavior for development server is 1 request at a time
    # https://stackoverflow.com/questions/10938360/how-many-concurrent-requests-does-a-single-flask-process-receive
    app.run(debug=False, host="0.0.0.0", port=8080)
