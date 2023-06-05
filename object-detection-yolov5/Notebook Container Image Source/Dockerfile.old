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

# Based on an image that has fixes for NCCL and CUDA on IBM Power
ARG IMAGE=quay.io/ibm/kubeflow-notebook-image-ppc64le:elyra3.14.1-py3.9-tf2.9.2-pt1.12.1-v0

FROM $IMAGE AS build_wheels
USER jovyan
RUN mamba install -y -c conda-forge gfortran==11.2.0
RUN mamba install -y  cmake
WORKDIR /home/jovyan/
RUN  pip wheel -v opencv-python  --prefer-binary --w /home/jovyan/wheels

FROM $IMAGE
ARG YOLOV5_Checkout=cca5e21995
USER root
RUN mkdir /opencv-python && \
    chown "${NB_USER}:${NB_GID}" /opencv-python && \
    mkdir /yolov5 && \
    chown "${NB_USER}:${NB_GID}" /yolov5

USER jovyan
WORKDIR /opencv-python
COPY --from=build_wheels /home/jovyan/wheels .

WORKDIR /yolov5

# Checks out a specific version of the yolov5 tools.
# This is a little newer that the latest release, but has some
# fixes in it for distriubed.
RUN git clone https://github.com/ultralytics/yolov5 . && \
    git checkout $YOLOV5_Checkout && \
    pip install -r requirements.txt --find-links /opencv-python

WORKDIR "${HOME}"

# When running distributed training, this file gets downloaded in all worker pods at the same time
# The download of the same file multiple times in a small timewindow causes the download request to
# be rejected. 
# https://stackoverflow.com/questions/71251177/how-to-use-yolov5-api-with-flask-offline
RUN mkdir -p /home/jovyan/.config/Ultralytics
RUN wget https://ultralytics.com/assets/Arial.ttf  -O /home/jovyan/.config/Ultralytics/Arial.ttf