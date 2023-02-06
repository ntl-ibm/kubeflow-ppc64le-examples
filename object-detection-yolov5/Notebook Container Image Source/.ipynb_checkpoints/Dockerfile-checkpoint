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

ARG IMAGE=quay.io/ibm/kubeflow-notebook-image-ppc64le:elyra3.14.1-py3.9-tf2.9.2-pt1.12.1-v0

FROM $IMAGE AS build_wheels
USER jovyan
RUN mamba install -y -c conda-forge gfortran==11.2.0
RUN mamba install -y  cmake
WORKDIR /home/jovyan/
RUN  pip wheel -v opencv-python  --prefer-binary --w /home/jovyan/wheels


FROM $IMAGE
ARG YOLOV5_TAG=v7.0
USER root
RUN mkdir /opencv-python && \
    chown "${NB_USER}:${NB_GID}" /opencv-python && \
    mkdir /yolov5 && \
    chown "${NB_USER}:${NB_GID}" /yolov5

USER jovyan
WORKDIR /opencv-python
COPY --from=build_wheels /home/jovyan/wheels .

WORKDIR /yolov5
RUN git clone -b $YOLOV5_TAG https://github.com/ultralytics/yolov5 . && \
    mamba install -y openblas && \
    pip install -r requirements.txt --find-links /opencv-python

WORKDIR "${HOME}"