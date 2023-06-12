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
ARG ROOT_CONTAINER=quay.io/almalinux/almalinux:8.6
FROM $ROOT_CONTAINER
LABEL maintainer="Nick Lawrence ntl@us.ibm.com"

# These arguments define where to install the distributed kf tools package from.
ARG DISTRIBUTED_KF_TOOLS_BRANCH="multi-gpu-yolov5"
ARG DISTRIBUTED_KF_TOOLS_SUBDIR="pytorch_distributed/pytorch_distributed_kf_tools"
ARG DISTRIBUTED_KF_TOOLS_GIT="https://github.com/ntl-ibm/kubeflow-ppc64le-examples@${DISTRIBUTED_KF_TOOLS_BRANCH}#subdirectory=${DISTRIBUTED_KF_TOOLS_SUBDIR}"

USER root
WORKDIR /root

# The trick here is to install into a root prefix that is first in the path
# Then the python environment will be what we installed using mamba, but we
# will not need to directly activate the environment.
ENV CONDA_DIR="/opt/conda"
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV YOLOV5_CONFIG_DIR="/yolov5-config"

## OS Dependencies
RUN dnf install -y bzip2 \
        git \
        wget && \
    dnf clean all -y 

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

## Install mamba
# https://mamba.readthedocs.io/en/latest/installation.html#manual-installation
WORKDIR /tmp
RUN curl --location --silent --output /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest && \
    tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  && \
    rm /tmp/micromamba.tar && \
    mkdir -p ${CONDA_DIR} && \
    ./micromamba install --root-prefix=${CONDA_DIR} --prefix=${CONDA_DIR} \
    -c rocketce \
    -c https://repo.anaconda.com/pkgs/main \
    -c conda-forge \
    --yes \
    python==3.9.16  \
    pandas>=1.1.4 \
    pip  \
    mamba \
    gitpython  \
    matplotlib  \
    pillow \
    psutil \
    PyYAML \
    requests \
    scipy \
    rocketce::pytorch==1.13.1 \
    rocketce::torchvision \
    tqdm \
    nccl \
    cudatoolkit \
    && rm /tmp/micromamba \
    && fix-permissions ${CONDA_DIR}  \
    && mkdir ~/.pip  \
    && echo "[global]" >> ~/.pip/pip.conf  \
    && echo "extra-index-url = https://repo.fury.io/mgiessing" >> ~/.pip/pip.conf \
    && pip install "pytorch_distributed_kf_tools @ git+${DISTRIBUTED_KF_TOOLS_GIT}" \
    && dnf clean all && rm -rf /var/cache/dnf/* && rm -rf /var/cache/yum  \
    && pip cache purge

WORKDIR /wheels/opencv-python
COPY --from=quay.io/ntlawrence/opencv-python:1.0.1  /wheels .

# Checks out a specific version of the yolov5 tools.
# This is a little newer that the latest release, but has some
# fixes in it for distriubed.
WORKDIR /yolov5
RUN git clone https://github.com/ultralytics/yolov5 . && \
    git checkout a199480ba6bb527598df11abbc1d679ccda82670 && \
    git config --system --add safe.directory '/yolov5' && \
    pip install -r requirements.txt --find-links /wheels/opencv-python

COPY train.py /yolov5
RUN fix-permissions /yolov5

# When running distributed training, this file gets downloaded in all worker pods at the same time
# The download of the same file multiple times in a small timewindow causes the download request to
# be rejected. 
# https://stackoverflow.com/questions/71251177/how-to-use-yolov5-api-with-flask-offline
WORKDIR /yolov5-config
RUN wget https://ultralytics.com/assets/Arial.ttf  -O /yolov5-config/Arial.ttf
RUN fix-permissions /yolov5-config

WORKDIR /yolov5
USER 1000:0
