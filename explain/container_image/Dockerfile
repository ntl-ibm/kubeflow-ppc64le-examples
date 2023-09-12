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

USER root
WORKDIR /root

# The trick here is to install into a root prefix that is first in the path
# Then the python environment will be what we installed using mamba, but we
# will not need to directly activate the environment.
ENV CONDA_DIR="/opt/conda"
ENV PATH=${CONDA_DIR}/bin:${PATH}

## OS Dependencies
RUN dnf install -y bzip2  \
    && dnf group -y install "Development Tools" \
    && dnf clean all && rm -rf /var/cache/dnf/* && rm -rf /var/cache/yum 

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

## Install mamba and required packages
# https://mamba.readthedocs.io/en/latest/installation.html#manual-installation
WORKDIR /tmp
RUN curl --location --silent --output /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest  \
    && tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  \
    && rm /tmp/micromamba.tar \
    && mkdir -p ${CONDA_DIR}  \
    && ./micromamba install --root-prefix=${CONDA_DIR} --prefix=${CONDA_DIR} \
    -c rocketce \
    -c defaults \
    --yes \
    rocketce::python==3.9.16 \
    pip \
    mamba \
    conda \
    rocketce::tensorflow-cpu==2.12.0 \
    rocketce::sklearn-pandas==2.1.0 \
    rocketce::tensorboard==2.12.2 \
    rocketce::tf2onnx==1.13.0 \
    rocketce::onnxruntime==1.15.1=hd867603_cpu_py39_pb4.21.12_1 \
    rocketce::numactl \
    nltk \
    python-kubernetes \
    matplotlib \
    pillow \
    statsmodels==0.13.5  \
    && rm /tmp/micromamba \
    && mkdir ~/.pip  \
    && echo "[global]" >> ~/.pip/pip.conf  \
    && echo "extra-index-url = https://repo.fury.io/mgiessing" >> ~/.pip/pip.conf \
    && fix-permissions ${CONDA_DIR}  \
    && fix-permissions ~/.pip

RUN pip install kserve==0.10.1 && pip cache purge
RUN pip install ibm-db==3.2.0 && pip cache purge
RUN pip install alibi==0.9.4 && pip cache purge
RUN pip install evidently==0.2.6  && pip cache purge
RUN fix-permissions ${CONDA_DIR}  
RUN fix-permissions ~/.pip



WORKDIR /workspace
ENV PYTHONPATH /workspace
RUN fix-permissions /workspace

USER 1000:0