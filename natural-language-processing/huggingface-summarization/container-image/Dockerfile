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

# This container includes packages and tooling to Run PyTorch distributed on
# IBM Power 9 or Power 10 hardware.
# It does NOT include a notebook server.

USER root
WORKDIR /root

# The trick here is to install into a root prefix that is first in the path
# Then the python environment will be what we installed using mamba, but we
# will not need to directly activate the environment.
ENV CONDA_DIR="/opt/conda"
ENV PATH=${CONDA_DIR}/bin:${PATH}

## OS Dependencies
RUN dnf install -y bzip2 \
        git \
        wget && \
    dnf clean all -y 

# dnf group install -y "Development Tools" && \

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

## Install mamba and required packages
# https://mamba.readthedocs.io/en/latest/installation.html#manual-installation
WORKDIR /tmp
RUN curl --location --silent --output  /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/1.4.2 && \
    tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  && \
    rm /tmp/micromamba.tar && \
    mkdir -p ${CONDA_DIR} && \
    ./micromamba install --root-prefix=${CONDA_DIR} --prefix=${CONDA_DIR} \
    -c rocketce \
    -c https://repo.anaconda.com/pkgs/main \
    --yes \
    rocketce::python==3.10.9  \
    pip==23.3 \
    mamba==1.4.2 \
    conda==23.7.4 \
    rocketce::pytorch==2.0.1 \
    rocketce::nccl==2.17.1 \
    rocketce::cudatoolkit==11.8.0 \
    huggingface::datasets==2.16.1 \
    huggingface::transformers==4.33.3 \
    conda-forge::optimum==1.16.1 \
    conda-forge::rouge-score==0.1.2 \
    conda-forge::rouge \
    # For KServe
    six==1.16.0 \
    python-dateutil \
    python-kubernetes \
    minio==7.1.0 \
    google-cloud-storage>=1.20.0 \
    adal>=1.2.7 \
    tabulate>=0.9.0 \
    numpy>=1.21.5 \
    #azure-storage-blob==12.9.0 \
    #azure-storage-file-share==12.7.0 \
    #azure-identity>=1.8.0 \
    #cloudevents>=1.6.2 \
    #avro>=1.11.0 \
    boto3==1.21.32 \
    psutil>=5.9.0 \
    # ray[serve]==2.0.0 (only 2.0.1 is in anaconda/rocketce) \
    # tritonclient==2.18.0 \
    protobuf>=3.19.0 \
    prometheus_client>=0.13.1 \
    conda-forge::grpcio==1.43.0   \
    orjson>=3.8.0 \
    httpx>=0.23.0 \
    rocketce::fastapi==0.85.1 \
    # timing_asgi>=0.3.0 \
    click==8.0.4 \
    jsonschema \
    virtualenv \
    uvicorn==0.16.0 \
    aiorwlock \
    conda-forge::gpustat \
    msgpack-python \
    && pip install --extra-index-url https://repo.fury.io/mgiessing ray[serve]==2.0.0 \
    && pip install kserve==0.10.1 \
    && pip cache purge \
    && fix-permissions ${CONDA_DIR} \
    && rm /tmp/micromamba \
    && fix-permissions ${CONDA_DIR}  

WORKDIR /root
RUN fix-permissions /root

CMD ["bash"]
USER 1000:0
