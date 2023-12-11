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

# These arguments define where to install the distributed kf tools package from.
ARG DISTRIBUTED_KF_TOOLS_BRANCH="2.0.3"
ARG DISTRIBUTED_KF_TOOLS_SUBDIR="distributed_training/distributed_kf_tools"
ARG DISTRIBUTED_KF_TOOLS_GIT="https://github.com/ntl-ibm/kubeflow-ppc64le-examples@${DISTRIBUTED_KF_TOOLS_BRANCH}#subdirectory=${DISTRIBUTED_KF_TOOLS_SUBDIR}"

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
RUN curl --location --silent --output /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest && \
    tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  && \
    rm /tmp/micromamba.tar && \
    mkdir -p ${CONDA_DIR} && \
    ./micromamba install --root-prefix=${CONDA_DIR} --prefix=${CONDA_DIR} \
    -c rocketce \
    -c https://repo.anaconda.com/pkgs/main \
    -c conda-forge \
    --yes \
    rocketce::python==3.10.9  \
    pip  \
    mamba \
    rocketce::pytorch==2.0.1 \
    rocketce::pytorch-lightning==2.0.1 \
    rocketce::torchvision==0.15.2 \
    rocketce::torchmetrics==0.11.4 \
    rocketce::nccl==2.17.1 \
    cudatoolkit==11.8.0 \
    && rm /tmp/micromamba

RUN pip install "distributed_kf_tools @ git+${DISTRIBUTED_KF_TOOLS_GIT}" \
    && dnf clean all && rm -rf /var/cache/dnf/* && rm -rf /var/cache/yum  \
    && pip cache purge \
    && fix-permissions ${CONDA_DIR}

WORKDIR /root
RUN fix-permissions /root

CMD ["bash"]
USER 1000:0

ENTRYPOINT ["python", "-m", "torch.distributed.run"]
