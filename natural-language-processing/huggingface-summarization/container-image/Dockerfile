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
RUN dnf upgrade -y almalinux-release
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
RUN curl --location --silent --output  /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest && \
    tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  && \
    rm /tmp/micromamba.tar && \
    mkdir -p ${CONDA_DIR} && \
    echo "Installing packages...." && \
    ./micromamba install --root-prefix=${CONDA_DIR} --prefix=${CONDA_DIR} \
    -c rocketce \
    -c https://repo.anaconda.com/pkgs/main \
    --yes \
    rocketce::python==3.10.9  \
    pip==23.3 \
    rocketce::pytorch==2.0.1 \
    rocketce::nccl==2.17.1 \
    rocketce::cudatoolkit==11.8.0 \
    huggingface::datasets==2.16.1  \
    # RocketCE versions didn't work error similar to
    # https://github.com/huggingface/transformers/issues/21805
    main::transformers==4.32.1  \
    main::tokenizers==0.13.2 \
    # main huggingface_accelerate is older
    conda-forge::accelerate==0.23.0 \
    conda-forge::rouge-score==0.1.2 \
    conda-forge::rouge \
    conda-forge::evaluate==0.4.1 \
    # conda-forge tensorboard has more flexible dependencies
    # than the version from rocketce
    conda-forge::tensorboard==2.12.3 \ 
    && mv ./micromamba ${CONDA_DIR}/bin \
    && pip cache purge \
    && fix-permissions ${CONDA_DIR} 

# These packages should stay at the same level for inferencing,
# do not allow them to be up or downgraded accidently
RUN   echo "python==3.10.9" >> ${CONDA_DIR}/conda-meta/pinned \
   && echo "pytorch==2.0.1" >> ${CONDA_DIR}/conda-meta/pinned \
   && echo "transformers==4.32.1" >> ${CONDA_DIR}/conda-meta/pinned \
   && echo "tokenizers==0.13.2"

WORKDIR /.cache
RUN fix-permissions /.cache

WORKDIR /root
RUN fix-permissions /root

CMD ["bash"]
USER 1000:0