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
FROM quay.io/almalinux/almalinux:8.6
LABEL maintainer="Nick Lawrence <ntl@us.ibm.com>"
USER root
WORKDIR /root

# The trick here is to install into a root prefix that is first in the path
# Then the python environment will be what we installed using mamba, but we
# will not need to directly activate the environment.
ENV CONDA_PREFIX="/opt/conda"
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

## OS Dependencies
RUN dnf install -y bzip2 && \
    dnf clean all -y 

## Install mamba
# https://mamba.readthedocs.io/en/latest/installation.html#manual-installation
RUN curl --location --silent --output /tmp/micromamba.tar https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest && \
    tar --extract --file /tmp/micromamba.tar --strip-components=1 bin/micromamba  && \
    rm /tmp/micromamba.tar

## Install python, pip, pillow (required by transformer)
RUN ./micromamba install --root-prefix=${CONDA_PREFIX} --prefix=${CONDA_PREFIX} \
    -c rocketce \
    -c conda-forge \
    --yes \
    python==3.9 \
    pip \
    pillow>=9.4.0 && \
    ./micromamba --root-prefix=${CONDA_PREFIX} --prefix=${CONDA_PREFIX} clean --all

## Install Kserve
# Install of kserve needs a dependent packages (Ray) that is in the mgiessing repo.
# If we didn't have that repo available, we would have to clone the github source
# for the Ray package and build from scratch.
RUN pip install --prefer-binary --no-cache-dir --extra-index-url="https://repo.fury.io/mgiessing" kserve==0.9.0

## Install source scripts
WORKDIR /app/kserve
# Opeshift requires images to support arbitrary user ids
# https://docs.openshift.com/container-platform/4.12/openshift_images/create-images.html
RUN chgrp -R 0 /app/kserve && \
    chmod -R g=u /app/kserve

COPY *.py .

# Vanilla Kubernetes will run under the user id defined here
# For security reasons, the container should never run under root
USER 1001



