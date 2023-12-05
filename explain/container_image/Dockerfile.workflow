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

ARG BASE_CONTAINER
FROM $BASE_CONTAINER

LABEL maintainer="Nick Lawrence ntl@us.ibm.com"

USER root

# data science stack
RUN mamba install \
    -c rocketce \
    -c defaults \
    --yes \
    main::python-kubernetes==23.6.0 \
    main::pandas==2.0.3 \
    main::scikit-learn==1.3.0 \
    main::scikit-image==0.19.3 \
    main::spacy==3.5.3 \
    rocketce::numpy==1.23.5 \
    main::matplotlib==3.7.2 \
    statsmodels==0.13.5 \
    main::nltk==3.8.1 \
    main::plotly==5.9.0 \
    main::pyyaml==5.4.1 \
    rocketce::tensorflow-cpu==2.12.0 \
    rocketce::tensorflow-io \
    rocketce::tensorflow-estimator \
    rocketce::onnxruntime==1.15.1=hd867603_cpu_py39_pb4.21.12_1 \
    rocketce::tf2onnx==1.13.0 \
    && mamba clean --all --yes \
    && fix-permissions ${CONDA_DIR}

# db2 and data quality
RUN pip install ibm_db==3.2.0 \
                evidently==0.2.6 \
                alibi==0.9.4 \
    && pip cache purge \
    && fix-permissions ${CONDA_DIR} \
    && fix-permissions ${HOME}

RUN pip check

USER 1000:0