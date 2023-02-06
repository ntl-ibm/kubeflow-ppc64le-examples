FROM quay.io/ibm/kubeflow-notebook-image-ppc64le:elyra3.14.1-py3.9-tf2.9.2-pt1.12.1-v0

RUN mamba install -y -c rocketce spacy && \
    mamba clean -ay

RUN python -m spacy download en_core_web_md