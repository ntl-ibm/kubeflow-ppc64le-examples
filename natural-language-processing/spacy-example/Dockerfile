FROM quay.io/ibm/kubeflow-notebook-image-ppc64le:elyra3.14.1-py3.9-tf2.10.1-pt1.10.2-v0

# In this specific FROM image, the default channel list includes a few that are
# private to IBM. "--override-channels" uses only the channels on the command list (rocketce)
RUN mamba install -y --override-channels -c rocketce spacy && \
    mamba clean -ay

RUN python -m spacy download en_core_web_md