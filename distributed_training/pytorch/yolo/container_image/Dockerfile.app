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
WORKDIR /root


COPY app.py .
COPY download_model.py .
RUN fix-permissions /root

CMD ["bash"]
USER 1000:0