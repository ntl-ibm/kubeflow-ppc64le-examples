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

"""
Environment variables that can be added to pytorch workers to provide enhanced
debug capabilities.
"""

from kubernetes.client import V1EnvVar

LOGLEVEL_INFO = V1EnvVar(name="LOGLEVEL", value="INFO")
NCCL_INFO = V1EnvVar(name="NCCL_DEBUG", value="INFO")
C10D_DEBUG_MODE_DETAIL = V1EnvVar(name="C10D_DEBUG_MODE", value="DETAIL")
