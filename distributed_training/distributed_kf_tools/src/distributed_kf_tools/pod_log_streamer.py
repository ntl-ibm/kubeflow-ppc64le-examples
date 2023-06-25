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
This module defines a class that will stream output from training
client workers.

Example:
with PodLogStreamer(
        "kubeflow-user-namespace",
        "job-name"
    ):
    # Logs will be streamed while this code runs

Author: ntl@us.ibm.com
"""
import multiprocessing
from kubeflow.training.constants import constants
from kubeflow.training import TrainingClient
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POD_LOG_STREAMER_LOGLEVEL", "INFO"))


class PodLogStreamer:
    def __init__(self, namespace: str, job_name: str, wait_for_completion_time=10):
        """
        Creates a PodLogStreamer

        params:
            namespace - namespace to watch for events
            job_name  - PytorchJob Name
            wait_for_completion_time - time to wait for the logging to return before killing the subprocess
        """
        self.namespace = namespace
        self.job_name = job_name
        self.training_client = TrainingClient()
        self.wait_for_completion_time = wait_for_completion_time

        def mp_target():
            self.training_client.get_job_logs(
                self.job_name,
                is_master=False,
                container=constants.PYTORCHJOB_CONTAINER,
                follow=True,
            )

        self.process = multiprocessing.Process(
            target=mp_target,
            daemon=False,
        )

    def __enter__(self):
        self.process.start()
        logger.debug(
            f"Started streaming logs for job {self.job_name} in pid {self.process.pid}"
        )

    def __exit__(self, type, value, traceback):
        del type, value, traceback
        if self.wait_for_completion_time:
            self.process.join(self.wait_for_completion_time)
        logger.debug(
            f"Finished streaming logs for job {self.job_name} in pid {self.process.pid}"
        )
        if self.process.is_alive():
            logger.debug(f"Killing process pid {self.process.pid}")
            self.process.kill()
