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
This module defines functions for synchronizing with a running Job and 
the pods that it manages.

Author: ntl@us.ibm.com
"""
import logging
import os
import time
from typing import Optional, Set, Iterable

import http.client
from kubernetes import client
from kubernetes.client import ApiException
from kubernetes.client import V1Pod
from kubeflow.training import TrainingClient
from kubeflow.training.constants import constants

TIMEOUT_ONE_YEAR: int = 60 * 60 * 24 * 365

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SYNCJOB_LOGGER_LOGLEVEL", "INFO"))


def wait_while_pod_is_in_phase(
    name: str,
    namespace: str,
    phases: Optional[Set[str]] = None,
    timeout_seconds=TIMEOUT_ONE_YEAR,
    polling_interval: int = 5,
) -> None:
    """Waits for a Pod to not have a specific Phase.

    name - name of the pod
    namespace - namespace of the pod
    phase - set of phases that the Pod should not be in
    timeout_seconds - max time to wait
    polling_interval - how often to poll

    raises a timeout error if a timeout happens
    """
    core_v1 = client.CoreV1Api()

    start_time = int(time.time())
    while True:
        try:
            pod: V1Pod = core_v1.read_namespaced_pod(name=name, namespace=namespace)  # type: ignore
        except ApiException as e:
            if e.status == http.client.NOT_FOUND:
                logger.info(f"The pod {name} was deleted")
                return
            else:
                raise
        # Phases: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
        if pod.status and (pod.status.phase not in phases):
            return

        if int(time.time()) - start_time > timeout_seconds:
            raise TimeoutError(
                f"The pod {namespace}/{name} did not exit phases {phases}."
            )
        time.sleep(polling_interval)


def wait_while_any_pod_is_in_phase(
    namespace: str,
    pod_names: Iterable[str],
    phases: Set[str],
    timeout_seconds: int = TIMEOUT_ONE_YEAR,
    polling_interval: int = 5,
) -> None:
    """Blocks until all pods are not in the specified phases

    params:
      namespace - namespace of the pods
      pod_names - iterable of pod names
      phases - set of phases the pods should not be in
      timeout_seconds - max time to wait
      polling_interval - how often to poll
    """
    start_time = int(time.time())
    for pod in pod_names:
        wait_while_pod_is_in_phase(
            pod,
            namespace,
            phases,
            timeout_seconds - (int(time.time()) - start_time),
            polling_interval,
        )


def wait_for_job_conditions(
    training_client: TrainingClient,
    pytorch_job_name: str,
    conditions: Set[str],
    timeout: int = TIMEOUT_ONE_YEAR,
    polling_interval: int = 15,
) -> None:
    """
    Waits for the pytorch job to have one of the expected conditions.

    Stops waiting if the job enters a failed state.
    """
    try:
        training_client.wait_for_job_conditions(
            pytorch_job_name,
            expected_conditions=conditions,
            job_kind=constants.PYTORCHJOB_KIND,
            timeout=timeout,
            polling_interval=polling_interval,
        )
    except RuntimeError as e:
        # https://github.com/kubeflow/training-operator/issues/1806#issue-1708084586
        pass
