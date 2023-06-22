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
This module contains tooling to deply a pytorch job on kubernetes from
within a Kubeflow pipeline.

Author: ntl@us.ibm.com
"""
import os
import logging
from typing import Dict, List, Optional
import signal
import sys
import time
import yaml

from kubeflow.training import TrainingClient
from kubeflow.training import KubeflowOrgV1PyTorchJob
from kubeflow.training.constants import constants
from kubernetes import config

import distributed_kf_tools.template as template
import distributed_kf_tools.syncjob as syncjob
import distributed_kf_tools.event_logger as event_logger
import distributed_kf_tools.pod_log_streamer as pod_log_streamer

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))


def _execute_job(
    pytorchjob_template: KubeflowOrgV1PyTorchJob, completion_timeout: int
) -> None:
    """
    Executes the job template and waits for completion
    """
    start_time = int(time.time())

    def remaining_time() -> int:
        return completion_timeout - (start_time - int(time.time()))

    assert pytorchjob_template.metadata is not None

    namespace = pytorchjob_template.metadata.namespace
    pytorch_job_name = pytorchjob_template.metadata.name

    # Within the scope of this with, all events targeting the pytorch job will appear in the log
    with event_logger.EventLogger(
        namespace,
        {event_logger.InvolvedObject(constants.PYTORCHJOB_KIND, name=pytorch_job_name)},
    ):
        training_client = TrainingClient()
        training_client.create_pytorchjob(pytorchjob_template)

        syncjob.wait_for_job_conditions(
            training_client,
            pytorch_job_name,
            {
                constants.JOB_CONDITION_RUNNING,
                constants.JOB_CONDITION_SUCCEEDED,
                constants.JOB_CONDITION_FAILED,
            },
        )

        pod_names = training_client.get_job_pod_names(
            name=pytorch_job_name, is_master=False
        )

        # Within the scope of this with, all events targeting pods for the pytorch job
        # will appear in the log
        with event_logger.EventLogger(
            namespace,
            {event_logger.InvolvedObject(kind="Pod", name=name) for name in pod_names},
        ):
            # Wait for pods to be Running (or succeeded/failed), must do this before reading logs
            # https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
            syncjob.wait_while_any_pod_is_in_phase(
                namespace,
                pod_names,
                {"Pending"},
                remaining_time(),
            )

            # stream logs for all workers (The interesting stuff is usually in worker 0)
            with pod_log_streamer.PodLogStreamer(namespace, pytorch_job_name):
                # Monitor pods until not running
                syncjob.wait_while_any_pod_is_in_phase(
                    namespace,
                    pod_names,
                    {"Running"},
                    remaining_time(),
                )

                # Job might still have a little cleanup, even after all the pods end
                syncjob.wait_for_job_conditions(
                    training_client,
                    pytorch_job_name,
                    {
                        constants.JOB_CONDITION_SUCCEEDED,
                        constants.JOB_CONDITION_FAILED,
                    },
                    remaining_time(),
                    polling_interval=20,
                )

    # Check for success or failure
    if training_client.is_job_failed(
        name=pytorch_job_name, job_kind=constants.PYTORCHJOB_KIND
    ):
        raise RuntimeError(f"Job {pytorch_job_name} Failed!")


def _delete_pytorch_job(pytorchjob_template: KubeflowOrgV1PyTorchJob) -> None:
    """
    Deletes the pytorch job
    """
    assert pytorchjob_template.metadata is not None
    namespace = pytorchjob_template.metadata.namespace
    name = pytorchjob_template.metadata.name

    try:
        training_client = TrainingClient()
        logger.info(f"Deleting pytorch job {name}")
        training_client.delete_pytorchjob(name=name, namespace=namespace)
    except RuntimeError as e:
        logger.exception(e)


def _execute_pytorch_job_and_delete(
    pytorchjob_template: KubeflowOrgV1PyTorchJob, completion_timeout: int
) -> None:
    """Executes the pytorch job and deletes on completion.

    Also responds to signal SIGTERM and deletes the job
    """
    pid = os.getpid()

    def hndlr(signal, stackframe) -> None:
        # Don't try to delete from a running subprocess (which will also get a SIGTERM signal)
        if os.getpid() == pid:
            logger.error(
                f"SIGTERM received in pid {os.getpid} , deleting the pytorch job"
            )
            _delete_pytorch_job(pytorchjob_template)
            sys.exit(143)
        else:
            sys.exit(143)

    signal.signal(signal.SIGTERM, hndlr)
    try:
        _execute_job(pytorchjob_template, completion_timeout)
    finally:
        _delete_pytorch_job(pytorchjob_template)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)


def run_pytorch_job(
    namespace: str,
    pytorch_job_name: str,
    pvcs: List[template.PvcMount],
    owning_workflow: Optional[template.OwningWorkFlow],
    command: List[str],
    num_workers: int,
    worker_image: str,
    gpus_per_worker: int = 1,
    env: Optional[Dict[str, str]] = None,
    working_dir: Optional[str] = None,
    image_pull_policy: str = "IfNotPresent",
    completion_timeout: int = syncjob.TIMEOUT_ONE_YEAR,
    log_pytorch_job_template: bool = True,
    load_in_cluster_config=True,
) -> None:
    """
    Builds a kubernetes PytorchJob template, creates the job, and waits for completion.
    RuntimeError is raised if the job fails.

    The version of the function does not support:
    * Adding or removing workers mid training
    * Restarting the training if a worker fails.

    Params:
    namespace - the namespace that the job is to run in. Within a kubeflow component, you can
                code the literal  '{{workflow.namespace}}', which will be replaced with the workflow's
                namespace when the pipeline is executed.
    pytorch_job_name  - the unique name for the job. Within a Kubeflow component, you can code
                        the literal {{workflow.name}} to use the workflow's name as the job name.
    pvcs      - list of PVCs to mount to each worker node. Most distributed training applications
                will require at least one PVC to share information between pods. PVCs should be created
                with read-write-many capabilities.
    owning_workflow - The created job will be garbage collected with the owning workflow is destroyed.
                Within a kubeflow component, you can create this value like this:
                owning_workflow=OwningWorkFlow(name="{{workflow.name}}", uid="{{workflow.uid}}")
                The workflow's name and uid will be set by the pipeline when the pipeline runs.
    command - list of command & arguments. Typically this will be something like:
              command=[
                "python",
                "-m",
                "torch.distributed.run",
                <<< your script + args here >>>
               ]
    num_workers - the number of workers (pods) to use in the training
    worker_image - the container image for the workers to use
    gpus_per_worker - gpus to assign to each worker (default is 1)
    env - the environment variables to pass to each worker (optional)
    working_dir - working directory for each worker (optional)
    image_pull_pollicy - when to pull a new container image (optional, default is IfNotPresent)
    completion_timeout - how long to wait for the training to complete (optional, default is one year)
    load_in_cluster_config - load the kubernetes configuration from within the cluster, if this is false,
                             you will need to initialize the config before calling the method.
    """

    if load_in_cluster_config:
        config.load_incluster_config()

    assert namespace
    assert pytorch_job_name

    pytorchjob_template = template.build_pytorch_job_template(
        namespace=namespace,
        pytorch_job_name=pytorch_job_name,
        pvcs=pvcs,
        owning_workflow=owning_workflow,
        command=command,
        num_workers=num_workers,
        worker_image=worker_image,
        gpus_per_worker=gpus_per_worker,
        env=env,
        working_dir=working_dir,
        image_pull_policy=image_pull_policy,
    )

    if log_pytorch_job_template:
        logger.debug(yaml.dump(pytorchjob_template.to_dict()))

    _execute_pytorch_job_and_delete(pytorchjob_template, completion_timeout)
