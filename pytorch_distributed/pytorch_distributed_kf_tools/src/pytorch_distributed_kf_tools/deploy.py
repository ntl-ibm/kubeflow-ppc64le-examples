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
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from kubeflow.training import (
    KubeflowOrgV1ElasticPolicy,
    KubeflowOrgV1PyTorchJob,
    KubeflowOrgV1PyTorchJobSpec,
    TrainingClient,
    V1ReplicaSpec,
    V1RunPolicy,
)
from kubeflow.training.constants import constants
from kubernetes import client, config, watch
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1ObjectMeta,
    V1OwnerReference,
    V1PersistentVolumeClaimVolumeSource,
    V1PodSpec,
    V1PodTemplateSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)


@dataclass(frozen=True)
class TimeOutConstants:
    """Utility values for common time out values"""

    ONE_YEAR: int = 60 * 60 * 24 * 365


@dataclass(frozen=True)
class ContainerEnv:
    """
    Environment variables that can be added to pytorch workers to provide enhanced
    debug capabilities.
    """

    LOGLEVEL_INFO: V1EnvVar = V1EnvVar(name="LOGLEVEL", value="INFO")
    NCCL_INFO: V1EnvVar = V1EnvVar(name="NCCL_DEBUG", value="INFO")
    C10D_DEBUG_MODE_DETAIL: V1EnvVar = V1EnvVar(name="C10D_DEBUG_MODE", value="DETAIL")
    DEFAULT_CONTAINER_ENV: List[V1EnvVar] = field(
        default_factory=lambda: [ContainerEnv.LOGLEVEL_INFO, ContainerEnv.NCCL_INFO]
    )


@dataclass
class OwningWorkFlow:
    """
    This class represents the owning workflow (kubeflow pipeline) for a
    pytorch job. "Owning" is a kubernetes concept that means when the
    owning object is deleted, the owned objects are also cleaned up.
    """

    name: str
    uid: str


@dataclass
class PvcMount:
    """
    This class represents a request to mount a volume to the worker pods referencing
    a pvc claim name.

    pvc_name - the claim associated with the pvc
    mount_path - the path to the mounted volume within the container
    subpath - a path within the pvc to mount at the mount path.
              This is optional and allows a subset of the pvc to be mounted.
    """

    pvc_name: str
    mount_path: str
    subpath: Optional[str] = None


def _wait_for_pod_ready(name: str, namespace: str) -> None:
    """Waits for a Pod to become ready.
    At that point all containers in the pod have been started

    name - name of the pod
    namespace - namespace of the pod
    """
    config.load_incluster_config()
    w = watch.Watch()
    core_v1 = client.CoreV1Api()

    # Watching a specific pod is done with a field selector on the name.
    # https://github.com/kubernetes-client/python/issues/467
    for event in w.stream(
        func=core_v1.list_namespaced_pod,
        namespace=namespace,
        field_selector=f"metadata.name={name}",
        timeout_seconds=120,
    ):
        # Phases: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase
        if event["object"].status.phase not in {"Pending"}:
            w.stop()
            return
        # event.type: ADDED, MODIFIED, DELETED
        if event["type"] == "DELETED":
            print(f" {name} deleted before it started")
            w.stop()
            return


def _wait_for_job_conditions(
    training_client: TrainingClient,
    pytorch_job_name: str,
    conditions: Set[str],
    timeout: int = TimeOutConstants.ONE_YEAR,
) -> None:
    """
    Waits for the pytorch job to have one of the expected conditions.

    Stops waiting if the pod enters a failed state.
    """
    try:
        training_client.wait_for_job_conditions(
            pytorch_job_name,
            expected_conditions=conditions,
            job_kind=constants.PYTORCHJOB_KIND,
            timeout=timeout,
        )
    except RuntimeError as e:
        # https://github.com/kubeflow/training-operator/issues/1806#issue-1708084586
        pass


def run_pytorch_job(
    namespace: str,
    pytorch_job_name: str,
    pvcs: List[PvcMount],
    owning_workflow: Optional[OwningWorkFlow],
    command: List[str],
    num_workers: int,
    worker_image: str,
    gpus_per_worker: int = 1,
    env: Optional[Dict[str, str]] = None,
    working_dir: Optional[str] = None,
    image_pull_policy: str = "IfNotPresent",
    completion_timeout: int = TimeOutConstants.ONE_YEAR,
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
    """

    # An owner reference for the workflow
    # When the workflow is deleted, the torch job is
    # garbage collected.
    workflow_ownership = None
    if owning_workflow:
        workflow_ownership = [
            V1OwnerReference(
                api_version="v1",
                kind="Workflow",
                name=owning_workflow.name,
                uid=owning_workflow.uid,
            )
        ]

    # Construct Environment parameters
    container_env = (
        [V1EnvVar(n, v) for n, v in env.items()]
        if env
        else ContainerEnv.DEFAULT_CONTAINER_ENV
    )

    # Construct resources parameter
    resources = (
        V1ResourceRequirements(limits={"nvidia.com/gpu": f"{gpus_per_worker}"})
        if gpus_per_worker
        else None
    )

    container_working_dir = working_dir if working_dir else None

    # Construct the volumes and volume mount parameters
    volume_mounts: List[V1VolumeMount] = []
    volumes: List[V1Volume] = []
    for i, pvc in enumerate(pvcs):
        name = f"{pvc.pvc_name}-{i}"
        volume_mounts.append(
            V1VolumeMount(mount_path=pvc.mount_path, name=name, sub_path=pvc.subpath)
        )
        volumes.append(
            V1Volume(
                name=name,
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc.pvc_name
                ),
            )
        )

    # PyTorch requires shared memory on each pod
    if not "/dev/shm" in {p.mount_path for p in pvcs}:
        volume_mounts.append(V1VolumeMount(mount_path="/dev/shm", name="dshm")),
        volumes.append(
            V1Volume(name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory"))
        )

    # Pod template for each worker replica
    # Defines the container image, command, and volume mounts
    pod_template = V1PodTemplateSpec(
        metadata=V1ObjectMeta(
            name=pytorch_job_name,
            namespace=namespace,
            owner_references=workflow_ownership,
            # https://github.com/kubeflow/website/issues/2011
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1PodSpec(
            containers=[
                V1Container(
                    name=constants.PYTORCHJOB_CONTAINER,
                    image=worker_image,
                    image_pull_policy=image_pull_policy,
                    working_dir=container_working_dir,
                    command=command,
                    env=container_env,
                    resources=resources,
                    volume_mounts=volume_mounts,
                )
            ],
            volumes=volumes,
        ),
    )

    # The PyTorchJob supports the concept of a master replica. When not
    # included, the first worker takes the master role. Having only workers
    # simplifies the template a bit. This is how the imagenet example is
    # setup. https://github.com/kubeflow/training-operator/blob/master/examples/pytorch/elastic/imagenet/
    worker_spec = V1ReplicaSpec(
        replicas=num_workers, restart_policy="Never", template=pod_template
    )

    # The owner references and replica spec are used to define the torch job
    pytorchjob = KubeflowOrgV1PyTorchJob(
        api_version=f"{constants.KUBEFLOW_GROUP}/{constants.OPERATOR_VERSION}",
        kind=constants.PYTORCHJOB_KIND,
        metadata=V1ObjectMeta(
            name=pytorch_job_name, owner_references=workflow_ownership
        ),
        spec=KubeflowOrgV1PyTorchJobSpec(
            # c10d is the most commonly used because it doesn't require additional
            # packages. The primary advantage that elastic solutions offer is the
            # ability to use cheap hardware in the cloud that can be taken away
            # at any time. That doesn't apply so much for Power servers that are
            # running on on-prem. Here we default to a fixed size of the
            # number of replicias.
            # We also set this template up to fail if a failure happens, in the
            # future we might support restarts with checkpointing. But for now,
            # this example needs to be a simple pass/fail type of thing.
            elastic_policy=KubeflowOrgV1ElasticPolicy(
                rdzv_backend="c10d",
                rdzv_id=pytorch_job_name,
                n_proc_per_node=gpus_per_worker,
                min_replicas=num_workers,
                max_replicas=num_workers,
                max_restarts=0,
            ),
            run_policy=V1RunPolicy(clean_pod_policy="None"),
            pytorch_replica_specs={"Worker": worker_spec},
        ),
    )

    # Submit training job
    training_client = TrainingClient()
    training_client.create_pytorchjob(pytorchjob)

    _wait_for_job_conditions(
        training_client,
        {
            constants.JOB_CONDITION_RUNNING,
            constants.JOB_CONDITION_SUCCEEDED,
            constants.JOB_CONDITION_FAILED,
        },
    )

    # Wait for pods to be ready (or succeeded/failed), must do this before reading logs
    pod_names = training_client.get_job_pod_names(name=pytorch_job_name, is_master=None)
    for pod in pod_names:
        _wait_for_pod_ready(pod, namespace)

    # stream logs for all workers (The interesting stuff is usually in worker 0)
    # I have seen cases where progress bars cause with log streaming at the
    # k8s client layer. I recommend turning those off if possible.
    training_client.get_job_logs(
        name=pytorch_job_name,
        is_master=False,
        container=constants.PYTORCHJOB_CONTAINER,
        follow=True,
    )

    # No more logs means workers have finished, wait for the rest of the job
    _wait_for_job_conditions(
        training_client,
        {
            constants.JOB_CONDITION_SUCCEEDED,
            constants.JOB_CONDITION_FAILED,
        },
        completion_timeout,
    )

    # Check for success or failure
    if training_client.is_job_failed(
        name=pytorch_job_name, job_kind=constants.PYTORCHJOB_KIND
    ):
        raise RuntimeError(f"Job {pytorch_job_name} Failed!")
