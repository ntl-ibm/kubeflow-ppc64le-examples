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

"""Functions to build job templates

   Author: ntl@us.ibm.com
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import uuid

from kubeflow.training import (
    KubeflowOrgV1ElasticPolicy,
    KubeflowOrgV1PyTorchJob,
    KubeflowOrgV1PyTorchJobSpec,
    V1ReplicaSpec,
    V1RunPolicy,
)
from kubeflow.training.constants import constants
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
import distributed_kf_tools.env_var as env_var


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


def build_pytorch_job_template(
    namespace: str,
    pytorch_job_name: str,
    pvcs: List[PvcMount],
    owning_workflow: Optional[OwningWorkFlow],
    command: List[str],
    num_workers: int,
    worker_image: str,
    gpus_per_worker: int,
    env: Optional[Dict[str, str]],
    working_dir: Optional[str],
    image_pull_policy: Optional[str],
    node_selector: Optional[Dict[str, str]],
) -> KubeflowOrgV1PyTorchJob:
    """Builds a PyTorchJob template
    Params:
    namespace - the namespace that the job is to run in. Within a kubeflow component, you can
                code the literal.
    pytorch_job_name  - the unique name for the job.
    pvcs      - list of PVCs to mount to each worker node. Most distributed training applications
                will require at least one PVC to share information between pods. PVCs should be created
                with read-write-many capabilities.
    owning_workflow - The created job will be garbage collected with the owning workflow is destroyed.
    command - list of command & arguments. Typically this will be something like:
              command=[
                "python",
                "-m",
                "torch.distributed.run",
                <<< your script + args here >>>
               ]
    num_workers - the number of workers (pods) to use in the training
    worker_image - the container image for the workers to use
    gpus_per_worker - gpus to assign to each worker
    env - the environment variables to pass to each worker (optional)
    working_dir - working directory for each worker (optional)
    image_pull_pollicy - when to pull a new container image (optional)
    node_selector - node selector for each worker (optional)
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
        else [
            env_var.LOGLEVEL_INFO,
        ]
    )

    # Construct resources parameter
    resources = (
        V1ResourceRequirements(limits={"nvidia.com/gpu": f"{gpus_per_worker}"})
        if gpus_per_worker
        else None
    )

    container_working_dir = working_dir if working_dir else None

    # Construct the volumes and volume mount parameters
    # The same volume can be mounted at different locations (possibly using a subpath),
    # and so it is legal to have the same volume twice in the volume_mounts.
    # It is not OK to have the same volume twice in the volumes list, and the error
    # given if you do that is not clear. The PvcMount class is just a mapping of
    # pvc name -> mount point, so the invoker is free of this problem as long as
    # we make sure volumes appear only once in the volume list.
    volume_mounts: List[V1VolumeMount] = []
    volumes: Dict[str, V1Volume] = {}
    for pvc in pvcs:
        volume_mounts.append(
            V1VolumeMount(
                mount_path=pvc.mount_path, name=pvc.pvc_name, sub_path=pvc.subpath
            )
        )
        if pvc.pvc_name not in volumes:
            volumes[pvc.pvc_name] = V1Volume(
                name=pvc.pvc_name,
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc.pvc_name
                ),
            )

    # PyTorch requires shared memory on each pod
    if not "/dev/shm" in {p.mount_path for p in pvcs}:
        volume_mounts.append(V1VolumeMount(mount_path="/dev/shm", name="dshm"))
        volumes["dshm"] = V1Volume(
            name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory")
        )

    # Pod template for each worker replica
    # Defines the container image, command, and volume mounts
    pod_template = V1PodTemplateSpec(
        metadata=V1ObjectMeta(
            # https://github.com/kubeflow/website/issues/2011
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1PodSpec(
            node_selector=node_selector,
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
            volumes=list(volumes.values()),
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
            name=pytorch_job_name,
            namespace=namespace,
            owner_references=workflow_ownership,
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
                rdzv_id=str(uuid.uuid4()),
                n_proc_per_node=gpus_per_worker,
                min_replicas=num_workers,
                max_replicas=num_workers,
                max_restarts=0,
            ),
            run_policy=V1RunPolicy(clean_pod_policy="None"),
            pytorch_replica_specs={"Worker": worker_spec},
        ),
    )

    return pytorchjob
