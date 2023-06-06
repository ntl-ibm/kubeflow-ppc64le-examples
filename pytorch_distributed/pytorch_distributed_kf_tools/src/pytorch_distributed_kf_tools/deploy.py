import os
import subprocess
import shutil
from dataclasses import dataclass

from typing import Optional, List, Dict

from kubernetes.client import (
    V1PodTemplateSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1EnvVar,
    V1ResourceRequirements,
    V1VolumeMount,
    V1Volume,
    V1PersistentVolumeClaimVolumeSource,
    V1EmptyDirVolumeSource,
    V1OwnerReference,
    V1ObjectFieldSelector,
    V1EnvVarSource,
)

from kubeflow.training import (
    V1ReplicaSpec,
    KubeflowOrgV1PyTorchJob,
    KubeflowOrgV1PyTorchJobSpec,
    KubeflowOrgV1ElasticPolicy,
    V1RunPolicy,
    TrainingClient,
)

from kubeflow.training.constants import constants
from kubernetes import client, config, watch

DEFAULT_CONTAINER_ENV = [
    V1EnvVar(name="LOGLEVEL", value="DEBUG"),
    V1EnvVar(name="NCCL_DEBUG", value="DEBUG"),
    V1EnvVar(name="TORCH_CPP_LOG_LEVEL", value="INFO"),
    V1EnvVar(name="C10D_DEBUG_MODE", value="DETAIL"),
    V1EnvVar(name="PET_VERBOSE", value="1"),
]


@dataclass
class OwningWorkFlow:
    name: str
    uid: str


def wait_for_pod_ready(name: str, namespace: str):
    """Waits for a Pod to become ready.
    At that point all containers in the pod have been started
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


def run_pytorch_job(
    namespace: str,
    pytorch_job_name: str,
    shared_pvc_name: str,
    shared_pvc_mount_point: str,
    owning_workflow: Optional[OwningWorkFlow],
    command: List[str],
    num_workers: int,
    gpus_per_worker: int = 1,
    env: Optional[Dict[str, str]] = None,
    working_dir: Optional[str] = None,
    worker_image: str = "quay.io/ntlawrence/pytorchv1.13:1.0",
    shared_pvc_subpath: Optional[str] = None,
    completion_timeout: int = 60 * 60 * 24,
):
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

    container_env = (
        [V1EnvVar(n, v) for n, v in env.items()] if env else DEFAULT_CONTAINER_ENV
    )
    resources = (
        V1ResourceRequirements(limits={"nvidia.com/gpu": f"{gpus_per_worker}"})
        if gpus_per_worker
        else None
    )

    container_working_dir = working_dir if working_dir else shared_pvc_mount_point

    # Pod definition for each worker replica
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
                    image_pull_policy="IfNotPresent",
                    working_dir=container_working_dir,
                    command=command,
                    env=container_env,
                    resources=resources,
                    volume_mounts=[
                        V1VolumeMount(
                            mount_path=shared_pvc_mount_point,
                            name="shared",
                            sub_path=shared_pvc_subpath,
                        ),
                        # PyTorch requires shared memory on each pod
                        V1VolumeMount(mount_path="/dev/shm", name="dshm"),
                    ],
                )
            ],
            volumes=[
                V1Volume(
                    name="shared",
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=shared_pvc_name
                    ),
                ),
                V1Volume(
                    name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory")
                ),
            ],
        ),
    )

    # The PyTorchJob supports the concept of a master replica, but not
    # specified the first worker takes that role. Having only workers
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
            # running on on-premise. Here we default to a fixed size of the
            # number of replicias.
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

    ##########################
    # Submit training job
    ##########################
    print(repr(pytorchjob))
    training_client = TrainingClient()
    training_client.create_pytorchjob(pytorchjob)

    try:
        training_client.wait_for_job_conditions(
            pytorch_job_name,
            expected_conditions={
                constants.JOB_CONDITION_RUNNING,
                constants.JOB_CONDITION_SUCCEEDED,
                constants.JOB_CONDITION_FAILED,
            },
            job_kind=constants.PYTORCHJOB_KIND,
            timeout=600,
        )
    except RuntimeError as e:
        # https://github.com/kubeflow/training-operator/issues/1806#issue-1708084586
        pass

    # Wait for pods to be ready, must do this before reading logs
    pod_names = training_client.get_job_pod_names(name=pytorch_job_name, is_master=None)
    for pod in pod_names:
        wait_for_pod_ready(pod, namespace)

    # stream logs for all workers
    # (most of the interesting stuff is in worker 0)
    training_client.get_job_logs(
        name=pytorch_job_name,
        is_master=False,
        container=constants.PYTORCHJOB_CONTAINER,
        follow=True,
    )

    # No more logs means workers have finished, wait for the rest of the job
    try:
        training_client.wait_for_job_conditions(
            pytorch_job_name,
            expected_conditions={
                constants.JOB_CONDITION_SUCCEEDED,
                constants.JOB_CONDITION_FAILED,
            },
            timeout=completion_timeout,
            job_kind=constants.PYTORCHJOB_KIND,
        )
    except RuntimeError as e:
        # https://github.com/kubeflow/training-operator/issues/1806#issue-1708084586
        pass

    if training_client.is_job_failed(
        name=pytorch_job_name, job_kind=constants.PYTORCHJOB_KIND
    ):
        raise RuntimeError(f"Job {pytorch_job_name} Failed!")
