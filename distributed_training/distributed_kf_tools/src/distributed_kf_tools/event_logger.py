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
This module defines a class that will write events to the log (async) as they occur
when they relate to the specified objects.

Example:

 with EventLogger(
        "kubeflow-user-namespace",
        {
            InvolvedObject(
                kind="PyTorchJob", name="job-name"
            )
        },
    ):
    # Code that needs to run goes here
    # The logger will monitor for events in a seconday thread,
    # events related to PyTorchJob/job-name will be written to the log.

Author: ntl@us.ibm.com
"""
from datetime import datetime
import http.client
import logging
import os
import multiprocessing
from typing import Set, NamedTuple

from kubernetes import client, watch
from kubernetes.client import ApiException, CoreV1Event

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("EVENT_LOGGER_LOGLEVEL", "INFO"))


class InvolvedObject(NamedTuple):
    """
    This class represents the object that events should be logged for.
    The object is assumed to exist within the context of a specific namespace.
    """

    kind: str
    name: str

    def is_for(self, event: CoreV1Event) -> bool:
        return (event.involved_object is not None) and (
            (event.involved_object.kind == self.kind)
            and (event.involved_object.name == self.name)
        )


class EventLogger:
    """
    Output events to the log as info messages.

    Normal events are written at the info level, while other events are written at the
    WARNING/ERROR Level.
    """

    namespace: str
    involved_objects: Set[InvolvedObject]

    def __init__(self, namespace: str, involved_objects: Set[InvolvedObject]):
        """
        Creates an EventLogger

        The event logger will write k8s events for the involved objects to the log.

        params:
            namespace - namespace to watch for events
            involved_objects - set of objects to log related events for
        """
        self.namespace = namespace
        self.involved_objects = involved_objects
        self.stop_monitoring = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=lambda: self._watch_events(), daemon=False
        )

    @classmethod
    def _build_msg(cls, event: CoreV1Event) -> str:
        """
        Formats the message string for logged events
        """
        name = (
            f"{event.involved_object.kind}/{event.involved_object.name}"
            if event.involved_object
            else "?/?"
        )
        ts = (
            event.last_timestamp
            or event.first_timestamp
            or event.event_time
            or datetime.now()
        )

        if event.source:
            source = (
                f"{event.source.component or ''}"
                + f"{':' if event.source.host and event.source.component else ''} {event.source.host or ''}"
            )
        else:
            source = ""

        msg = (
            f"{event.type:10.10s} {ts.isoformat()} {name:30s} {source} {event.message}"
        )
        return msg

    @classmethod
    def _set_log_level(cls, event: CoreV1Event) -> int:
        """Converts a core event to a log level"""
        if not event.type:
            return logging.ERROR
        if event.type == "Normal":
            return logging.INFO
        if event.type == "Error":
            return logging.ERROR
        return logging.WARNING

    def _is_relevant(self, event: CoreV1Event) -> bool:
        """Determines whether a Core Event is relevant to the set of involved objects"""
        return (
            next(filter(lambda io: io.is_for(event), self.involved_objects), None)
            is not None
        )

    def _watch_events(self):
        """Watches for and logs relevant events

        This method runs in a secondary thread until it is signaled that it should stop
        """
        logger.debug(
            f"Watching events for {self.involved_objects} in process {self.process.pid}"
        )
        w = watch.Watch()
        api = client.CoreV1Api()
        resource_version = None
        while not self.stop_monitoring.is_set():
            # Loop until told to stop
            # We wait for events up to 10 seconds, passing the resource version from the last batch of events each time,
            # this ensures that we only receive new events.
            # https://stackoverflow.com/questions/72133783/how-to-avoid-resource-too-old-when-retrying-a-watch-with-kubernetes-python-cli
            try:
                for event in w.stream(
                    api.list_namespaced_event,
                    self.namespace,
                    timeout_seconds=10,
                    resource_version=resource_version,
                ):
                    if not isinstance(event, dict):
                        logger.warn(
                            f"Attempt to process an event that was type {type(event)} when dict was expected. {event}"
                        )
                        continue

                    core_event = event["object"]
                    resource_version = core_event.metadata.resource_version
                    if self._is_relevant(core_event):
                        logger.log(
                            EventLogger._set_log_level(core_event),
                            EventLogger._build_msg(core_event),
                        )

                if self.stop_monitoring.is_set():
                    w.stop()
            except ApiException as e:
                # It's possible that the watched version might not be valid anymore, since watches are only
                # kept around for a short period of time. If that happens, retrieve list of events again.
                # We poll often enough that this probably doesn't happen in practice.
                if e.status == http.client.GONE:
                    logger.debug(
                        f"Watch API return GONE while monitoring events {self.involved_objects}"
                    )
                    resource_version = None
                else:
                    raise
        logger.debug(f"Done monitoring events for {self.involved_objects}")

    def __enter__(self):
        logger.debug(f"Starting event Logger Process")
        self.process.start()
        logger.debug(
            f"Event Logger Process {self.process.pid} has started for objects {self.involved_objects}"
        )

    def __exit__(self, type, value, traceback):
        del type, value, traceback
        self.stop_monitoring.set()
        logger.debug(f"Joining with event Logger Process {self.process.pid}")
        self.process.join(60)
        if self.process.is_alive():
            logger.debug(f"Killing process {self.process.pid}")
            self.process.kill()
