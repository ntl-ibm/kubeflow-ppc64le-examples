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
from dataclasses import dataclass
from datetime import datetime
import http.client
import logging
import os
import multiprocessing
from typing import Set, List, NamedTuple, Optional, Generator
from kubernetes import client, watch
from kubernetes.client import ApiException, CoreV1Event, CoreV1EventList, CoreV1Api

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


@dataclass
class WatchState:
    """This class tracks the current list of events, and which ones have been processed

    It includes changes to watch for new events, and process those new events as they
    happen.
    """

    namespace: str
    resource: CoreV1EventList
    processed_uids: Set[str]

    @property
    def unprocessed(self) -> Generator[CoreV1Event, None, None]:
        if self.resource.items:
            for core_event in self.resource.items:
                if (
                    core_event.metadata
                    and core_event.metadata.uid
                    and core_event.metadata.uid not in self.processed_uids
                ):
                    yield core_event

    def mark_all_events_as_processed(self) -> None:
        for core_event in self.unprocessed:
            self.mark_event_processed(core_event)

    def stream_unprocessed_events(
        self, api: CoreV1Api, timeout: int
    ) -> Generator[CoreV1Event, None, None]:
        """
        Streams unprocessed events, up until the timeout.

        Unprocessed events can come from one of two sources:
        * Events in the event list from the last reload that have not been processed
        * Events that happen after the last reload while we are watching

        When the iteration returns to the generator for the next item, the
        core event is marked as processed.
        """
        for core_event in self.unprocessed:
            yield core_event
            self.mark_event_processed(core_event)

        w = watch.Watch()
        try:
            for event in w.stream(
                api.list_namespaced_event,
                namespace=self.namespace,
                timeout_seconds=timeout,
                resource_version=self.version,
            ):
                if not isinstance(event, dict):
                    logger.warn(
                        f"Attempt to process an event that was type {type(event)} when dict was expected. {event}"
                    )
                    continue

                if event["type"] == "ADDED":
                    core_event = event["object"]
                    if not self.has_processed(core_event):
                        yield core_event
                        self.mark_event_processed(core_event)

        except ApiException as e:
            # It's possible that the watched version might not be valid anymore, since watches are only
            # kept around for a short period of time. If that happens, treat it like a timeout
            # The caller will reload and stream again
            if e.status == http.client.GONE:
                logger.debug(f"Watch API returned GONE while monitoring events")
            else:
                raise

    @property
    def version(self) -> Optional[int]:
        if self.resource.metadata:
            return self.resource.metadata.resource_version
        return None

    @classmethod
    def initialize(cls, namespace: str, api: CoreV1Api) -> "WatchState":
        try:
            resource = api.list_namespaced_event(namespace)
        except ApiException as e:
            logger.warn(f"Error when retrieving namespaced events {e}")
            resource = CoreV1EventList()

        return WatchState(namespace=namespace, resource=resource, processed_uids=set())

    def mark_event_processed(self, core_event: CoreV1Event):
        if core_event.metadata and core_event.metadata.uid:
            self.processed_uids.add(core_event.metadata.uid)
        else:
            pass

    def has_processed(self, core_event: CoreV1Event) -> bool:
        if core_event.metadata and core_event.metadata.uid:
            return core_event.metadata.uid in self.processed_uids
        else:
            return False

    def reload_event_list(self, api: CoreV1Api) -> None:
        try:
            self.resource = api.list_namespaced_event(self.namespace)

            # If events have been deleted, we can remove them from
            # our processed list, since we'll never compare again
            loaded_uids = {
                core_event.metadata.uid
                for core_event in self.resource.items
                if (core_event.metadata and core_event.metadata.uid)
            }
            self.processed_uids = self.processed_uids & loaded_uids

        except ApiException as e:
            logger.warn(f"Error when reloading namespaced events {e}")
            self.resource = CoreV1EventList()


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
        self.is_monitoring = multiprocessing.Event()
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

    def _process_event(self, core_event: CoreV1Event) -> None:
        """Processes an event (if relevant) by logging it to the logger"""
        if self._is_relevant(core_event):
            logger.log(
                EventLogger._set_log_level(core_event),
                EventLogger._build_msg(core_event),
            )

    def _watch_events(self):
        """Watches for and logs relevant events

        This method runs in a secondary thread until it is signaled that it should stop
        """

        logger.debug(
            f"Watching events for {self.involved_objects} in process {self.process.pid}"
        )
        api = client.CoreV1Api()
        state = WatchState.initialize(self.namespace, api)

        # Ignore events from before we started monitoring, these can happen if a pod name is reused
        # Example: Someone terminates a pipeline run and retries. The pod keeps the same name, and
        # the previous events are still around.
        state.mark_all_events_as_processed()
        self.is_monitoring.set()

        while not self.stop_monitoring.is_set():
            # We get a performance benefit from streaming and processing
            # new events as they happen, rather than constantly polling.
            # However watches timeout, and we need to wake up often to
            # see if we should stop. So this logic streams and processes
            # new events, and then after the timeout reloads
            for core_event in state.stream_unprocessed_events(api, timeout=20):
                self._process_event(core_event)
            state.reload_event_list(api)

        logger.debug(f"Done monitoring events for {self.involved_objects}")

    def __enter__(self):
        logger.debug(f"Starting event Logger Process")
        self.process.start()
        self.is_monitoring.wait(60)
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
