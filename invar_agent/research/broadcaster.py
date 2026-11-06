"""Research run manager + SSE event broadcaster (3 channels)."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Dict, List, Optional

from ..schemas.research import ResearchBlueprint
from .state_machine import ResearchStateMachine, ResearchRunState

CHANNELS = ("search_log", "goi_stream", "accelerator_log")


class ResearchEventBroadcaster:
    """Fan-out SSE events to subscribed asyncio queues."""

    def __init__(self) -> None:
        # Thread-safe sync queues: the research state machine publishes from a
        # worker thread while the SSE endpoint consumes on the event loop.
        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
            # Replay recent events so late subscribers (UI reconnects, tests)
            # do not miss the start of a run.
            for payload in self._history[-200:]:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    break
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, channel: str, event: str, data: Dict[str, Any]) -> None:
        """Thread-safe fan-out; safe to call from any thread/loop."""
        payload = {"channel": channel, "event": event, **data}
        with self._lock:
            self._history.append(payload)
            if len(self._history) > 500:
                del self._history[: len(self._history) - 500]
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


class ResearchRunManager:
    """Owns the active research run; supports approve/edit/abort."""

    def __init__(self, workspace_root, broadcaster: ResearchEventBroadcaster,
                 llm_planner=None, llm_fast=None, search_client=None):
        self.workspace_root = workspace_root
        self.broadcaster = broadcaster
        self.llm_planner = llm_planner
        self.llm_fast = llm_fast
        self.search_client = search_client
        self.pending_blueprint: Optional[ResearchBlueprint] = None
        self.active_state: Optional[ResearchRunState] = None
        self._thread: Optional[threading.Thread] = None
        self._abort = threading.Event()

    async def _progress(self, channel: str, payload: Dict[str, Any]) -> None:
        event = payload.pop("event", "log")
        self.broadcaster.publish(channel, event, payload)

    def request_plan(self, blueprint: ResearchBlueprint) -> None:
        """Stage a blueprint awaiting user approval (approve_plan action)."""
        self.pending_blueprint = blueprint

    def approve(self) -> bool:
        bp = self.pending_blueprint
        if bp is None:
            return False
        self._abort.clear()
        machine = ResearchStateMachine(
            self.workspace_root / "research", progress_cb=self._progress,
            llm_planner=self.llm_planner, llm_fast=self.llm_fast,
            search_client=self.search_client,
            abort_flag=self._abort.is_set,
            workspace_root=self.workspace_root)
        self.active_state = ResearchRunState(blueprint=bp)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._execute(machine, bp))
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    async def _execute(self, machine: ResearchStateMachine, bp: ResearchBlueprint) -> None:
        self.broadcaster.publish("accelerator_log", "run_start",
                                 {"blueprint_id": bp.blueprint_id})
        state = await machine.run(bp)
        self.active_state = state
        if state.aborted:
            self.broadcaster.publish("accelerator_log", "aborted", {})
        else:
            self.broadcaster.publish("accelerator_log", "run_done",
                                     {"stage": state.stage, "status": state.status,
                                      "snapshot": state.snapshot()})

    def edit(self, blueprint: ResearchBlueprint) -> None:
        self.pending_blueprint = blueprint

    def abort(self) -> bool:
        self._abort.set()
        if self.active_state is not None:
            self.active_state.aborted = True
        return True

    def status(self) -> Dict[str, Any]:
        bp = self.pending_blueprint
        return {
            "pending": bp.model_dump() if bp else None,
            "pending_feasibility": (bp.feasibility.model_dump()
                                    if bp and bp.feasibility else None),
            "running": self._thread is not None and self._thread.is_alive(),
            "active": self.active_state.snapshot() if self.active_state else None,
        }
