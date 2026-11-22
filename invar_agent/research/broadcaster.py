"""Research run manager + SSE event broadcaster (3 channels)."""

from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional

from ..schemas.research import ResearchBlueprint
from .state_machine import ResearchStateMachine, ResearchRunState
from .vault import ResearchVaultManager

CHANNELS = ("search_log", "goi_stream", "accelerator_log")


class ResearchEventBroadcaster:
    """Fan-out SSE events to subscribed asyncio queues."""

    def __init__(self) -> None:
        # Thread-safe sync queues: the research state machine publishes from a
        # worker thread while the SSE endpoint consumes on the event loop.
        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._active_run_uid: Optional[str] = None

    @property
    def seq(self) -> int:
        """Monotonic event counter; subscribers use it to ignore replayed history."""
        return self._seq

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

    def clear_history(self, run_uid: Optional[str] = None) -> None:
        """Clear replayed event history so a new run never leaks prior commit banners."""
        with self._lock:
            self._history.clear()
            if run_uid is not None:
                self._active_run_uid = run_uid

    def publish(self, channel: str, event: str, data: Dict[str, Any],
                run_uid: Optional[str] = None) -> None:
        """Thread-safe fan-out; drops stale events from superseded runs."""
        with self._lock:
            active_uid = self._active_run_uid
            if run_uid is not None and active_uid is not None and run_uid != active_uid:
                return
            self._seq += 1
            seq = self._seq
            eff_uid = run_uid if run_uid is not None else active_uid
        payload = {"channel": channel, "event": event, "seq": seq, **data}
        if eff_uid:
            payload.setdefault("run_uid", eff_uid)
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
        self.active_run_uid: Optional[str] = None
        self.committed_bundles: List[str] = []
        self.pinned_checkpoints: List[str] = []
        self._thread: Optional[threading.Thread] = None
        self._abort = threading.Event()

    async def _progress(self, channel: str, payload: Dict[str, Any],
                        run_uid: Optional[str] = None) -> None:
        event = payload.pop("event", "log")
        if event == "commit" and payload.get("bundle_hash"):
            bh = str(payload.get("bundle_hash"))
            if run_uid is None or run_uid == self.active_run_uid:
                self.committed_bundles.append(bh)
        if event == "checkpoint" and payload.get("bundle_hash"):
            bh = str(payload.get("bundle_hash"))
            if run_uid is None or run_uid == self.active_run_uid:
                self.pinned_checkpoints.append(bh)
        self.broadcaster.publish(channel, event, payload, run_uid=run_uid)

    def request_plan(self, blueprint: ResearchBlueprint) -> None:
        """Stage a blueprint awaiting user approval (approve_plan action)."""
        self.pending_blueprint = blueprint

    def approve(self) -> bool:
        bp = self.pending_blueprint
        if bp is None:
            return False
        vault = ResearchVaultManager(self.workspace_root)
        base_id = vault.run_dir(bp.blueprint_id).name[:60]
        suffix = 0
        while vault.run_dir(bp.blueprint_id).exists():
            suffix += 1
            bp.blueprint_id = f"{base_id}_{suffix}"
        self._abort.clear()
        run_uid = uuid.uuid4().hex[:12]
        self.active_run_uid = run_uid
        self.committed_bundles = []
        self.pinned_checkpoints = []
        self.broadcaster.clear_history(run_uid=run_uid)
        # Guarantee each fresh run starts with zero leftover layers/history from prior runs.
        fresh_bp = bp.model_copy(update={
            "verified_layers": [],
            "evolution_history": [],
            "unified_module_code": None,
        })

        async def _scoped_progress(channel: str, payload: Dict[str, Any]) -> None:
            await self._progress(channel, payload, run_uid=run_uid)

        machine = ResearchStateMachine(
            self.workspace_root / "research", progress_cb=_scoped_progress,
            llm_planner=self.llm_planner, llm_fast=self.llm_fast,
            search_client=self.search_client,
            abort_flag=self._abort.is_set,
            workspace_root=self.workspace_root)
        self.active_state = ResearchRunState(blueprint=fresh_bp)
        seeded = list(getattr(self, "seeded_upload_ids", None) or [])

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._execute(machine, fresh_bp, seeded, run_uid=run_uid))
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    async def _execute(self, machine: ResearchStateMachine, bp: ResearchBlueprint,
                       seeded: Optional[List[str]] = None,
                       run_uid: Optional[str] = None) -> None:
        self.broadcaster.publish("accelerator_log", "run_start",
                                 {"blueprint_id": bp.blueprint_id}, run_uid=run_uid)
        try:
            state = await machine.run(bp)
        except Exception as exc:  # noqa: BLE001
            if run_uid and run_uid != self.active_run_uid:
                return
            if self.active_state is not None:
                self.active_state.status = "FAILED"
                self.active_state.last_error = str(exc)
            self.broadcaster.publish(
                "accelerator_log",
                "run_done",
                {"stage": "Error", "status": "FAILED", "error": str(exc)},
                run_uid=run_uid,
            )
            return
        if run_uid and run_uid != self.active_run_uid:
            return
        self.active_state = state
        # Attached user-uploaded research runs flip from UPLOADED_READY once
        # the verified loop has computationally confirmed the baseline.
        if seeded:
            from .vault import ResearchVaultManager
            vault = ResearchVaultManager(self.workspace_root)
            for rid in seeded:
                if state.status == "VERIFIED":
                    vault.mark_status(rid, "VERIFIED")
                elif state.status == "PARTIAL":
                    vault.mark_status(rid, "PARTIALLY_VERIFIED")
        if state.aborted:
            self.broadcaster.publish("accelerator_log", "aborted", {}, run_uid=run_uid)
        else:
            self.broadcaster.publish("accelerator_log", "run_done",
                                     {"stage": state.stage, "status": state.status,
                                      "snapshot": state.snapshot()}, run_uid=run_uid)

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
