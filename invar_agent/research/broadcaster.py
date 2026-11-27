"""Research run manager + SSE v2.0 event broadcaster with per-run isolation & Total FSM."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..ir.total_fsm import TotalBoundaryAdapter
from ..schemas.research import ResearchBlueprint
from .state_machine import ResearchRunState, ResearchStateMachine
from .vault import ResearchVaultManager

CHANNELS = ("search_log", "goi_stream", "accelerator_log")
TERMINAL_EVENTS = frozenset({"terminal_verdict", "research_completed", "research_failed"})
MAX_TRACKED_RUNS = 64
MAX_EVENTS_PER_RUN = 500


class ResearchEventBroadcaster:
    """Thread-safe SSE v2.0 broadcaster with per-run history isolation and exactly-once terminality."""

    def __init__(self) -> None:
        self._subscribers: List[Tuple["queue.Queue[Dict[str, Any]]", Optional[str]]] = []
        self._histories: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._terminal_emitted: Set[str] = set()
        self._lock = threading.Lock()
        self._seq = 0
        self._active_run_uid: Optional[str] = None

    @property
    def seq(self) -> int:
        """Monotonic event counter."""
        return self._seq

    def has_terminal(self, run_uid: str) -> bool:
        with self._lock:
            return run_uid in self._terminal_emitted

    def subscribe(
        self, since_seq: int = 0, run_uid: Optional[str] = None
    ) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append((q, run_uid))
            target_uid = run_uid if run_uid is not None else self._active_run_uid
            history = list(self._histories.get(target_uid, ())) if target_uid else []
            for payload in history[-200:]:
                if since_seq > 0 and int(payload.get("seq", 0)) <= since_seq:
                    continue
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    break
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            self._subscribers = [(sub_q, uid) for sub_q, uid in self._subscribers if sub_q is not q]

    def clear_history(self, run_uid: Optional[str] = None) -> None:
        """Register a new active run_uid without destroying concurrent runs' histories."""
        with self._lock:
            if run_uid is not None:
                self._active_run_uid = run_uid
                self._histories.setdefault(run_uid, [])
                while len(self._histories) > MAX_TRACKED_RUNS:
                    old_uid, _ = self._histories.popitem(last=False)
                    self._terminal_emitted.discard(old_uid)
            else:
                self._histories.clear()
                self._terminal_emitted.clear()

    def publish(
        self,
        channel: str,
        event: str,
        data: Dict[str, Any],
        run_uid: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe fan-out; preserves concurrent runs and enforces exactly-once terminality."""
        with self._lock:
            eff_uid = run_uid if run_uid is not None else self._active_run_uid
            if event in TERMINAL_EVENTS and eff_uid:
                if eff_uid in self._terminal_emitted:
                    return None
                self._terminal_emitted.add(eff_uid)
            self._seq += 1
            seq = self._seq
            payload: Dict[str, Any] = {
                "channel": channel,
                "event": event,
                "event_version": "2.0",
                "seq": seq,
                **data,
            }
            if eff_uid:
                payload["run_uid"] = eff_uid
                bucket = self._histories.setdefault(eff_uid, [])
                bucket.append(payload)
                if len(bucket) > MAX_EVENTS_PER_RUN:
                    del bucket[: len(bucket) - MAX_EVENTS_PER_RUN]
                while len(self._histories) > MAX_TRACKED_RUNS:
                    old_uid, _ = self._histories.popitem(last=False)
                    self._terminal_emitted.discard(old_uid)
            subs = list(self._subscribers)
        for q, sub_uid in subs:
            if sub_uid is not None and eff_uid != sub_uid:
                continue
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass
        return payload

    def publish_terminal(
        self,
        run_uid: str,
        *,
        status: str,
        epistemic_verdict: str = "INCONCLUSIVE_UNVERIFIED",
        stage: str = "Done",
        snapshot: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        receipt: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit the unique absorbing SSE v2.0 terminal_verdict event for `run_uid`."""
        data: Dict[str, Any] = {
            "stage": stage,
            "status": status,
            "epistemic_verdict": epistemic_verdict,
        }
        if snapshot is not None:
            data["snapshot"] = snapshot
        if error is not None:
            data["error"] = error
        if receipt is not None:
            data["verdict_receipt"] = receipt
        if extra:
            data.update(extra)
        return self.publish("accelerator_log", "terminal_verdict", data, run_uid=run_uid)

    def get_run_history(self, run_uid: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._histories.get(run_uid, ()))


class ResearchRunManager:
    """Owns concurrent research runs via a bounded worker pool with per-run isolation."""

    def __init__(
        self,
        workspace_root,
        broadcaster: ResearchEventBroadcaster,
        llm_planner=None,
        llm_fast=None,
        search_client=None,
        verification_runner=None,
        max_workers: int = 4,
    ):
        self.workspace_root = Path(workspace_root)
        self.broadcaster = broadcaster
        self.llm_planner = llm_planner
        self.llm_fast = llm_fast
        self.search_client = search_client
        self.verification_runner = verification_runner
        self.pending_blueprint: Optional[ResearchBlueprint] = None
        self.active_state: Optional[ResearchRunState] = None
        self.active_run_uid: Optional[str] = None
        self.committed_bundles: List[str] = []
        self.pinned_checkpoints: List[str] = []
        self.runs: Dict[str, ResearchRunState] = {}
        self._futures: Dict[str, Future] = {}
        self._aborts: Dict[str, threading.Event] = {}
        self._abort = threading.Event()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="invar-research",
        )
        self._thread: Optional[threading.Thread] = None

    def allocate_run_uid(self) -> str:
        """Allocate a fresh run_uid BEFORE planning starts (ARCH-006)."""
        uid = uuid.uuid4().hex[:12]
        self.broadcaster.clear_history(run_uid=uid)
        return uid

    async def _progress(
        self, channel: str, payload: Dict[str, Any], run_uid: Optional[str] = None
    ) -> None:
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

    def approve(self, *, run_uid: Optional[str] = None) -> bool:
        with self._lock:
            bp = self.pending_blueprint
            if bp is None:
                return False
            vault = ResearchVaultManager(self.workspace_root)
            base_id = vault.run_dir(bp.blueprint_id).name[:60]
            suffix = 0
            candidate_id = bp.blueprint_id
            while vault.run_dir(candidate_id).exists() or any(
                st.blueprint.blueprint_id == candidate_id for st in self.runs.values()
            ):
                suffix += 1
                candidate_id = f"{base_id}_{suffix}"
            bp = bp.model_copy(update={"blueprint_id": candidate_id})
            uid = run_uid or self.allocate_run_uid()
            self._abort.clear()
            run_abort = threading.Event()
            self._aborts[uid] = run_abort
            self.active_run_uid = uid
            self.committed_bundles = []
            self.pinned_checkpoints = []
            self.broadcaster.clear_history(run_uid=uid)
            fresh_bp = bp.model_copy(
                update={
                    "verified_layers": [],
                    "evolution_history": [],
                    "unified_module_code": None,
                }
            )
            run_scratch = self.workspace_root / "research" / "runs" / uid
            run_scratch.mkdir(parents=True, exist_ok=True)
            run_state = ResearchRunState(blueprint=fresh_bp)
            self.runs[uid] = run_state
            self.active_state = run_state
            seeded = list(getattr(self, "seeded_upload_ids", None) or [])

        async def _scoped_progress(channel: str, payload: Dict[str, Any]) -> None:
            await self._progress(channel, payload, run_uid=uid)

        machine = ResearchStateMachine(
            run_scratch,
            progress_cb=_scoped_progress,
            llm_planner=self.llm_planner,
            llm_fast=self.llm_fast,
            search_client=self.search_client,
            abort_flag=lambda: run_abort.is_set() or self._abort.is_set(),
            workspace_root=self.workspace_root,
            **(
                {"verification_runner": self.verification_runner}
                if self.verification_runner is not None
                else {}
            ),
        )

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self._execute(machine, fresh_bp, seeded, run_uid=uid)
                )
            finally:
                loop.close()

        fut = self._executor.submit(_run)
        with self._lock:
            self._futures[uid] = fut
        return True

    async def _execute(
        self,
        machine: ResearchStateMachine,
        bp: ResearchBlueprint,
        seeded: Optional[List[str]] = None,
        run_uid: Optional[str] = None,
    ) -> None:
        uid = run_uid or self.active_run_uid or uuid.uuid4().hex[:12]
        self.broadcaster.publish(
            "accelerator_log",
            "run_start",
            {"blueprint_id": bp.blueprint_id},
            run_uid=uid,
        )
        try:
            state = await machine.run(bp)
        except Exception as exc:  # noqa: BLE001
            lifted = TotalBoundaryAdapter.lift_sync_error(exc)
            with self._lock:
                st = self.runs.get(uid)
                if st is not None:
                    st.status = "FAILED"
                    st.last_error = str(exc)
                if uid == self.active_run_uid and self.active_state is not None:
                    self.active_state.status = "FAILED"
                    self.active_state.last_error = str(exc)
            self.broadcaster.publish(
                "accelerator_log",
                "run_done",
                {"stage": "Error", "status": "FAILED", "error": str(exc)},
                run_uid=uid,
            )
            self.broadcaster.publish_terminal(
                uid,
                status="FAILED",
                epistemic_verdict=lifted.get("terminal_state", "INCONCLUSIVE_UNVERIFIED"),
                stage="Error",
                error=str(exc),
                receipt=lifted,
            )
            return

        with self._lock:
            self.runs[uid] = state
            if uid == self.active_run_uid:
                self.active_state = state

        if seeded:
            vault = ResearchVaultManager(self.workspace_root)
            for rid in seeded:
                if state.status == "VERIFIED":
                    vault.mark_status(rid, "VERIFIED")
                elif state.status == "PARTIAL":
                    vault.mark_status(rid, "PARTIALLY_VERIFIED")

        if state.aborted:
            self.broadcaster.publish("accelerator_log", "aborted", {}, run_uid=uid)
            self.broadcaster.publish_terminal(
                uid,
                status="ABORTED",
                epistemic_verdict="INCONCLUSIVE_UNVERIFIED",
                stage=state.stage,
                snapshot=state.snapshot(),
            )
        else:
            snap = state.snapshot()
            self.broadcaster.publish(
                "accelerator_log",
                "run_done",
                {"stage": state.stage, "status": state.status, "snapshot": snap},
                run_uid=uid,
            )
            self.broadcaster.publish_terminal(
                uid,
                status=state.status,
                epistemic_verdict=str(
                    getattr(state, "epistemic_verdict", None) or "INCONCLUSIVE_UNVERIFIED"
                ),
                stage=state.stage,
                snapshot=snap,
            )

    def edit(self, blueprint: ResearchBlueprint) -> None:
        self.pending_blueprint = blueprint

    def abort(self, run_uid: Optional[str] = None) -> bool:
        with self._lock:
            if run_uid and run_uid in self._aborts:
                self._aborts[run_uid].set()
                st = self.runs.get(run_uid)
                if st is not None:
                    st.aborted = True
            else:
                self._abort.set()
                for ev in self._aborts.values():
                    ev.set()
                if self.active_state is not None:
                    self.active_state.aborted = True
        return True

    def is_running(self, run_uid: Optional[str] = None) -> bool:
        with self._lock:
            if run_uid:
                fut = self._futures.get(run_uid)
                return fut is not None and not fut.done()
            return any(not fut.done() for fut in self._futures.values())

    def status(self, run_uid: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            bp = self.pending_blueprint
            st = self.runs.get(run_uid) if run_uid else self.active_state
            return {
                "pending": bp.model_dump() if bp else None,
                "pending_feasibility": (
                    bp.feasibility.model_dump() if bp and bp.feasibility else None
                ),
                "running": self.is_running(run_uid),
                "active_run_uid": run_uid or self.active_run_uid,
                "active": st.snapshot() if st else None,
            }
