"""Per-call LLM telemetry accounting, structured logging, and per-run budget summaries."""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_CURRENT_PHASE: ContextVar[str] = ContextVar("invar_llm_phase", default="unspecified")
_CURRENT_ATTEMPT: ContextVar[int] = ContextVar("invar_llm_attempt", default=1)
_CURRENT_TIER: ContextVar[str] = ContextVar("invar_llm_tier", default="")


@dataclass
class LLMCallRecord:
    phase: str
    tier: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0
    cache_hit_tokens: int = 0
    latency_s: float = 0.0
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMTelemetrySession:
    run_id: str = "default"
    calls: List[LLMCallRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, rec: LLMCallRecord) -> None:
        with self.lock:
            self.calls.append(rec)

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            snapshot = list(self.calls)
        prompt_total = sum(max(0, c.prompt_tokens) for c in snapshot)
        completion_total = sum(max(0, c.completion_tokens) for c in snapshot)
        reasoning_total = sum(max(0, c.reasoning_tokens) for c in snapshot)
        cache_hit_total = sum(max(0, c.cache_hit_tokens) for c in snapshot)
        latency_total = round(sum(max(0.0, c.latency_s) for c in snapshot), 3)
        by_phase: Dict[str, Dict[str, Any]] = {}
        by_tier: Dict[str, Dict[str, Any]] = {}
        for c in snapshot:
            for bucket, key in ((by_phase, c.phase or "unspecified"), (by_tier, c.tier or "unspecified")):
                entry = bucket.setdefault(
                    key,
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "cache_hit_tokens": 0,
                        "latency_s": 0.0,
                    },
                )
                entry["calls"] += 1
                entry["prompt_tokens"] += max(0, c.prompt_tokens)
                entry["completion_tokens"] += max(0, c.completion_tokens)
                entry["reasoning_tokens"] += max(0, c.reasoning_tokens)
                entry["cache_hit_tokens"] += max(0, c.cache_hit_tokens)
                entry["latency_s"] = round(entry["latency_s"] + max(0.0, c.latency_s), 3)
        return {
            "run_id": self.run_id,
            "total_calls": len(snapshot),
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "reasoning_tokens": reasoning_total,
            "cache_hit_tokens": cache_hit_total,
            "total_tokens": prompt_total + completion_total,
            "total_latency_s": latency_total,
            "by_phase": by_phase,
            "by_tier": by_tier,
            "calls": [c.to_dict() for c in snapshot],
        }


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_SESSION = LLMTelemetrySession(run_id="global")
_ACTIVE_SESSION: ContextVar[Optional[LLMTelemetrySession]] = ContextVar("invar_llm_session", default=None)
_LISTENERS: List[Callable[[LLMCallRecord], None]] = []


@contextmanager
def llm_call_context(
    phase: Optional[str] = None,
    *,
    tier: Optional[str] = None,
    attempt: Optional[int] = None,
):
    """Tag all nested LLM calls with `(phase, tier, attempt)` metadata."""
    tok_p = _CURRENT_PHASE.set(phase) if phase is not None else None
    tok_t = _CURRENT_TIER.set(tier) if tier is not None else None
    tok_a = _CURRENT_ATTEMPT.set(int(attempt)) if attempt is not None else None
    try:
        yield
    finally:
        if tok_p is not None:
            _CURRENT_PHASE.reset(tok_p)
        if tok_t is not None:
            _CURRENT_TIER.reset(tok_t)
        if tok_a is not None:
            _CURRENT_ATTEMPT.reset(tok_a)


@dataclass
class RunContext:
    """ARCH-010: Run-scoped execution context propagated via contextvars."""
    run_uid: str
    telemetry_session: LLMTelemetrySession
    literature: List[Dict[str, Any]] = field(default_factory=list)
    deadline: Optional[float] = None


_ACTIVE_RUN_CONTEXT: ContextVar[Optional[RunContext]] = ContextVar(
    "invar_active_run_context", default=None
)


def get_active_run_context() -> Optional[RunContext]:
    return _ACTIVE_RUN_CONTEXT.get()


@contextmanager
def run_context_scope(
    run_uid: str,
    *,
    literature: Optional[List[Dict[str, Any]]] = None,
    deadline: Optional[float] = None,
):
    """Create a run-scoped RunContext with an isolated LLMTelemetrySession."""
    with llm_telemetry_scope(run_id=run_uid) as session:
        ctx = RunContext(
            run_uid=run_uid,
            telemetry_session=session,
            literature=list(literature or []),
            deadline=deadline,
        )
        token = _ACTIVE_RUN_CONTEXT.set(ctx)
        try:
            yield ctx
        finally:
            _ACTIVE_RUN_CONTEXT.reset(token)


@contextmanager
def llm_telemetry_scope(run_id: str = "run"):
    """Create an isolated per-run telemetry accumulator (while also updating global)."""
    session = LLMTelemetrySession(run_id=run_id)
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)


def current_llm_context() -> Dict[str, Any]:
    return {
        "phase": _CURRENT_PHASE.get(),
        "tier": _CURRENT_TIER.get(),
        "attempt": _CURRENT_ATTEMPT.get(),
    }


def extract_usage_tokens(usage: Any) -> Dict[str, int]:
    """Extract prompt, completion, reasoning, and prefix-cache-hit tokens across OpenAI/DeepSeek formats."""
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_tokens": 0,
        }

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    prompt_tokens = int(_get(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get(usage, "completion_tokens", 0) or 0)

    reasoning_tokens = int(_get(usage, "reasoning_tokens", 0) or 0)
    if reasoning_tokens == 0:
        comp_details = _get(usage, "completion_tokens_details")
        reasoning_tokens = int(_get(comp_details, "reasoning_tokens", 0) or 0)

    cache_hit_tokens = int(_get(usage, "prompt_cache_hit_tokens", 0) or 0)
    if cache_hit_tokens == 0:
        cache_hit_tokens = int(_get(usage, "cache_hit_tokens", 0) or 0)
    if cache_hit_tokens == 0:
        prompt_details = _get(usage, "prompt_tokens_details")
        cache_hit_tokens = int(_get(prompt_details, "cached_tokens", 0) or 0)

    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "reasoning_tokens": max(0, reasoning_tokens),
        "cache_hit_tokens": max(0, cache_hit_tokens),
    }


def record_llm_call(
    *,
    model: str,
    tier: Optional[str] = None,
    phase: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_hit_tokens: int = 0,
    latency_s: float = 0.0,
    attempt: Optional[int] = None,
) -> LLMCallRecord:
    """Record a completed LLM call, emit a flushed structured log line, and notify listeners."""
    eff_phase = phase or _CURRENT_PHASE.get() or "unspecified"
    eff_tier = str(tier or _CURRENT_TIER.get() or "fast")
    eff_attempt = int(attempt if attempt is not None else (_CURRENT_ATTEMPT.get() or 1))
    rec = LLMCallRecord(
        phase=eff_phase,
        tier=eff_tier,
        model=str(model or "unknown"),
        prompt_tokens=max(0, int(prompt_tokens or 0)),
        completion_tokens=max(0, int(completion_tokens or 0)),
        reasoning_tokens=max(0, int(reasoning_tokens or 0)),
        cache_hit_tokens=max(0, int(cache_hit_tokens or 0)),
        latency_s=round(max(0.0, float(latency_s or 0.0)), 3),
        attempt=max(1, eff_attempt),
    )
    _GLOBAL_SESSION.record(rec)
    active = _ACTIVE_SESSION.get()
    if active is not None and active is not _GLOBAL_SESSION:
        active.record(rec)

    msg = (
        f"[LLM_CALL] phase={rec.phase} tier={rec.tier} model={rec.model} "
        f"prompt_tokens={rec.prompt_tokens} completion_tokens={rec.completion_tokens} "
        f"reasoning_tokens={rec.reasoning_tokens} cache_hit_tokens={rec.cache_hit_tokens} "
        f"latency_s={rec.latency_s:.3f} attempt={rec.attempt}"
    )
    logger.info(msg)
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass

    with _GLOBAL_LOCK:
        listeners = list(_LISTENERS)
    for fn in listeners:
        try:
            fn(rec)
        except Exception:
            pass
    return rec


def get_llm_telemetry_summary() -> Dict[str, Any]:
    active = _ACTIVE_SESSION.get()
    if active is not None:
        return active.summary()
    return _GLOBAL_SESSION.summary()


def reset_llm_telemetry() -> None:
    global _GLOBAL_SESSION
    with _GLOBAL_LOCK:
        _GLOBAL_SESSION = LLMTelemetrySession(run_id="global")
    active = _ACTIVE_SESSION.get()
    if active is not None:
        with active.lock:
            active.calls.clear()


@contextmanager
def register_llm_telemetry_listener(listener: Callable[[LLMCallRecord], None]):
    with _GLOBAL_LOCK:
        _LISTENERS.append(listener)
    try:
        yield
    finally:
        with _GLOBAL_LOCK:
            if listener in _LISTENERS:
                _LISTENERS.remove(listener)
