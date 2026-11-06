"""Co-pilot response post-processor and chat agent.

Guarantees that every Co-pilot reply sent to the UI is human-readable
Markdown with an isolated, sanitized action payload.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from invar_agent.config import Tier
from invar_agent.copilot.action_parser import (

    ActionParser,
    _has_build_intent,
    _infer_target_from_text,
    _normalize_acceleration,
    _normalize_target,
    clean_explanation_text,
    sanitize_builder_prompt,
)
from invar_agent.copilot.prompts import COPILOT_SYSTEM_PROMPT
from invar_agent.llm.errors import LLMProviderError, record_llm_error, redact_secrets

logger = logging.getLogger("invar_agent.copilot.agent")

_RESEARCH_INTENT = re.compile(
    r"\b(research|hypothesis|soliton|kdv|korteweg|investigate|experiment|"
    r"scientific|verify that|verify the)\b",
    re.IGNORECASE,
)


def is_research_intent(message: str) -> bool:
    return bool(_RESEARCH_INTENT.search(message or ""))


def _legacy_action_type(action: Dict[str, Any]) -> str:
    """Map the canonical ActionParser type to the legacy UI action type."""
    new_type = action.get("type", "build")
    source = action.get("source", "")
    if (
        action.get("contract")
        or source in ("blueprint_contract", "legacy_json", "plain_text")
        or new_type in ("apply_blueprint", "PROPOSE_BUILD")
    ):
        return "PROPOSE_BUILD"
    if new_type in ("suggest_build_prompt", "trigger_build") or source in (
        "build_prompt_fence",
        "xml_tag",
        "structured_json",
        "action_trigger_build",
    ):
        return "SUGGEST_BUILD_PROMPT"
    return "SUGGEST_BUILD_PROMPT"


def _to_legacy_action(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an ActionParser result to the historical action shape used by the UI."""
    action = parsed.get("action")
    if not action:
        return None

    params = action.get("parameters") or {}
    clean = sanitize_builder_prompt(action.get("clean_prompt", ""))
    legacy = {
        "type": _legacy_action_type(action),
        "params": {
            "prompt": clean,
            "target": params.get("target")
            or _normalize_target(clean)
            or _infer_target_from_text(clean),
            "acceleration": params.get("acceleration")
            or _normalize_acceleration(clean),
            "explanation": parsed.get("display_text", ""),
            "parameters": params,
        },
    }
    if params.get("target_files"):
        legacy["params"]["target_files"] = list(params["target_files"])
    return legacy


def _has_markdown_heading(text: str) -> bool:
    """Return True if the text already contains a Markdown heading."""
    return bool(re.search(r"^#{1,4} ", text, re.MULTILINE))


def format_copilot_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return a Markdown-formatted reply and an optional action.

    The action payload is isolated, meta-preamble-free, and never wrapped in a
    YAML/JSON code block inside the Markdown reply.
    """
    if not response or not response.strip():
        return "", None

    parsed = ActionParser().parse(response)
    action = parsed.get("action") or {}
    clean_prompt = sanitize_builder_prompt(action.get("clean_prompt", "") or "")
    display_text = clean_explanation_text(parsed.get("display_text", ""), clean_prompt)
    legacy_action = _to_legacy_action(parsed)

    # If the model returned raw JSON without a human display_text, pretty-print it
    # for the chat timeline but do not expose it as an action.
    if not display_text and _looks_like_json(response):
        try:
            data = json.loads(response.strip())
            display_text = (
                "### Response\n```json\n" + json.dumps(data, indent=2) + "\n```"
            )
        except json.JSONDecodeError:
            display_text = response.strip()

    if legacy_action:
        prompt_text = clean_prompt or legacy_action["params"]["prompt"]
        # Avoid duplicating the executable prompt in the chat bubble.
        if not display_text or display_text.strip() == prompt_text.strip():
            target = legacy_action["params"]["target"]
            target_label = (target or "build").replace("_", " ").title()
            display_text = f"### Architecture Overview\n\nI propose a **{target_label}** build. Use the Action Card to edit or trigger it."

    if not _has_markdown_heading(display_text) and display_text:
        display_text = "### Architecture Overview\n\n" + display_text

    if not display_text:
        display_text = clean_explanation_text(
            parsed.get("display_text", response.strip()), clean_prompt
        )

    return display_text.strip(), legacy_action


def _looks_like_json(text: str) -> bool:
    """Return True if the text appears to be a raw JSON object/list."""
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith(("{", "["))


# ---------------------------------------------------------------------------
# Workspace context (replaces aero-forge's blueprint.aero provider)
# ---------------------------------------------------------------------------

def workspace_state(workspace_path: Path) -> str:
    """Classify the workspace as 'draft' or 'active' from its source inventory."""
    try:
        from invar_agent.context.bundle_repo import WorkspaceIngestor

        report = WorkspaceIngestor(workspace_path).ingest_repository_graph()
    except Exception:  # noqa: BLE001
        return "draft"
    if report.file_count == 0 or (report.python_files == 0 and report.rust_files == 0):
        return "draft"
    return "active"


def workspace_context_tag(workspace_path: Path, max_files: int = 200) -> str:
    """Return a ``<workspace_context>`` tag populated from the invar-agent workspace tree.

    Contains the file inventory plus AST skeletons (imports and signatures) so the
    model can reason about the current workspace without receiving full sources.
    """
    workspace_path = Path(workspace_path)
    try:
        from invar_agent.context.bundle_repo import WorkspaceIngestor
        from invar_agent.context.context_bundler import HolographicContextBundler

        ingestor = WorkspaceIngestor(workspace_path)
        report = ingestor.ingest_repository_graph()
        sources = ingestor.read_sources()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not harvest workspace %s: %s", workspace_path, exc)
        return ""

    if not report.files:
        return ""

    bundler = HolographicContextBundler()
    skeletons = bundler.compress_ast_skeletons(sources, max_chars_per_file=1200)
    file_list = "\n".join(f"- {rel}" for rel in report.files[:max_files])
    payload = (
        f"workspace_root: {workspace_path}\n"
        f"state: {workspace_state(workspace_path)}\n"
        f"file_count: {report.file_count} "
        f"(python={report.python_files}, rust={report.rust_files}, "
        f"total_lines={report.total_lines})\n"
        f"files:\n{file_list}\n\n"
        f"AST skeletons (imports + signatures):\n{skeletons}"
    )
    return f"<workspace_context>\n{payload}\n</workspace_context>"


# ---------------------------------------------------------------------------
# Co-pilot chat agent (FAST tier)
# ---------------------------------------------------------------------------

class CopilotAgent:
    """Conversational planning agent backed by the FAST-tier LLM.

    Every LLM call is routed through ``get_llm_client(..., tier=Tier.FAST)`` —
    copilot chat is a lightweight sub-task (planning/summarisation), so it must
    never consume the reasoning-tier (pro) budget.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        provider: str = "gemini",
        api_key: str = "",
        model: Optional[str] = None,
        tier: Tier = Tier.FAST,
        max_history: int = 12,
        client_factory=None,
        tier_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.workspace_path = Path(workspace_path)
        self.provider = provider or "gemini"
        self.api_key = api_key or ""
        self.model = model
        self.tier = tier
        self.max_history = max_history
        self._client_factory = client_factory
        self.tier_overrides = dict(tier_overrides or {})

    # -- prompt assembly ---------------------------------------------------------

    def system_prompt(self) -> str:
        state = workspace_state(self.workspace_path)
        status = (
            "Workspace is empty/blank. Focus on understanding the user's intent and "
            "drafting an initial build prompt from scratch; do NOT reference existing "
            "build pipelines or contracts."
            if state == "draft"
            else "Workspace is active with existing source files. Analyze the existing "
            "architecture, entrypoints, and contracts, then design updates and "
            "expansions that integrate cleanly with the current code."
        )
        prompt = COPILOT_SYSTEM_PROMPT
        prompt += f"\n\n[WORKSPACE STATE]\n{state}\n\n[WORKSPACE STATUS]\n{status}"
        context = workspace_context_tag(self.workspace_path)
        if context:
            prompt += "\n\n" + context
        return prompt

    def _tier_override(self) -> Optional[str]:
        """Explicit FAST-tier model override (e.g. from ``/api/config``)."""
        model = self.tier_overrides.get(Tier.FAST.value)
        return model.strip() if isinstance(model, str) and model.strip() else None

    def _llm_client(self):
        if self._client_factory is not None:
            return self._client_factory(self.provider, tier=self.tier, api_key=self.api_key, model=self.model)
        from invar_agent.llm.clients import get_llm_client

        # LLM call rationale: copilot chat is a lightweight planning/summarisation
        # task, so it always uses Tier.FAST (light model, low latency).
        return get_llm_client(
            self.provider, tier=Tier.FAST, api_key=self.api_key, model=self.model,
            override=self._tier_override(),
        )

    # -- chat ---------------------------------------------------------------------

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run one copilot turn; returns ``{"reply": markdown, "action": {...}|None}``."""
        message = (message or "").strip()
        if not message:
            return {"reply": "", "action": None}

        client = self._llm_client()
        if client is None:
            # No provider/key at all — a different state from a failed call.
            return self._offline_reply(message, llm_state="not_configured")

        messages = [{"role": "system", "content": self.system_prompt()}]
        for turn in self._sanitize_history(history):
            messages.append(turn)
        messages.append({"role": "user", "content": message})

        try:
            raw = client.generate(messages, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Copilot LLM call failed: %s", exc)
            error = self._as_provider_error(exc, client)
            record_llm_error(error)
            return self._provider_error_reply(message, error)

        if not raw or not raw.strip():
            error = LLMProviderError(
                self.provider, getattr(client, "model", None),
                upstream_message="provider returned an empty response",
                api_key=self.api_key,
            )
            record_llm_error(error)
            return self._provider_error_reply(message, error)

        reply, action = format_copilot_response(raw)
        action = self._normalize_action(action)
        research_action = self._maybe_research_action(message)
        if research_action is not None:
            action = research_action
        return {"reply": reply or "I have no further comments.", "action": action, "llm_state": "ok"}

    # -- research planning ------------------------------------------------------

    def _maybe_research_action(self, message: str) -> Optional[Dict[str, Any]]:
        """Tier-2 REASONING research planning for research-intent messages.

        Plan Mode never exposes raw ResearchBlueprint JSON: it returns an
        editable Expert Deep Research Prompt (SUGGEST_RESEARCH_PROMPT) plus the
        feasibility verdict, or a RESEARCH_REFUSED card.
        """
        if not is_research_intent(message):
            return None
        import asyncio

        from invar_agent.llm.clients import get_llm_client
        from invar_agent.research.planner import generate_expert_research_prompt

        # LLM call rationale: expert research-prompt synthesis must stay fast
        # (the Tier-2 reasoning model was observed taking 5-10 minutes on
        # planning prompts), so it uses Tier.FAST with a bounded budget. Deep
        # reasoning happens later inside the verified sandbox.
        client = get_llm_client(
            self.provider, tier=Tier.FAST, api_key=self.api_key or None,
            override=self.tier_overrides.get(Tier.FAST.value),
        )

        async def _call(question: str):
            if client is None:
                return None
            from invar_agent.llm.deepseek import DeepSeekChatClient
            kwargs = {"thinking": False} if isinstance(client, DeepSeekChatClient) else {}
            resp = client.complete(question, max_tokens=4096, temperature=0.2, **kwargs)
            return resp.content if resp else None

        def _run(coro):
            try:
                return asyncio.run(coro)
            except RuntimeError:
                # Already inside a running event loop: execute in a worker thread.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()

        try:
            result = _run(generate_expert_research_prompt(message, llm_complete=_call))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Research expert-prompt synthesis failed: %s", exc)
            return None
        feasibility = result["feasibility"]
        if feasibility.verdict.startswith("REFUSED"):
            return {
                "type": "RESEARCH_REFUSED",
                "feasibility": feasibility.model_dump(),
                "params": {"question": message},
            }
        return {
            "type": "SUGGEST_RESEARCH_PROMPT",
            "research_prompt": {
                "expert_prompt": result["expert_prompt"],
                "question": message,
                "feasibility": feasibility.model_dump(),
            },
            "params": {"question": message},
        }

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _sanitize_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if not history:
            return []
        turns: List[Dict[str, str]] = []
        for turn in history[-12:]:
            role = turn.get("role") if isinstance(turn, dict) else None
            content = turn.get("content") if isinstance(turn, dict) else None
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                turns.append({"role": role, "content": content[:8000]})
        return turns

    @staticmethod
    def _normalize_action(action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """invar-agent exposes a single builder action type: SUGGEST_BUILD_PROMPT."""
        if not action:
            return None
        if action.get("type") == "PROPOSE_BUILD":
            action = dict(action)
            action["type"] = "SUGGEST_BUILD_PROMPT"
        params = action.get("params") or {}
        prompt = params.get("prompt") or ""
        if not prompt.strip():
            return None
        return {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": sanitize_builder_prompt(prompt),
                "target": params.get("target"),
                "acceleration": params.get("acceleration"),
                "explanation": params.get("explanation", ""),
            },
        }

    # -- fallback replies ---------------------------------------------------------

    @staticmethod
    def _as_provider_error(exc: Exception, client: Any) -> "LLMProviderError":
        """Normalize any client exception to LLMProviderError (keys redacted)."""
        if isinstance(exc, LLMProviderError):
            return exc
        return LLMProviderError(
            getattr(exc, "provider", None) or "unknown",
            getattr(client, "model", None) or getattr(exc, "model", None),
            status_code=getattr(exc, "status_code", None),
            upstream_message=str(exc),
            retryable=bool(getattr(exc, "retryable", False)),
        )

    def _provider_error_reply(self, message: str, error: "LLMProviderError") -> Dict[str, Any]:
        """Reply for a configured provider whose call failed (distinct from offline)."""
        provider = getattr(error, "provider", None) or self.provider
        model = getattr(error, "model", None) or self.model or "unknown"
        status = getattr(error, "status_code", None)
        detail = redact_secrets(getattr(error, "upstream_message", "") or str(error))
        if status:
            head = f"The LLM call to {provider}/{model} failed (HTTP {status}): {detail}"
        else:
            head = f"The LLM call to {provider}/{model} failed: {detail}"
        hint = (
            "Check that the API key matches the selected provider, that the model "
            "is available for your account, and use \"Test connection\" in the settings modal."
        )
        reply = f"### LLM Provider Error\n\n{head}\n\n_{hint}_"
        result: Dict[str, Any] = {
            "reply": reply, "action": None, "llm_state": "provider_error",
            "llm_error": f"{provider}/{model}: {detail}"[:500],
        }
        # Keep the offline build-prompt draft as a fallback for build requests.
        if _has_build_intent(message):
            prompt = sanitize_builder_prompt(message) or message.strip()
            result["reply"] += (
                "\n\nI drafted a build prompt directly from your request in the "
                "Action Card; review it before sending it to the Builder."
            )
            result["action"] = {
                "type": "SUGGEST_BUILD_PROMPT",
                "params": {
                    "prompt": prompt,
                    "target": _normalize_target(prompt) or _infer_target_from_text(prompt),
                    "acceleration": _normalize_acceleration(prompt),
                    "explanation": "offline fallback after provider error",
                },
            }
        return result

    def _offline_reply(self, message: str, llm_state: str = "not_configured") -> Dict[str, Any]:
        """Deterministic offline fallback when no LLM provider is configured."""
        if _has_build_intent(message):
            prompt = sanitize_builder_prompt(message) or message.strip()
            reply = (
                "### Suggested Build Prompt\n\n"
                "No LLM provider is configured, so I drafted a build prompt directly "
                "from your request. Review it in the Action Card before sending it to "
                "the Builder."
            )
            return {
                "reply": reply,
                "action": {
                    "type": "SUGGEST_BUILD_PROMPT",
                    "params": {
                        "prompt": prompt,
                        "target": _normalize_target(prompt) or _infer_target_from_text(prompt),
                        "acceleration": _normalize_acceleration(prompt),
                        "explanation": "offline fallback",
                    },
                },
                "llm_state": llm_state,
            }
        reply = (
            "I'm running in offline mode — no LLM provider is configured. "
            "Add an API key via the settings (gear) icon to enable full copilot "
            "planning. I can still draft build prompts for concrete build requests."
        )
        return {"reply": reply, "action": None, "llm_state": llm_state}
