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

logger = logging.getLogger("invar_agent.copilot.agent")


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
    ) -> None:
        self.workspace_path = Path(workspace_path)
        self.provider = provider or "gemini"
        self.api_key = api_key or ""
        self.model = model
        self.tier = tier
        self.max_history = max_history
        self._client_factory = client_factory

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

    def _llm_client(self):
        if self._client_factory is not None:
            return self._client_factory(self.provider, tier=self.tier, api_key=self.api_key, model=self.model)
        from invar_agent.llm.clients import get_llm_client

        # LLM call rationale: copilot chat is a lightweight planning/summarisation
        # task, so it always uses Tier.FAST (light model, low latency).
        return get_llm_client(
            self.provider, tier=Tier.FAST, api_key=self.api_key, model=self.model
        )

    # -- chat ---------------------------------------------------------------------

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run one copilot turn; returns ``{"reply": markdown, "action": {...}|None}``."""
        message = (message or "").strip()
        if not message:
            return {"reply": "", "action": None}

        client = self._llm_client()
        if client is None:
            return self._fallback_reply(message)

        messages = [{"role": "system", "content": self.system_prompt()}]
        for turn in self._sanitize_history(history):
            messages.append(turn)
        messages.append({"role": "user", "content": message})

        try:
            raw = client.generate(messages, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Copilot LLM call failed: %s", exc)
            return self._fallback_reply(message, error=str(exc))

        if not raw or not raw.strip():
            return self._fallback_reply(message)

        reply, action = format_copilot_response(raw)
        action = self._normalize_action(action)
        return {"reply": reply or "I have no further comments.", "action": action}

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

    def _fallback_reply(self, message: str, error: str = "") -> Dict[str, Any]:
        """Deterministic offline fallback when no LLM provider is configured."""
        if _has_build_intent(message):
            prompt = sanitize_builder_prompt(message) or message.strip()
            reply = (
                "### Suggested Build Prompt\n\n"
                "No LLM provider is configured, so I drafted a build prompt directly "
                "from your request. Review it in the Action Card before sending it to "
                "the Builder.\n\n"
                + (f"_Provider error: {error}_" if error else "")
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
            }
        reply = (
            "I'm running in offline mode — no LLM provider is configured. "
            "Add an API key via the settings (gear) icon to enable full copilot "
            "planning. I can still draft build prompts for concrete build requests."
        )
        if error:
            reply += f"\n\n_Provider error: {error}_"
        return {"reply": reply, "action": None}
