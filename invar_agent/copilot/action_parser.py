"""Extract structured build actions from Co-pilot Markdown and JSON responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger("invar_agent.copilot.action_parser")


_CODE_FENCE_RE = re.compile(
    r"```(?:yaml|json)\s+blueprint\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_SUGGEST_JSON_FENCE_RE = re.compile(
    r"```(?:json)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_BUILD_PROMPT_FENCE_RE = re.compile(
    r"```build_prompt\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# New strict delimiter preferred by the system prompt.
_BUILDER_PROMPT_TAG_RE = re.compile(
    r"<builder_prompt>(.*?)</builder_prompt>",
    re.DOTALL | re.IGNORECASE,
)

# Legacy XML tag alias.
_LEGACY_BUILD_PROMPT_TAG_RE = re.compile(
    r"<build_prompt>(.*?)</build_prompt>",
    re.DOTALL | re.IGNORECASE,
)

_EXPLANATION_TAG_RE = re.compile(
    r"<explanation>(.*?)</explanation>",
    re.DOTALL | re.IGNORECASE,
)

# Generic prompt/markdown fences used by some LLMs.
_PROMPT_FENCE_RE = re.compile(
    r"```(?:prompt|markdown)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Structured ``action:trigger_build`` speedup card emitted by the Copilot.
_TRIGGER_BUILD_RE = re.compile(
    r"```action:trigger_build\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Meta wrappers and conversational preambles that must not leak into a build prompt.
_META_PROMPT_PATTERNS = [
    re.compile(
        r"^\s*Here(?:'s| is)?\s+(?:a\s+)?(?:detailed\s+)?(?:ready-to-use\s+)?prompt[^\n]*\n*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*You\s+can\s+paste(?:\s+this)?(?:\s+directly)?(?:\s+into|\s+in)?(?:\s+the\s+(?:builder|prompt))?[^\n]*\n*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*Below\s+is\s+(?:the\s+)?(?:build\s+)?prompt[^\n]*\n*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*Build\s+Contract\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Build\s+Prompt\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*yaml\s+blueprint\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s* acceleration\s*[:=]\s*[^\n]+\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s* target\s*[:=]\s*[^\n]+\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*["\']{1,2}', re.MULTILINE),
    re.compile(r'["\']{1,2}\s*$', re.MULTILINE),
]

# Intro / outro phrases that may wrap a prompt when the LLM ignores the strict delimiters.
_INTRO_PATTERNS = [
    re.compile(
        r"^\s*(?:I['’]ll\s+give\s+you\s+a\s+(?:ready-to-use\s+)?(?:build\s+)?prompt[^\n]*(?:\n\n|\n))",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:Here(?:'s| is)?\s+(?:a\s+)?(?:detailed\s+)?(?:ready-to-use\s+)?(?:build\s+)?prompt[^\n]*(?:\n\n|\n))",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:Below\s+is\s+(?:the\s+)?(?:build\s+)?(?:ready-to-use\s+)?prompt[^\n]*(?:\n\n|\n))",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:You\s+can\s+use\s+this\s+(?:build\s+)?prompt[^\n]*(?:\n\n|\n))",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:This\s+(?:build\s+)?prompt[^\n]*(?:\n\n|\n))", re.IGNORECASE),
]

_OUTRO_PATTERNS = [
    re.compile(
        r"\n\s*(?:You\s+can\s+paste\s+(?:this\s+)?(?:directly\s+)?(?:into|in)\s+(?:the\s+)?(?:builder|prompt)[^\n]*)\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\n\s*(?:Let\s+me\s+know\s+if[^\n]*)\s*$", re.IGNORECASE),
    re.compile(
        r"\n\s*(?:Suggested\s+Builder\s+Prompt|Copy|Edit\s+in\s+Builder)[^\n]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\n\s*(?:Here\s+is\s+the\s+full\s+prompt[^\n]*)\s*$", re.IGNORECASE),
    re.compile(
        r"(?:\s*[,.-])?\s*Target:\s*[^\n]+(?:\s*Acceleration:\s*[^\n]+)?\s*$",
        re.IGNORECASE,
    ),
]

# Preambles specifically for the *inside* of a builder prompt payload, e.g. when the
# LLM echoes "I've crafted a prompt for you..." inside the suggested_prompt field.
# Each pattern stops at the first colon or newline so the actual payload is preserved.
_BUILDER_PROMPT_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"I(?:['’]ve| have|['’]m| am)?\s+(?:crafted|created|designed|written|prepared|built|generated|assembled)\s+(?:a|the|this)\s+(?:detailed\s+)?(?:build\s+)?prompt"
    r"|Here(?:['’]s| is)?\s+(?:a|the|this)\s+(?:detailed\s+)?(?:build\s+)?prompt"
    r"|Below\s+is\s+(?:a|the|this)\s+(?:detailed\s+)?(?:build\s+)?prompt"
    r"|Sure[,!]?\s+I\s+can\s+help[^\n]*?\s+(?:with\s+(?:a|the|this)\s+)?(?:detailed\s+)?(?:build\s+)?prompt"
    r"|This\s+(?:is\s+)?(?:a|the|this)?\s+(?:detailed\s+)?(?:build\s+)?prompt"
    r"|You\s+can\s+use\s+(?:this\s+)?(?:build\s+)?prompt"
    r")\s*(?:[^\n:]*?:\s*|[^\n]*?\n\s*)",
    re.IGNORECASE,
)


def _parse_trigger_build_block(text: str) -> Optional[Dict[str, Any]]:
    """Parse an ``action:trigger_build`` fenced JSON block if present."""
    match = _TRIGGER_BUILD_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    builder_prompt = str(payload.get("builder_prompt") or "").strip()
    if not builder_prompt:
        return None
    return {
        "builder_prompt": builder_prompt,
        "target_language": str(payload.get("target_language") or "").strip(),
        "architecture": str(payload.get("architecture") or "").strip(),
        "target_files": list(payload.get("target_files", []) or []),
    }


def _trigger_build_parameters(block: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Build a parameter bag from an ``action:trigger_build`` block."""
    builder_prompt = block["builder_prompt"]
    architecture = block["architecture"]
    target = _normalize_target(architecture) or _infer_target_from_text(builder_prompt)
    params = {
        "target": target or "pure_python",
        "target_language": block["target_language"]
        or ("rust" if "rust" in (target or "") else "cpp"),
        "architecture": _normalize_target(architecture) or target or "pure_python",
        "target_files": block["target_files"],
        "acceleration": _normalize_acceleration(builder_prompt),
    }
    params.update(_extract_engine_parameters(block))
    return params


def _normalize_target(raw: Any) -> Optional[str]:
    """Return a recognized Aero-Forge target name or None."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    valid = {
        "pure_python",
        "pure_rust",
        "hybrid_rust_python",
        "hybrid_cpp_python",
        "hybrid_cpp_rust",
        "multi_crate_rust",
        "tri_polyglot_rust_cpp_python",
        "wasm",
    }
    if value in valid:
        return value
    synonyms = {
        "rust_python": "hybrid_rust_python",
        "python_rust": "hybrid_rust_python",
        "cpp_python": "hybrid_cpp_python",
        "python_cpp": "hybrid_cpp_python",
        "c++_python": "hybrid_cpp_python",
        "rust_cpp": "hybrid_cpp_rust",
        "cpp_rust": "hybrid_cpp_rust",
        "tri_polyglot": "tri_polyglot_rust_cpp_python",
    }
    return synonyms.get(value)


def _normalize_acceleration(raw: Any) -> str:
    """Return a recognized acceleration policy string."""
    if not isinstance(raw, str):
        return "Selective Acceleration (Auto-Detect Heavy Compute)"
    lowered = raw.lower()
    if "force" in lowered or "force native" in lowered:
        return "Force Native Bridge"
    if "bypass" in lowered or "standard" in lowered:
        return "Standard Runtime (Bypass Bridge)"
    return "Selective Acceleration (Auto-Detect Heavy Compute)"


def _normalize_engine_backend(raw: Any) -> Optional[str]:
    """Return a recognized HIN engine backend name or None."""
    if not isinstance(raw, str):
        return None
    value = raw.lower().replace("-", "_").replace(" ", "_")
    if value in ("hin_cpu", "hin_gpu", "hin_wasm"):
        return value
    if value in ("cpu",):
        return "hin_cpu"
    if value in ("gpu", "cuda", "vulkan"):
        return "hin_gpu"
    if value in ("wasm", "wasm32"):
        return "hin_wasm"
    return None


def _normalize_precision_shield(raw: Any) -> Optional[str]:
    """Return a recognized precision shield mode or None."""
    if not isinstance(raw, str):
        return None
    value = raw.lower().replace("-", "_").replace(" ", "_")
    if value == "shield_checks":
        value = "shield"
    if value in ("ieee", "fast_math", "shield"):
        return value
    return None


def _normalize_jit_level(raw: Any) -> Optional[int]:
    """Return a recognized HIN JIT optimization level (0-2) or None."""
    if isinstance(raw, int):
        return raw if 0 <= raw <= 2 else None
    if isinstance(raw, str):
        try:
            level = int(raw.strip())
            return level if 0 <= level <= 2 else None
        except ValueError:
            return None
    return None


def _extract_wavefront_parallelism(raw: Any) -> Optional[int]:
    """Return an integer wavefront parallelism value clamped to 1-16."""
    if isinstance(raw, int):
        return max(1, min(16, raw))
    if isinstance(raw, str):
        try:
            value = int(raw.strip())
            return max(1, min(16, value))
        except ValueError:
            return None
    return None


def _extract_engine_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize engine configuration fields from a parameter dict."""
    engine_backend = _normalize_engine_backend(params.get("engine_backend"))
    wavefront_parallelism = _extract_wavefront_parallelism(
        params.get("wavefront_parallelism")
    )
    precision_shield_mode = _normalize_precision_shield(
        params.get("precision_shield_mode")
    )
    hin_jit_opt_level = _normalize_jit_level(
        params.get("jit_optimization_level")
        if "jit_optimization_level" in params
        else params.get("hin_jit_opt_level")
    )
    result: Dict[str, Any] = {}
    if engine_backend:
        result["engine_backend"] = engine_backend
    if wavefront_parallelism is not None:
        result["wavefront_parallelism"] = wavefront_parallelism
    if precision_shield_mode:
        result["precision_shield_mode"] = precision_shield_mode
    if hin_jit_opt_level is not None:
        result["hin_jit_opt_level"] = hin_jit_opt_level
    return result


def _infer_target_from_text(text: str) -> Optional[str]:
    """Infer a build target from prose when no explicit contract is present."""
    lowered = text.lower()
    if "wasm" in lowered:
        return "wasm"
    has_python = "python" in lowered or "pyo3" in lowered
    has_rust = "rust" in lowered or "cargo" in lowered or "pyo3" in lowered
    has_cpp = (
        "cpp" in lowered
        or "c++" in lowered
        or "cxx" in lowered
        or "clang" in lowered
        or "g++" in lowered
    )

    if has_python and has_rust and has_cpp:
        return "tri_polyglot_rust_cpp_python"
    if has_python and has_rust:
        return "hybrid_rust_python"
    if has_python and has_cpp:
        return "hybrid_cpp_python"
    if has_rust and has_cpp:
        return "hybrid_cpp_rust"
    if has_rust:
        return "pure_rust"
    if has_cpp:
        return "hybrid_cpp_python"
    return "pure_python"


def _has_build_intent(text: str) -> bool:
    """Return True when the text clearly requests a build/feature."""
    lowered = text.lower()
    keywords = (
        "build",
        "create",
        "implement",
        "generate",
        "function",
        "module",
        "project",
        "fast",
        "speed",
        "accelerate",
        "optimize",
        "blueprint",
    )
    return any(k in lowered for k in keywords)


def _strip_outer_quotes(text: str) -> str:
    """Remove surrounding single or double quotes and common escapes."""
    text = text.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    text = (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )
    return text.strip()


def _extract_code_block_from_prompt(text: str) -> str:
    """If the prompt text is wrapped in a Markdown code fence, return the inner content."""
    if not text:
        return text or ""
    # Prefer an explicit ```build_prompt fence, then any generic triple-backtick block.
    for pattern in (
        _BUILD_PROMPT_FENCE_RE,
        re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE),
    ):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text


def _strip_trailing_metadata_tags(text: str) -> str:
    """Remove accidental trailing ``Target:`` and ``Acceleration:`` tags."""
    return re.sub(
        r"(?:\s*[,.-])?\s*Target:\s*[^\n]+(?:\s*Acceleration:\s*[^\n]+)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def sanitize_builder_prompt(prompt_text: str) -> str:
    """Strip conversational preambles and code-fence wrappers from a builder prompt.

    The returned string should start directly with the executable build/task
    requirements and contain no meta-commentary like "I've crafted..." or
    "Here is the prompt...", and no trailing ``Target:`` / ``Acceleration:`` tags.
    """
    if not prompt_text or not prompt_text.strip():
        return ""
    cleaned = _strip_outer_quotes(prompt_text.strip())
    cleaned = _extract_code_block_from_prompt(cleaned)
    cleaned = _BUILDER_PROMPT_PREAMBLE_RE.sub("", cleaned)
    # Fall back to generic intro/outro stripping if the prompt is still wrapped.
    for pattern in _INTRO_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in _OUTRO_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in _META_PROMPT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _strip_trailing_metadata_tags(cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned.rstrip("-:\n ").strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_prompt_fragment(text: str, prompt: str) -> bool:
    """Return True if *text* contains a significant fragment of *prompt*."""
    norm_text = _normalize_whitespace(text)
    norm_prompt = _normalize_whitespace(prompt)
    if not norm_prompt or not norm_text:
        return False
    if norm_prompt in norm_text:
        return True
    # Also match a leading or trailing chunk of the prompt.
    head = norm_prompt[:80]
    tail = norm_prompt[-80:] if len(norm_prompt) > 80 else ""
    for fragment in (head, tail):
        if fragment and fragment in norm_text:
            return True
    return False


def _remove_fenced_prompt_blocks(text: str, prompt: str) -> str:
    """Remove Markdown/JSON/XML code fences whose contents duplicate *prompt*."""
    if not text or not prompt:
        return text or ""

    # Match any triple-backtick block, optionally with an info string.
    fence_re = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    def repl(match: re.Match) -> str:
        content = match.group(1)
        if _contains_prompt_fragment(content, prompt):
            return ""
        return match.group(0)

    return fence_re.sub(repl, text)


def _remove_prompt_lines(text: str, prompt: str) -> str:
    """Remove lines that contain a verbatim copy of *prompt* or a major chunk."""
    if not text or not prompt:
        return text or ""

    norm_prompt = _normalize_whitespace(prompt)
    head = norm_prompt[:80]
    tail = norm_prompt[-80:] if len(norm_prompt) > 80 else ""

    kept: List[str] = []
    for line in text.splitlines():
        norm_line = _normalize_whitespace(line)
        if norm_prompt in norm_line:
            continue
        if head and head in norm_line:
            continue
        if tail and tail in norm_line:
            continue
        kept.append(line)
    return "\n".join(kept)


# Meta headers that may appear above a duplicated prompt in the explanation.
_EXPLANATION_META_HEADERS = [
    re.compile(
        r"^\s*(?:Suggested\s+Builder\s+Prompt|Here\s+is\s+(?:the\s+)?prompt|Builder\s+Prompt|Build\s+Prompt)\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*The\s+suggested\s+build\s+prompt\s+is\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def clean_explanation_text(response_text: str, suggested_prompt: str) -> str:
    """Remove duplicated prompt/code fences and meta headers from the conversational text.

    The remaining text should be a short rationale (1-3 sentences) with no
    repetition of the build prompt.
    """
    if not response_text:
        return ""
    cleaned = response_text
    if suggested_prompt:
        cleaned = _remove_fenced_prompt_blocks(cleaned, suggested_prompt)
        cleaned = _remove_prompt_lines(cleaned, suggested_prompt)

    for pattern in _EXPLANATION_META_HEADERS:
        cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    # Trim trailing punctuation/whitespace artifacts.
    cleaned = cleaned.rstrip("-:\n ")
    return cleaned.strip()


class ActionParser:
    """Parse a Co-pilot response into a clean display text and an isolated action."""

    def __init__(self) -> None:
        self.meta_patterns = _META_PROMPT_PATTERNS
        self.intro_patterns = _INTRO_PATTERNS
        self.outro_patterns = _OUTRO_PATTERNS

    def _strip_intro_outro(self, text: str) -> str:
        """Remove common conversational wrappers around the prompt body."""
        if not text:
            return ""
        cleaned = text
        for pattern in self.intro_patterns:
            cleaned = pattern.sub("", cleaned)
        for pattern in self.outro_patterns:
            cleaned = pattern.sub("", cleaned)
        return cleaned

    def sanitize(self, text: str) -> str:
        """Remove meta wrappers and collapse surrounding whitespace."""
        if not text:
            return ""
        cleaned = _strip_outer_quotes(text)
        cleaned = self._strip_intro_outro(cleaned)
        for pattern in self.meta_patterns:
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a raw or fenced JSON response, returning None on failure."""
        if not text:
            return None
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            fence = _SUGGEST_JSON_FENCE_RE.search(stripped)
            if fence:
                stripped = fence.group(1).strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    def _extract_prompt_from_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Pull the executable prompt from a structured JSON object."""
        if not isinstance(data, dict):
            return None
        action = data.get("action") or {}
        for key in (
            "suggested_prompt",
            "clean_prompt",
            "target_prompt",
            "build_prompt",
            "prompt",
        ):
            candidate = action.get(key) if isinstance(action, dict) else None
            if not candidate:
                candidate = data.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def _extract_display_text(self, data: Dict[str, Any], fallback: str = "") -> str:
        """Return the conversational display_text/explanation or the fallback."""
        if not isinstance(data, dict):
            return fallback.strip()
        for key in ("display_text", "explanation", "reply", "message"):
            candidate = data.get(key)
            if isinstance(candidate, str):
                return candidate.strip()
        return fallback.strip()

    def _extract_parameters(self, prompt: str, data: Any) -> Dict[str, Any]:
        """Build a parameter bag from explicit JSON fields or inferred from prompt text."""
        params: Dict[str, Any] = {}
        action_data: Optional[Dict[str, Any]] = None
        if isinstance(data, dict):
            action_data = data.get("action") or {}
            if isinstance(action_data, dict):
                params = dict(
                    action_data.get("parameters") or action_data.get("params") or {}
                )

        target = (
            params.get("target")
            or params.get("architecture")
            or (action_data.get("target") if isinstance(action_data, dict) else None)
            or (data.get("target") if isinstance(data, dict) else None)
        )
        acceleration = (
            params.get("acceleration")
            or (
                action_data.get("acceleration")
                if isinstance(action_data, dict)
                else None
            )
            or (data.get("acceleration") if isinstance(data, dict) else None)
        )
        target_files = (
            params.get("target_files")
            or (
                action_data.get("target_files")
                if isinstance(action_data, dict)
                else None
            )
            or (data.get("target_files") if isinstance(data, dict) else None)
        )

        if not target:
            target = _normalize_target(prompt) or _infer_target_from_text(prompt)
        if not acceleration:
            acceleration = _normalize_acceleration(prompt)

        target = _normalize_target(target) or target
        acceleration = _normalize_acceleration(acceleration)

        engine_params = _extract_engine_parameters(params)

        result = {
            "target": target or "pure_python",
            "acceleration": acceleration,
            **engine_params,
        }
        if target_files:
            result["target_files"] = list(target_files)
        return result

    def _extract_blueprint(self, data: Any) -> Optional[str]:
        """Return a raw blueprint string if one is embedded in the action."""
        if not isinstance(data, dict):
            return None
        action = data.get("action") or {}
        if isinstance(action, dict):
            for key in ("blueprint", "blueprint_yaml", "blueprint_json"):
                candidate = action.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        candidate = data.get("blueprint")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None

    def extract_clean_prompt(self, text: str) -> Optional[str]:
        """Return only the distilled inner prompt or blueprint string.

        The returned string has meta-preambles, outer quotes, and YAML/JSON
        blueprint wrappers removed.
        """
        if not text or not text.strip():
            return None

        # 1. Structured JSON response with a clean prompt field.
        data = self._try_parse_json(text)
        if data:
            candidate = self._extract_prompt_from_json(data)
            if candidate:
                return sanitize_builder_prompt(self.sanitize(candidate))

        # 1a. Structured ``action:trigger_build`` speedup card.
        trigger = _parse_trigger_build_block(text)
        if trigger:
            return sanitize_builder_prompt(self.sanitize(trigger["builder_prompt"]))

        # 2. New <builder_prompt> XML delimiter (preferred by the system prompt).
        tag = _BUILDER_PROMPT_TAG_RE.search(text)
        if tag:
            return sanitize_builder_prompt(self.sanitize(tag.group(1)))

        # 3. New ```build_prompt fence format.
        fence = _BUILD_PROMPT_FENCE_RE.search(text)
        if fence:
            return sanitize_builder_prompt(self.sanitize(fence.group(1)))

        # 4. Legacy <build_prompt> XML alias and generic prompt/markdown fences.
        tag = _LEGACY_BUILD_PROMPT_TAG_RE.search(text)
        if tag:
            return sanitize_builder_prompt(self.sanitize(tag.group(1)))
        fence = _PROMPT_FENCE_RE.search(text)
        if fence:
            return sanitize_builder_prompt(self.sanitize(fence.group(1)))

        # 4. YAML/JSON blueprint contract block: return the builder prompt inside it.
        contract_match = _CODE_FENCE_RE.search(text)
        if contract_match:
            raw_contract = contract_match.group(1).strip()
            try:
                parsed_contract = yaml.safe_load(raw_contract)
                if isinstance(parsed_contract, dict):
                    prompt = parsed_contract.get("prompt") or parsed_contract.get(
                        "build_prompt"
                    )
                    if isinstance(prompt, str) and prompt.strip():
                        return sanitize_builder_prompt(self.sanitize(prompt))
            except yaml.YAMLError:
                try:
                    parsed_contract = json.loads(raw_contract)
                    if isinstance(parsed_contract, dict):
                        prompt = parsed_contract.get("prompt") or parsed_contract.get(
                            "build_prompt"
                        )
                        if isinstance(prompt, str) and prompt.strip():
                            return sanitize_builder_prompt(self.sanitize(prompt))
                except json.JSONDecodeError:
                    pass
            return sanitize_builder_prompt(self.sanitize(raw_contract))

        # 5. Plain text with build intent: return sanitized text as-is.
        if _has_build_intent(text):
            return sanitize_builder_prompt(self.sanitize(text))

        return None

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse a Co-pilot response into a structured packet.

        Returns
        -------
        {
            "display_text": str,
            "action": None | {
                "type": str,
                "clean_prompt": str,
                "parameters": {"target": str, "acceleration": str},
                "blueprint": str | None,
            },
        }
        """
        if not text or not text.strip():
            return {"display_text": "", "action": None}

        data = self._try_parse_json(text)

        # New structured JSON response with explicit display_text and action.
        if data and ("display_text" in data or "action" in data):
            display_text = self._extract_display_text(data, "")
            raw_prompt = self._extract_prompt_from_json(data)
            if raw_prompt:
                params = self._extract_parameters(raw_prompt, data)
                clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
                display_text = clean_explanation_text(display_text, clean)
                raw_action = data.get("action")
                action_type = "build"
                if isinstance(raw_action, dict):
                    action_type = raw_action.get("type") or action_type
                elif isinstance(raw_action, str):
                    action_type = raw_action
                blueprint = self._extract_blueprint(data)
                return {
                    "display_text": display_text,
                    "action": {
                        "type": action_type,
                        "source": "structured_json",
                        "clean_prompt": clean,
                        "parameters": params,
                        "blueprint": blueprint,
                    },
                }

        # Legacy JSON reply/action with prompt/target/acceleration inside params.
        if data:
            legacy_action = data.get("action")
            if isinstance(legacy_action, dict):
                legacy_params = (
                    legacy_action.get("params") or legacy_action.get("parameters") or {}
                )
                prompt = (
                    legacy_params.get("prompt")
                    or legacy_params.get("build_prompt")
                    or legacy_params.get("clean_prompt")
                    or data.get("build_prompt")
                    or data.get("prompt")
                    or data.get("reply")
                    or ""
                )
                if isinstance(prompt, str) and prompt.strip():
                    params = self._extract_parameters(prompt, data)
                    clean = sanitize_builder_prompt(self.sanitize(prompt))
                    display_text = clean_explanation_text(
                        self._extract_display_text(data, ""), clean
                    )
                    action_type = legacy_action.get("type")
                    if not action_type or action_type in (
                        "PROPOSE_BUILD",
                        "SUGGEST_BUILD_PROMPT",
                    ):
                        action_type = "build"
                    return {
                        "display_text": display_text,
                        "action": {
                            "type": action_type,
                            "source": "legacy_json",
                            "clean_prompt": clean,
                            "parameters": params,
                            "blueprint": self._extract_blueprint(data),
                        },
                    }

            # JSON with reply only.
            display_text = self._extract_display_text(data, "")
            if display_text:
                return {"display_text": display_text, "action": None}
            # Non-action raw JSON should be left for the caller to pretty-print.
            if text.strip().startswith(("{", "[")):
                return {"display_text": "", "action": None}

        # Markdown/fenced fallback.
        display_text = text

        # ``action:trigger_build`` speedup card (highest priority markdown action).
        trigger = _parse_trigger_build_block(text)
        if trigger:
            raw_prompt = trigger["builder_prompt"]
            clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
            display_text = _TRIGGER_BUILD_RE.sub("", display_text)
            display_text = clean_explanation_text(self.sanitize(display_text), clean)
            params = _trigger_build_parameters(trigger, text)
            return {
                "display_text": display_text,
                "action": {
                    "type": "trigger_build",
                    "source": "action_trigger_build",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # Build prompt fence (new clean format).
        fence = _BUILD_PROMPT_FENCE_RE.search(text)
        if fence:
            raw_prompt = fence.group(1)
            clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
            display_text = _BUILD_PROMPT_FENCE_RE.sub("", display_text)
            display_text = clean_explanation_text(self.sanitize(display_text), clean)
            params = self._extract_parameters(raw_prompt, {})
            return {
                "display_text": display_text,
                "action": {
                    "type": "build",
                    "source": "build_prompt_fence",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # XML-style <builder_prompt> / <build_prompt> fallback.
        for tag_re in (_BUILDER_PROMPT_TAG_RE, _LEGACY_BUILD_PROMPT_TAG_RE):
            tag = tag_re.search(text)
            if tag:
                raw_prompt = tag.group(1)
                clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
                display_text = tag_re.sub("", display_text)
                display_text = clean_explanation_text(
                    self.sanitize(display_text), clean
                )
                params = self._extract_parameters(raw_prompt, {})
                return {
                    "display_text": display_text,
                    "action": {
                        "type": "build",
                        "source": "xml_tag",
                        "clean_prompt": clean,
                        "parameters": params,
                        "blueprint": None,
                    },
                }

        # YAML/JSON blueprint contract block.
        contract_match = _CODE_FENCE_RE.search(text)
        if contract_match:
            raw_contract = contract_match.group(1).strip()
            contract_data: Any = None
            try:
                contract_data = yaml.safe_load(raw_contract)
            except yaml.YAMLError:
                try:
                    contract_data = json.loads(raw_contract)
                except json.JSONDecodeError:
                    pass
            prompt = ""
            if isinstance(contract_data, dict):
                prompt = (
                    contract_data.get("prompt")
                    or contract_data.get("build_prompt")
                    or ""
                )
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            raw_prompt = prompt if prompt else raw_contract
            clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
            display_text = _CODE_FENCE_RE.sub("", display_text)
            display_text = clean_explanation_text(self.sanitize(display_text), clean)
            params = self._extract_parameters(
                raw_prompt, contract_data if isinstance(contract_data, dict) else {}
            )
            return {
                "display_text": display_text,
                "action": {
                    "type": "build",
                    "source": "blueprint_contract",
                    "contract": True,
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": raw_contract,
                },
            }

        # Plain text with build intent.
        if _has_build_intent(text):
            raw_prompt = text
            clean = sanitize_builder_prompt(self.sanitize(raw_prompt))
            params = self._extract_parameters(raw_prompt, {})
            return {
                "display_text": "",
                "action": {
                    "type": "build",
                    "source": "plain_text",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # No actionable payload.
        return {"display_text": self.sanitize(text), "action": None}


def extract_clean_prompt(text: str) -> Optional[str]:
    """Module-level helper for the ActionParser clean-prompt extraction."""
    return ActionParser().extract_clean_prompt(text)


# ---------------------------------------------------------------------------
# Resilient Builder Plan payload recovery
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Closed JSON string value (handles escaped quotes/backslashes, may span lines).
_JSON_STRING_RE = r'"((?:[^"\\]|\\.)*)"'

_DISPLAY_TEXT_RE = re.compile(
    r'"display_text"\s*:\s*' + _JSON_STRING_RE, re.DOTALL)
# Prefer a properly terminated clean_prompt value...
_CLEAN_PROMPT_CLOSED_RE = re.compile(
    r'"(?:clean_prompt|build_prompt|expert_prompt|suggested_prompt)"\s*:\s*'
    + _JSON_STRING_RE + r'(?=\s*[,}])', re.DOTALL)
# ...but recover a truncated value running to EOF as well (spec recovery path).
_CLEAN_PROMPT_OPEN_RE = re.compile(
    r'"(?:clean_prompt|build_prompt|expert_prompt|suggested_prompt)"\s*:\s*'
    + _JSON_STRING_RE[:-1], re.DOTALL)


def _unescape_json_string(value: str) -> str:
    """Unescape a JSON string body (\n, \\", \\\\ …) without requiring quotes."""
    try:
        return json.loads('"' + value + '"')
    except Exception:  # noqa: BLE001
        # Truncated escapes: decode the common sequences manually.
        out = value.replace('\\"', '"').replace("\\n", "\n")
        out = out.replace("\\t", "\t").replace("\\r", "\r")
        return out.replace("\\\\", "\\")


def _balanced_json_span(text: str) -> Optional[str]:
    """First ``{...}`` span from ``text`` (balanced, string-aware).

    Unterminated objects (truncated LLM output) extend to EOF.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def extract_builder_plan_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """Recover a Builder plan payload from raw (possibly truncated) LLM output.

    Handles leading prose (e.g. ``"Architecture Overview"``), ``<think>``
    blocks, markdown fences, and JSON truncated near EOF. Returns a normalized
    dict with ``display_text`` / ``clean_prompt`` (+ ``expert_prompt`` /
    ``action_obj`` aliases), or ``None`` when no build prompt can be recovered.
    """
    if not raw_text or not raw_text.strip():
        return None
    text = _THINK_BLOCK_RE.sub("", raw_text).strip()

    display_text = ""
    clean_prompt = ""

    # 1. Well-formed JSON object anywhere in the response.
    span = _balanced_json_span(text)
    data: Optional[Dict[str, Any]] = None
    if span:
        try:
            parsed = json.loads(span)
            if isinstance(parsed, dict):
                data = parsed
        except Exception:  # noqa: BLE001
            data = None
    if data is not None:
        display_text = str(data.get("display_text") or "")
        action = data.get("action")
        if isinstance(action, dict):
            clean_prompt = str(action.get("clean_prompt") or action.get("prompt") or "")
        if not clean_prompt:
            for key in ("clean_prompt", "build_prompt", "expert_prompt", "suggested_prompt"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    clean_prompt = value
                    break
        # A parsed reply with action=null is plain chat, not a build plan.
        if not clean_prompt.strip():
            return None
    else:
        # 2. Truncated JSON recovery: regex-extract string fields even without
        #    closing quotes/braces (e.g. cut off mid ``clean_prompt``).
        match = _DISPLAY_TEXT_RE.search(text)
        if match:
            display_text = _unescape_json_string(match.group(1))
        match = _CLEAN_PROMPT_CLOSED_RE.search(text)
        if not match:
            match = _CLEAN_PROMPT_OPEN_RE.search(text)
        if match:
            clean_prompt = _unescape_json_string(match.group(1))

    clean_prompt = sanitize_builder_prompt(clean_prompt).strip()
    display_text = display_text.strip()
    if not clean_prompt:
        return None

    return {
        "status": "success",
        "action": "SUGGEST_BUILD_PROMPT",
        "display_text": display_text,
        "expert_prompt": clean_prompt,
        "clean_prompt": clean_prompt,
        "action_obj": {"type": "build", "clean_prompt": clean_prompt},
    }


def extract_action(text: str) -> Dict[str, Any]:
    """Module-level helper that returns a full structured action packet."""
    return ActionParser().parse(text)


def extract_build_prompt(text: str) -> Tuple[str, Optional[str]]:
    """Split a copilot response into conversational reply and build prompt.

    Priority:
      1. A fenced `` ```build_prompt ... ``` `` block.
      2. A legacy JSON code-fence containing a ``suggest_build_prompt`` action.
      3. XML-style ``<build_prompt>`` / ``<explanation>`` tags.
      4. No prompt found: return the full text as the reply.

    Returns
    -------
    (conversational_reply, extracted_prompt)
    """
    raw = text or ""
    stripped = raw.strip()

    # 1. New ``build_prompt`` fence format.
    fence = _BUILD_PROMPT_FENCE_RE.search(stripped)
    if fence:
        extracted = fence.group(1).strip()
        reply = _BUILD_PROMPT_FENCE_RE.sub("", stripped).strip()
        return reply, extracted

    # 2. JSON code fence containing a suggest_build_prompt payload.
    json_fence = _SUGGEST_JSON_FENCE_RE.search(stripped)
    if json_fence:
        payload = json_fence.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and (
            parsed.get("action") == "suggest_build_prompt" or "build_prompt" in parsed
        ):
            reply = str(parsed.get("explanation") or "").strip()
            extracted = str(parsed.get("build_prompt") or "").strip() or None
            if not reply:
                reply = _SUGGEST_JSON_FENCE_RE.sub("", stripped).strip()
            if extracted:
                return reply, extracted

    # 3. XML-style fallback tags.
    bp_match = _BUILDER_PROMPT_TAG_RE.search(
        stripped
    ) or _LEGACY_BUILD_PROMPT_TAG_RE.search(stripped)
    if bp_match:
        extracted = bp_match.group(1).strip()
        reply = ""
        ex_match = _EXPLANATION_TAG_RE.search(stripped)
        if ex_match:
            reply = ex_match.group(1).strip()
        if not reply:
            reply = _BUILDER_PROMPT_TAG_RE.sub(
                "", _LEGACY_BUILD_PROMPT_TAG_RE.sub("", stripped)
            ).strip()
        return reply, extracted

    # 4. No prompt found.
    return stripped, None


def extract_build_contract(text: str) -> Optional[Dict[str, Any]]:
    """Parse a YAML or JSON build contract from a Markdown code fence.

    Only explicitly tagged blocks (```yaml blueprint ...``` or
    ```json blueprint ...```) are treated as build contracts. This avoids
    matching broad document keys like ``architecture_overview``.
    """
    if not text or not text.strip():
        return None

    match = _CODE_FENCE_RE.search(text)
    if not match:
        return None

    payload = match.group(1).strip()
    if not payload:
        return None

    parsed: Any = None
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.debug("Build contract block is not valid YAML/JSON")
            return None

    if not isinstance(parsed, dict):
        return None

    prompt = parsed.get("prompt") or parsed.get("build_prompt")
    target = _normalize_target(parsed.get("target", parsed.get("architecture")))
    if not prompt or not target:
        return None

    result = {
        "prompt": str(prompt).strip(),
        "target": target,
        "acceleration": _normalize_acceleration(parsed.get("acceleration")),
    }
    result.update(_extract_engine_parameters(parsed))
    return result


def parse_suggested_build_prompt(text: str) -> Dict[str, Any]:
    """Extract the new ``suggest_build_prompt`` structured payload.

    Returns a dict with ``explanation``, ``has_suggestion``, ``build_prompt``,
    and ``raw``.  Tries ``build_prompt`` fenced blocks first, then JSON code
    fences, then XML-style ``<build_prompt>`` / ``<explanation>`` fallbacks.
    """
    raw = text or ""
    stripped = raw.strip()

    reply, build_prompt = extract_build_prompt(stripped)
    if build_prompt:
        return {
            "explanation": reply,
            "has_suggestion": True,
            "build_prompt": build_prompt,
            "raw": stripped,
        }

    return {
        "explanation": stripped,
        "has_suggestion": False,
        "build_prompt": None,
        "raw": stripped,
    }


def _build_suggestion(parsed: Dict[str, Any], raw: str) -> Dict[str, Any]:
    build_prompt = str(parsed.get("build_prompt") or "").strip()
    explanation = str(parsed.get("explanation") or "").strip()
    return {
        "explanation": explanation,
        "has_suggestion": bool(build_prompt),
        "build_prompt": build_prompt or None,
        "raw": raw,
    }


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a build action from any assistant text."""
    suggestion = parse_suggested_build_prompt(text)
    if suggestion["has_suggestion"]:
        raw_prompt = suggestion["build_prompt"] or ""
        target = _normalize_target(raw_prompt) or _infer_target_from_text(raw_prompt)
        acceleration = _normalize_acceleration(raw_prompt)
        clean_prompt = sanitize_builder_prompt(raw_prompt)
        return {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": clean_prompt,
                "explanation": suggestion["explanation"],
                "target": target,
                "acceleration": acceleration,
            },
        }

    contract = extract_build_contract(text)
    if contract:
        return {
            "type": "PROPOSE_BUILD",
            "params": {
                "prompt": sanitize_builder_prompt(contract["prompt"]),
                "target": contract["target"],
                "acceleration": contract["acceleration"],
            },
        }

    target = _infer_target_from_text(text)
    if not target or not _has_build_intent(text):
        return None

    raw_prompt = text.strip()
    if len(raw_prompt) > 500:
        raw_prompt = raw_prompt[:500].rsplit(" ", 1)[0] + "..."
    if raw_prompt.startswith("{"):
        raw_prompt = raw_prompt[:200]

    return {
        "type": "PROPOSE_BUILD",
        "params": {
            "prompt": sanitize_builder_prompt(raw_prompt),
            "target": target,
            "acceleration": _normalize_acceleration(text),
        },
    }


def _maybe_parse_json_object(
    response: str,
) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
    """Handle legacy top-level JSON responses with ``reply`` and ``action``."""
    text = response.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    reply = str(parsed.get("reply", response)).strip()
    action = parsed.get("action")
    if action and isinstance(action, dict):
        params = action.get("params") or {}
        prompt = params.get("prompt") or reply
        target = _normalize_target(params.get("target", params.get("architecture")))
        if prompt and target:
            result = {
                "type": "PROPOSE_BUILD",
                "params": {
                    "prompt": str(prompt).strip(),
                    "target": target,
                    "acceleration": _normalize_acceleration(params.get("acceleration")),
                },
            }
            result["params"].update(_extract_engine_parameters(params))
            return reply, result
    return reply, None


def parse_copilot_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Split an assistant response into a Markdown reply and an optional action.

    Supports, in order:
    - ``build_prompt`` fenced blocks (new clean format)
    - ``suggest_build_prompt`` JSON payloads
    - ``yaml blueprint`` / ``json blueprint`` fenced build contracts
    - legacy top-level JSON objects with ``reply``/``action``
    - best-effort extraction from plain prose
    """
    if not response or not response.strip():
        return "", None

    # New ``build_prompt`` fence format.
    reply, raw_prompt = extract_build_prompt(response)
    if raw_prompt:
        reply = clean_explanation_text(reply, raw_prompt)
        target = _normalize_target(raw_prompt) or _infer_target_from_text(raw_prompt)
        action = {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": sanitize_builder_prompt(raw_prompt),
                "explanation": reply,
                "target": target,
                "acceleration": _normalize_acceleration(raw_prompt),
            },
        }
        return reply, action

    # New structured action-card format.
    suggestion = parse_suggested_build_prompt(response)
    if suggestion["has_suggestion"]:
        raw_prompt = suggestion["build_prompt"] or ""
        target = _normalize_target(raw_prompt) or _infer_target_from_text(raw_prompt)
        reply = (
            suggestion["explanation"]
            or _SUGGEST_JSON_FENCE_RE.sub("", response).strip()
        )
        reply = clean_explanation_text(reply, raw_prompt)
        action = {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": sanitize_builder_prompt(raw_prompt),
                "explanation": reply,
                "target": target,
                "acceleration": _normalize_acceleration(raw_prompt),
            },
        }
        return reply, action

    contract = extract_build_contract(response)
    if contract:
        reply = _CODE_FENCE_RE.sub("\n", response).strip()
        reply = clean_explanation_text(reply, contract["prompt"])
        action = {
            "type": "PROPOSE_BUILD",
            "params": {
                "prompt": sanitize_builder_prompt(contract["prompt"]),
                "target": contract["target"],
                "acceleration": contract["acceleration"],
            },
        }
        return reply, action

    legacy = _maybe_parse_json_object(response)
    if legacy is not None:
        reply, action = legacy
        prompt = action["params"]["prompt"] if action and action.get("params") else ""
        return clean_explanation_text(reply, prompt), action

    action = parse_action_from_text(response)
    prompt = action["params"]["prompt"] if action and action.get("params") else ""
    return clean_explanation_text(response.strip(), prompt), action
