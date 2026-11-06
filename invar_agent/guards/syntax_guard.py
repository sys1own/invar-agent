"""Multi-language syntax guards for staged overlay buffers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class SyntaxCheckResult:
    language: str
    valid: bool
    errors: list[str] = field(default_factory=list)


class MultiLangSyntaxGuard:
    """Verifies staged buffers parse cleanly before they may be committed."""

    def verify_python_cst(self, source: str) -> SyntaxCheckResult:
        errors: list[str] = []
        try:
            import libcst as cst

            cst.parse_module(source)
            return SyntaxCheckResult(language="python", valid=True)
        except Exception as libcst_error:  # noqa: BLE001
            errors.append(f"libcst: {libcst_error}")
        try:
            compile(source, "<staged>", "exec")
            return SyntaxCheckResult(language="python", valid=True, errors=errors)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} (line {e.lineno})")
            return SyntaxCheckResult(language="python", valid=False, errors=errors)

    def verify_python_ast(self, source: str) -> SyntaxCheckResult:
        return self.verify_python_cst(source)

    def verify_rust_syntax(self, source: str) -> SyntaxCheckResult:
        errors: list[str] = []
        # Brace/paren/bracket balance outside comments and strings (approximate).
        pairs = {"{": "}", "(": ")", "[": "]"}
        stack: list[str] = []
        in_str = False
        in_char = False
        in_line_comment = False
        in_block_comment = 0
        prev = ""
        for lineno, line in enumerate(source.splitlines(), start=1):
            in_line_comment = False
            i = 0
            while i < len(line):
                ch = line[i]
                nxt = line[i + 1] if i + 1 < len(line) else ""
                if in_line_comment:
                    break
                if in_block_comment:
                    if ch == "/" and nxt == "*":
                        in_block_comment += 1
                        i += 2
                        continue
                    if ch == "*" and nxt == "/":
                        in_block_comment -= 1
                        i += 2
                        continue
                    i += 1
                    continue
                if ch == "/" and nxt == "/":
                    break
                if ch == "/" and nxt == "*":
                    in_block_comment += 1
                    i += 2
                    continue
                if in_str:
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == '"':
                        in_str = False
                    i += 1
                    continue
                if in_char:
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == "'":
                        in_char = False
                    i += 1
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "'" and nxt != "'":
                    # Lifetime vs char heuristic: treat as char only if closing soon.
                    if re.match(r"^'(\\.|[^'])'", line[i:]):
                        in_char = True
                elif ch in pairs:
                    stack.append(pairs[ch])
                elif ch in pairs.values():
                    if not stack or stack.pop() != ch:
                        errors.append(f"unbalanced '{ch}' at line {lineno}")
                        return SyntaxCheckResult(language="rust", valid=False, errors=errors)
                prev = ch
                i += 1
            if in_str:
                errors.append(f"unterminated string at line {lineno}")
                return SyntaxCheckResult(language="rust", valid=False, errors=errors)
        if stack:
            errors.append(f"unclosed delimiter(s): {stack}")
            return SyntaxCheckResult(language="rust", valid=False, errors=errors)
        return SyntaxCheckResult(language="rust", valid=True)

    def verify_json(self, source: str) -> SyntaxCheckResult:
        try:
            json.loads(source)
            return SyntaxCheckResult(language="json", valid=True)
        except json.JSONDecodeError as e:
            return SyntaxCheckResult(language="json", valid=False, errors=[f"JSONDecodeError: {e}"])

    def verify_toml(self, source: str) -> SyntaxCheckResult:
        try:
            import tomllib  # type: ignore
        except ImportError:
            tomllib = None  # type: ignore
        try:
            if tomllib is not None:
                tomllib.loads(source)
            else:
                # Minimal structural check: sections and key = value lines.
                for lineno, line in enumerate(source.splitlines(), start=1):
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("[") and not s.endswith("]"):
                        return SyntaxCheckResult(language="toml", valid=False,
                                                 errors=[f"malformed section at line {lineno}"])
            return SyntaxCheckResult(language="toml", valid=True)
        except Exception as e:  # noqa: BLE001
            return SyntaxCheckResult(language="toml", valid=False, errors=[f"TOML error: {e}"])

    def verify_by_extension(self, path: str, source: str) -> SyntaxCheckResult:
        if path.endswith(".py"):
            return self.verify_python_cst(source)
        if path.endswith(".rs"):
            return self.verify_rust_syntax(source)
        if path.endswith(".json"):
            return self.verify_json(source)
        if path.endswith(".toml"):
            return self.verify_toml(source)
        return SyntaxCheckResult(language="unknown", valid=True)
