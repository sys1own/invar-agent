"""Static, provenance-preserving readers. Uploaded code is never executed."""
from __future__ import annotations

import ast
import csv
import io
import json
import re
from pathlib import Path

from .archive_guard import IngestionError
from .schemas import (DisambiguationFlag, DisambiguationTriggerKind, ParsedDocument,
                      RawConstant, RawEquation, SourceSpan)

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _group(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise IngestionError("LATEX_MACRO_INVALID", "Expected balanced argument")
    depth = 1
    end = start + 1
    while end < len(text):
        if text[end] in "{}" and (end == 0 or text[end - 1] != "\\"):
            depth += 1 if text[end] == "{" else -1
            if depth == 0:
                return text[start + 1:end], end + 1
        end += 1
    raise IngestionError("LATEX_MACRO_INVALID", "Unclosed argument")


def expand_latex_newcommands(tex: str, max_depth: int = 8) -> str:
    definitions = {}
    declaration = re.compile(r"\\(?:newcommand|renewcommand)\*?\s*")
    while match := declaration.search(tex):
        start = match.end()
        if start < len(tex) and tex[start] == "{":
            name, start = _group(tex, start)
        else:
            macro = re.match(r"\\[A-Za-z]+", tex[start:])
            if not macro:
                raise IngestionError("LATEX_MACRO_INVALID", "Missing macro name")
            name, start = macro[0], start + macro.end()
        arity = re.match(r"\s*\[([0-9])\]", tex[start:])
        count = int(arity[1]) if arity else 0
        if arity:
            start += arity.end()
        body, end = _group(tex, start)
        definitions[name.strip()] = (count, body)
        tex = tex[:match.start()] + tex[end:]
    for _ in range(min(max(0, max_depth), 8)):
        pieces = []
        cursor = 0
        for match in re.finditer(r"\\[A-Za-z]+", tex):
            if match.start() < cursor or match[0] not in definitions:
                continue
            count, body = definitions[match[0]]
            end = match.end()
            for index in range(1, count + 1):
                argument, end = _group(tex, end)
                body = body.replace(f"#{index}", argument)
            pieces.extend([tex[cursor:match.start()], body])
            cursor = end
        pieces.append(tex[cursor:])
        expanded = "".join(pieces)
        if len(expanded) > 1024 * 1024:
            raise IngestionError("LATEX_MACRO_EXPANSION_LIMIT", "Expanded document exceeds 1 MB")
        if expanded == tex:
            break
        tex = expanded
    return tex


def _span(path: str, location: str, quote: str, start=None, end=None) -> SourceSpan:
    return SourceSpan(file=path, location=location, quote=quote[:2000] or " ",
                      char_start=start, char_end=end)


def _flag(doc: ParsedDocument, span: SourceSpan, target="") -> None:
    doc.flags.append(DisambiguationFlag(
        flag_id=f"{doc.path}:{span.location}:prose:{len(doc.flags)}",
        trigger=DisambiguationTriggerKind.PROSE_ONLY_CONSTANT,
        target_symbol_or_eq=target, context_excerpt=span.quote, source_span=span))


def _table(doc: ParsedDocument, rows, location: str) -> bool:
    rows = [[str(v or "").strip() for v in row] for row in rows if row]
    if not rows:
        return False
    headers = [h.lower() for h in rows[0]]
    symbol = next((headers.index(h) for h in ("symbol", "param", "parameter", "constant") if h in headers), None)
    if symbol is None or "value" not in headers:
        return False
    value = headers.index("value")
    unit = next((headers.index(h) for h in ("unit", "units") if h in headers), None)
    for index, row in enumerate(rows[1:], 1):
        if max(symbol, value, unit or 0) >= len(row) or re.fullmatch(r"[-: ]+", row[value]):
            continue
        quote = next((line for line in doc.text.splitlines() if all(cell in line for cell in row if cell)), doc.text[:2000])
        span = _span(doc.path, f"{location}:row:{index}", quote)
        if re.fullmatch(NUMBER, row[value]):
            doc.constants.append(RawConstant(symbol=row[symbol].strip("$ "), value=row[value],
                                            unit=row[unit] if unit is not None else "1", provenance=span))
        else:
            _flag(doc, span, row[symbol])
    return True


def _python(doc: ParsedDocument, code: str, location: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise IngestionError("PY_SYNTAX_ERROR", str(exc)) from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = " -> " + ast.unparse(node.returns) if node.returns else ""
            doc.function_signatures.append(f"{node.name}({ast.unparse(node.args)}){annotation}")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                for target in targets:
                    if isinstance(target, ast.Name):
                        quote = ast.get_source_segment(code, node) or target.id
                        doc.constants.append(RawConstant(symbol=target.id, value=str(value),
                            provenance=_span(doc.path, f"{location}:line:{node.lineno}", quote)))


def _text(doc: ParsedDocument, text: str, location: str = "line") -> None:
    covered = set()
    math = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\$)\$(?!\$)([^\n$]+)\$|```(?:math|latex)\s*\n(.*?)```", re.S)
    for match in math.finditer(text):
        expression = next(v for v in match.groups() if v is not None)
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.end())
        end = len(text) if end == -1 else end
        doc.equations.append(RawEquation(text=expression.strip(), provenance=_span(
            doc.path, f"{location}:{text.count(chr(10), 0, match.start()) + 1}", text[start:end], start, end)))
        covered.update(range(text.count("\n", 0, match.start()) + 1,
                             text.count("\n", 0, match.end()) + 2))
    table_rows = [line.strip().strip("|").split("|") for line in text.splitlines() if "|" in line]
    _table(doc, table_rows, f"{location}:table")
    for index, line in enumerate(text.splitlines(), 1):
        stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line.strip())
        if index in covered or "|" in line or not stripped:
            continue
        assignment = re.fullmatch(rf"([A-Za-z_]\w*)\s*=\s*({NUMBER})\s*([A-Za-zµμ°%][\w/*^(). -]*)?", stripped.rstrip(".,;"))
        span = _span(doc.path, f"{location}:{index}", line)
        if assignment:
            raw_u = (assignment[3] or "1").strip().rstrip(".,;")
            doc.constants.append(RawConstant(symbol=assignment[1], value=assignment[2], unit=raw_u or "1", provenance=span))
        elif (
            re.fullmatch(r"[A-Za-z_\d\s+*/^().<>=!{}\\-]+", stripped)
            and re.search(r"[=<>]", stripped)
            and not re.search(r"\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", stripped)
        ):
            doc.equations.append(RawEquation(text=stripped, provenance=span))
        else:
            inline_assigns = list(re.finditer(
                rf"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*=\s*({NUMBER})(?:\s+(1/(?:s|min|hr|day|yr)|s\^-1|m/s|M|mM|uM|nM|cm|mm|m|kg|g|mol|s|min|hr|K|Pa|W|J|V|A|Hz|eV))?(?![A-Za-z0-9_])",
                stripped,
            ))
            if inline_assigns:
                for m_as in inline_assigns:
                    sym_name = m_as.group(1)
                    if sym_name.lower() in {"if", "is", "at", "in", "for", "and", "or", "by", "to", "of"}:
                        continue
                    doc.constants.append(
                        RawConstant(symbol=sym_name, value=m_as.group(2), unit=m_as.group(3) or "1", provenance=span)
                    )
            elif re.search(rf"{NUMBER}\s+(?:1/(?:s|min)|M|mM|cM|cm|mm|m|kg|mol|s|min|K)\b", stripped):
                _flag(doc, span)


def _tex(doc: ParsedDocument, text: str) -> None:
    from pylatexenc.latexwalker import LatexWalker, LatexMathNode, LatexEnvironmentNode, LatexMacroNode, LatexCharsNode

    # Validate definitions and bound their expansion before walking the source tree.
    expand_latex_newcommands(text)
    definitions = []
    for match in re.finditer(r"\\(?:newcommand|renewcommand)\*?", text):
        start = match.end()
        if text[start:].lstrip().startswith("{"):
            _, end = _group(text, start)
        else:
            name = re.match(r"\s*\\[A-Za-z]+", text[start:])
            if not name:
                raise IngestionError("LATEX_MACRO_INVALID", "Missing macro name")
            end = start + name.end()
        arity = re.match(r"\s*\[\d\]", text[end:])
        if arity:
            end += arity.end()
        _, end = _group(text, end)
        definitions.append(text[match.start():end])
    nodes, _, _ = LatexWalker(text, tolerant_parsing=True).get_latex_nodes()

    def walk(nodes):
        for node in nodes or []:
            quote = text[node.pos:node.pos + node.len]
            if isinstance(node, LatexMathNode) or (isinstance(node, LatexEnvironmentNode)
                    and node.environmentname in {"equation", "equation*", "align", "align*", "gather", "gather*", "displaymath"}):
                inner = quote
                if isinstance(node, LatexMathNode):
                    inner = quote[len(node.delimiters[0]):-len(node.delimiters[1])]
                else:
                    inner = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", quote)
                inner = expand_latex_newcommands("\n".join(definitions) + "\n" + inner)
                for row in re.split(r"\\\\", inner):
                    row = re.sub(r"\\label\{[^}]*\}", "", row).replace("&", "").strip()
                    if row:
                        doc.equations.append(RawEquation(text=row, provenance=_span(doc.path,
                            f"line:{text.count(chr(10), 0, node.pos) + 1}", quote, node.pos, node.pos + node.len)))
            elif isinstance(node, LatexMacroNode) and node.macroname == "ensuremath" and node.nodeargd:
                argument = node.nodeargd.argnlist[-1]
                if argument is not None:
                    raw = text[argument.pos + 1:argument.pos + argument.len - 1]
                    doc.equations.append(RawEquation(text=expand_latex_newcommands("\n".join(definitions) + raw),
                        provenance=_span(doc.path, f"line:{text.count(chr(10), 0, node.pos) + 1}", quote, node.pos, node.pos + node.len)))
            elif isinstance(node, LatexEnvironmentNode) and node.environmentname == "tabular":
                body = re.sub(r"\\(?:begin\{tabular\}(?:\{[^}]*\})?|end\{tabular\}|hline|toprule|midrule|bottomrule)", "", quote)
                _table(doc, [row.strip().split("&") for row in re.split(r"\\\\", body) if row.strip()], f"line:{text.count(chr(10), 0, node.pos) + 1}:table")
            elif isinstance(node, LatexCharsNode):
                _text(doc, quote, f"line:{text.count(chr(10), 0, node.pos) + 1}:prose")
            else:
                walk(getattr(node, "nodelist", None))
    walk(nodes)
    if not doc.equations and not doc.constants:
        _text(doc, text)


def parse_document(path: Path) -> ParsedDocument:
    path = Path(path)
    doc = ParsedDocument(path=str(path), text="")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber
        from pdfminer.pdfparser import PDFSyntaxError
        from pypdf import PdfReader

        try:
            with pdfplumber.open(path) as pdf:
                for index, page in enumerate(pdf.pages, 1):
                    text = page.extract_text(layout=True) or ""
                    doc.text += text + "\n"
                    for number, rows in enumerate(page.extract_tables(), 1):
                        _table(doc, rows, f"page:{index}:table:{number}")
                    # Table cells are parsed once, not again as unbound prose.
                    _text(doc, "\n".join(line for line in text.splitlines()
                                         if re.search(r"[=<>]", line)), f"page:{index}:line")
                    if len(re.sub(r"\s", "", text)) < 40:
                        _flag(doc, _span(doc.path, f"page:{index}", text or "[No machine-readable text]"))
                    elif not page.extract_tables() and not doc.constants and re.search(NUMBER, text):
                        _flag(doc, _span(doc.path, f"page:{index}", text))
        except (ValueError, TypeError, KeyError, OSError, PDFSyntaxError):
            reader = PdfReader(path)
            for index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                doc.text += text + "\n"
                _text(doc, text, f"page:{index}:line")
                if not doc.constants:
                    _flag(doc, _span(doc.path, f"page:{index}", text or "[No machine-readable text]"))
        return doc
    doc.text = path.read_text(encoding="utf-8")
    if suffix == ".tex":
        _tex(doc, doc.text)
    elif suffix in {".csv", ".tsv"}:
        if not _table(doc, list(csv.reader(io.StringIO(doc.text), delimiter="\t" if suffix == ".tsv" else ",")), "table:1"):
            doc.datasets.append(str(path))
    elif suffix == ".json":
        data = json.loads(doc.text)
        rows = data if isinstance(data, list) else data.get("parameters", []) if isinstance(data, dict) else []
        if rows and all(isinstance(row, dict) for row in rows):
            headers = list(rows[0])
            found = _table(doc, [headers] + [[row.get(key, "") for key in headers] for row in rows], "table:1")
        else:
            found = False
        if not found:
            doc.datasets.append(str(path))
    elif suffix == ".ipynb":
        notebook = json.loads(doc.text)
        for index, cell in enumerate(notebook.get("cells", [])):
            source = cell.get("source", [])
            source = "".join(source) if isinstance(source, list) else source
            if cell.get("cell_type") == "code":
                _python(doc, source, f"cell:{index}")
            else:
                _text(doc, source, f"cell:{index}")
    elif suffix == ".py":
        _python(doc, doc.text, "module")
    elif suffix == ".md":
        _text(doc, doc.text)
        for index, match in enumerate(re.finditer(r"```python\s*\n(.*?)```", doc.text, re.S)):
            _python(doc, match[1], f"code:{index}")
    else:
        raise IngestionError("UPLOAD_DISALLOWED_TYPE", path.name)
    return doc
