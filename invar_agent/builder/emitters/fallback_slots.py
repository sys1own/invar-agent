"""Dependency-free leaf implementations generated solely from frozen contracts."""
from __future__ import annotations

import math

from ..ingestion.schemas import CanonicalResearchIR
from ..uast_models import InterfacePortNodeModel


COMMON = '''import csv
import json
import math
from html import escape

def adapt_rows(rows, columns=None, si_scales=None):
    result = []
    for row in rows:
        mapped = {}
        for key, value in row.items():
            name = (columns or {}).get(key, key)
            number = float(value) * (si_scales or {}).get(name, 1.0)
            if not math.isfinite(number):
                raise ValueError("Nonfinite observation: " + str(name))
            mapped[name] = number
        result.append(mapped)
    return result

def load_data(path, columns=None, si_scales=None):
    with open(path, encoding="utf-8", newline="") as stream:
        rows = json.load(stream) if str(path).endswith(".json") else list(csv.DictReader(stream))
    return adapt_rows(rows, columns, si_scales)

def save_data(rows, path):
    rows = adapt_rows(rows)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        if str(path).endswith(".json"):
            json.dump(rows, stream, allow_nan=False)
        elif rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def render_svg(trajectories):
    traces = trajectories if isinstance(trajectories, dict) else {"trajectory": trajectories}
    paths = []
    for label, values in traces.items():
        values = [float(v) for v in values]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite trajectory")
        scale = max([abs(v) for v in values] + [1.0])
        points = " ".join(str(10 + 380*i/max(1, len(values)-1)) + "," + str(100-80*v/scale) for i,v in enumerate(values))
        paths.append('<polyline fill="none" stroke="blue" aria-label="' + escape(str(label), quote=True) + '" points="' + points + '"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">' + "".join(paths) + '</svg>'

def run_workflow(value, stages, lower=None, upper=None):
    for stage in stages:
        value = stage(value)
        if not math.isfinite(value) or lower is not None and value < lower or upper is not None and value > upper:
            raise ValueError("Workflow bound violation")
    return value
'''


def narrative(ir: CanonicalResearchIR) -> str:
    from .base import relativize_provenance_path

    lines = ["# " + ir.title, "", "## Verified equations"]
    lines.extend(f"- ${e.canonical_latex}$ — {relativize_provenance_path(e.provenance.file)}: {e.provenance.location}" for e in ir.equations)
    lines.append("\n## SI constants")
    lines.extend(f"- {c.symbol} = {c.si_value!r} {c.dimension.canonical_unit} — {relativize_provenance_path(c.provenance.file)}: {c.provenance.location}" for c in ir.constants)
    return "\n".join(lines) + "\n"


def emit_slot(emitter_id: str, port: InterfacePortNodeModel, ir: CanonicalResearchIR) -> str:
    if emitter_id == "fb_markdown_narrative_doc":
        return narrative(ir)
    names = [field.name for field in port.inputs]
    values = []
    for field in port.outputs:
        if field.name in names:
            expression = field.name
        elif field.scalar_type == "str":
            expression = "render_svg({'trajectory': [0.0]})" if emitter_id == "fb_svg_trajectory_viz" else "''"
        else:
            expression = {"f64": "0.0", "i64": "0", "bool": "False", "vec_f64": "[]", "table": "[]"}[field.scalar_type]
        if field.scalar_type in {"f64", "i64"} and field.bounds:
            if field.bounds.lower is not None:
                lo = field.bounds.lower if field.bounds.lower_inclusive else math.nextafter(field.bounds.lower, float("inf"))
                expression = f"max({lo!r}, {expression})"
            if field.bounds.upper is not None:
                hi = field.bounds.upper if field.bounds.upper_inclusive else math.nextafter(field.bounds.upper, -float("inf"))
                expression = f"min({hi!r}, {expression})"
        values.append(repr(field.name) + ": " + expression)
    return COMMON + f"\ndef {port.port_name}({', '.join(names)}):\n    return {{{', '.join(values)}}}\n"
