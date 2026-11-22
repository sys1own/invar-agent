from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from invar_agent import _fallback
from invar_agent.builder.ingestion import (
    CanonicalResearchIngestionPipeline, IngestionRequest, IRReviewOverride, apply_ir_override,
)
from invar_agent.builder.ingestion.archive_guard import IngestionError, validate_and_unpack_upload
from invar_agent.builder.ingestion.disambiguator import disambiguate_ir
from invar_agent.builder.ingestion.equation_lowerer import (
    FormalGateError, lower_and_verify_equations, normalize_constant,
)
from invar_agent.builder.ingestion.parsers import expand_latex_newcommands, parse_document
from invar_agent.builder.ingestion.schemas import (
    CanonicalResearchIR, ConstantOverridePatch, DisambiguationFlag,
    DisambiguationTriggerKind, RawConstant, RawEquation, SourceSpan,
)
from invar_agent.llm.clients import LLMResponse
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.schemas.research import (
    ResearchBlueprint, VerificationContract, InvariantConstraint,
    StructuredResearchEvidence, ExtractedParameter, ExtractedEquation as VaultEquation,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_API",
                 "OPENROUTER_API_KEY", "OPENAI_API_KEY", "INVAR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")


def span(quote="k = 2", **kwargs):
    return SourceSpan(file="paper.md", location="line:1", quote=quote, **kwargs)


def ingest(tmp_path, *paths, client=None):
    return CanonicalResearchIngestionPipeline(tmp_path, llm_client=client).ingest(
        IngestionRequest(upload_paths=[str(p) for p in paths]))


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_latex_newcommand_expansion_and_depth_cap(tmp_path):
    text = r"""\newcommand{\identity}[1]{#1}
\renewcommand*{\ddt}[1]{\frac{d\identity{#1}}{dt}}
\begin{equation}\ddt{x} = -k x\end{equation}
"""
    expanded = expand_latex_newcommands(text)
    assert r"\frac{dx}{dt}" in expanded
    recursive = expand_latex_newcommands(r"\newcommand{\a}{\a}\a", max_depth=8)
    assert recursive == r"\a"
    mutual = expand_latex_newcommands(r"\newcommand{\a}{\b}\newcommand{\b}{\a}\a")
    assert len(mutual) == 2
    path = write(tmp_path, "paper.tex", text)
    doc = parse_document(path)
    assert doc.equations and r"\frac{dx}{dt}" in doc.equations[0].text
    provenance = doc.equations[0].provenance
    assert text[provenance.char_start:provenance.char_end] == provenance.quote
    with pytest.raises(IngestionError, match="LATEX_MACRO"):
        expand_latex_newcommands(r"\newcommand{\broken}{")


def _scientific_pdf(path):
    """Minimal PDF 1.4 writer using only the PDF standard's built-in Helvetica."""
    commands = ["0.5 w"]
    for x in (50, 160, 280, 420):
        commands.append(f"{x} 660 m {x} 760 l S")
    for y in (660, 685, 710, 735, 760):
        commands.append(f"50 {y} m 420 {y} l S")
    rows = [("Symbol", "Value", "Unit"), ("k", "1.2e-6", "M/min"),
            ("length", "150", "cm"), ("distance", "12.5", "cM")]
    for index, row in enumerate(rows):
        for x, text in zip((60, 175, 295), row):
            commands.append(f"BT /F1 12 Tf {x} {742 - 25 * index} Td ({text}) Tj ET")
    commands.append("BT /F1 12 Tf 60 620 Td (r = k) Tj ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    pdf, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(pdf)
    pdf.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(pdf)


def test_pdf_table_and_equation_extraction(tmp_path):
    path = tmp_path / "paper.pdf"
    _scientific_pdf(path)
    ir = ingest(tmp_path, path)
    constants = {c.symbol: c for c in ir.constants}
    assert constants["k"].si_value == pytest.approx(2e-5)
    assert constants["k"].dimension.canonical_unit == "mol/(m^3*s)"
    assert constants["length"].si_value == pytest.approx(1.5)
    assert constants["distance"].si_value == pytest.approx(0.125)
    assert constants["distance"].dimension.canonical_unit == "Morgan"
    assert ir.equations[0].hin_homogeneous
    assert "page:1" in constants["k"].provenance.location
    assert ir.llm_disambiguation_tokens_used == 0


def test_zip_slip_symlink_and_bomb_rejected(tmp_path, monkeypatch):
    for index, name in enumerate(("../evil.py", "/evil.py", "C:\\evil.py", "..\\evil.py")):
        path = tmp_path / f"slip{index}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("good.md", "$k=2$")
            archive.writestr(name, "bad")
        destination = tmp_path / f"dest{index}"
        with pytest.raises(IngestionError, match="SANDBOX_ZIP_SLIP_REJECTED"):
            validate_and_unpack_upload(path, destination)
        assert not destination.exists()
    path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("link.md")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(IngestionError, match="SANDBOX_ZIP_SYMLINK_REJECTED"):
        validate_and_unpack_upload(path, tmp_path / "links")
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.md", "a" * 200_000)
    with pytest.raises(IngestionError, match="SANDBOX_ZIP_BOMB_REJECTED"):
        validate_and_unpack_upload(path, tmp_path / "bomb")
    assert not (tmp_path / "bomb").exists()
    path = tmp_path / "entries.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(501): archive.writestr(f"{index}.md", "x")
    with pytest.raises(IngestionError, match="SANDBOX_ZIP_BOMB_REJECTED"):
        validate_and_unpack_upload(path, tmp_path / "entries")


class NoLLM:
    def complete(self, *args, **kwargs):
        pytest.fail("A clean document must never invoke the LLM")


def test_clean_paper_zero_llm_tokens(tmp_path):
    path = write(tmp_path, "clean.tex", r"\[k=2\]" + "\n" + r"\[y=k+k\]")
    ir = ingest(tmp_path, path, client=NoLLM())
    assert ir.llm_disambiguation_tokens_used == 0
    assert not ir.disambiguation_flags
    assert len(ir.ir_blake3_digest) == 64


def test_tex_numeric_initial_conditions_populate_state_values(tmp_path):
    path = write(tmp_path, 'initial.tex', r'''\begin{equation}\dot{x}=0\end{equation}
\begin{equation}\dot{y}=0\end{equation}
\begin{equation}x(0)=1\end{equation}
\begin{equation}y(0)=0\end{equation}
\begin{equation}x+y == 1\end{equation}
''')
    ir = ingest(tmp_path, path)
    assert {s.symbol: s.initial_value for s in ir.state_variables} == {'x': 1., 'y': 0.}
    assert ir.llm_disambiguation_tokens_used == 0
    assert all(e.hin_homogeneous and e.z3_satisfiable for e in ir.equations)


def test_tex_mixed_equations_and_prose_are_scoped_and_schema_guided(tmp_path):
    path = write(tmp_path, 'mixed.tex', r'\begin{equation}\dot{z}=-beta*z\end{equation}' + '\nThe coefficient beta was 0.2 1/s.\n')

    class SchemaClient:
        def complete(self, prompt, **kwargs):
            from invar_agent.llm.errors import builder_transport_active
            assert builder_transport_active()
            flags = json.loads(prompt)['flags']
            assert all(set(flag) == {'flag_id', 'trigger', 'target', 'file', 'location', 'excerpt'} for flag in flags)
            assert '"proposal_kind"' in kwargs['system'] and '"raw_unit"' in kwargs['system']
            prose = next(f for f in flags if f['trigger'] == 'PROSE_ONLY_CONSTANT')
            assert prose['excerpt'] == 'The coefficient beta was 0.2 1/s.'
            proposal = dict(flag_id=prose['flag_id'], proposal_kind='constant_binding', symbol='beta',
                raw_value='0.2', raw_unit='1/s', provenance={'file': prose['file'], 'location': prose['location'], 'quote': prose['excerpt']})
            return LLMResponse(content=json.dumps({'proposals': [proposal]}), input_tokens=7, output_tokens=11)

    deterministic = ingest(tmp_path, path)
    assert any(f.trigger.value == 'PROSE_ONLY_CONSTANT' for f in deterministic.disambiguation_flags)
    assert deterministic.llm_disambiguation_tokens_used == 0
    resolved = ingest(tmp_path, path, client=SchemaClient())
    assert resolved.llm_disambiguation_tokens_used == 18
    assert resolved.constants[0].symbol == 'beta' and resolved.constants[0].si_value == .2
    assert resolved.constants[0].disambiguated_by_llm
    assert all(f.resolved for f in resolved.disambiguation_flags)
    assert all(e.hin_homogeneous and e.z3_satisfiable for e in resolved.equations)


def _flagged_ir():
    provenance = span("Coefficient k was 4 s. y = c + k")
    equations, constants, flags = lower_and_verify_equations(
        [RawEquation(text="y = c + k", provenance=provenance)],
        [RawConstant(symbol="c", value="2", unit="m", provenance=span("c = 2 m"))])
    return CanonicalResearchIR(ir_id="flagged", source_mode="upload", title="Flagged paper",
        equations=equations, constants=constants, disambiguation_flags=flags)


class ProposalClient:
    def __init__(self, proposal):
        self.proposal = proposal
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResponse(content=json.dumps({"proposals": [self.proposal]}), input_tokens=7, output_tokens=11)


def test_disambiguator_rejects_hallucinated_quote_and_units():
    for quote in ("The fabricated constant was 4 s.", "Coefficient k was 4 s."):
        ir = _flagged_ir()
        flag = ir.disambiguation_flags[0]
        proposal = dict(flag_id=flag.flag_id, proposal_kind="constant_binding", symbol="k", raw_value="4", raw_unit="s",
                        provenance={**flag.source_span.model_dump(), "quote": quote})
        result = disambiguate_ir(ir, ProposalClient(proposal))
        assert not result.disambiguation_flags[0].resolved
        assert "REJECTED" in result.disambiguation_flags[0].resolution_note
        assert {c.symbol for c in result.constants} == {"c"}
        assert result.llm_disambiguation_tokens_used == 18
        assert ir.llm_disambiguation_tokens_used == 0


def test_ir_override_rechecks_z3(tmp_path):
    ir = ingest(tmp_path, write(tmp_path, "bounds.md", "$k = 2$\n$k >= 0$"))
    before = ir.model_dump_json()
    with pytest.raises(FormalGateError) as error:
        apply_ir_override(ir, IRReviewOverride(constant_patches=[ConstantOverridePatch(constant_id=ir.constants[0].constant_id, si_value=-1)]))
    assert error.value.code == "Z3_SPEC_UNSAT"
    assert error.value.status_code == 422
    assert any(label.startswith("eq:") for label in error.value.details)
    assert ir.model_dump_json() == before
    approved = apply_ir_override(ir, IRReviewOverride(constant_patches=[ConstantOverridePatch(constant_id=ir.constants[0].constant_id, si_value=3)]))
    assert approved.user_approved
    assert approved.ir_blake3_digest != ir.ir_blake3_digest


@pytest.mark.parametrize("name,content", [
    ("paper.md", "| symbol | value | unit |\n|---|---|---|\n| k | 2 | m |\n$y=k+k$"),
    ("paper.tex", r"\begin{tabular}{lll}symbol & value & unit \\ k & 2 & m \\\end{tabular}" + "\n" + r"\[y=k+k\]"),
    ("params.csv", "symbol,value,unit\nk,2,m\n"),
    ("params.tsv", "param\tvalue\tunit\nk\t2\tm\n"),
    ("params.json", '[{"constant":"k","value":2,"unit":"m"}]'),
    ("paper.ipynb", json.dumps({"cells": [{"cell_type": "code", "source": ["k = 2\n", "raise RuntimeError('never execute')\n"]},
                                         {"cell_type": "markdown", "source": ["$y=k+k$"]}]})),
    ("constants.py", "k = 2\nraise RuntimeError('never execute')\n"),
])
def test_upload_formats_are_static_and_zero_token(tmp_path, name, content):
    ir = ingest(tmp_path, write(tmp_path, name, content), client=NoLLM())
    assert ir.constants[0].symbol == "k"
    assert ir.constants[0].si_value == 2
    assert ir.llm_disambiguation_tokens_used == 0


def test_dataset_and_valid_archive(tmp_path):
    archive = tmp_path / "paper.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("paper.md", "$k=2$")
        output.writestr("data.csv", "time,response\n0,1\n1,2\n")
    ir = ingest(tmp_path, archive)
    assert len(ir.datasets_attached) == 1
    assert ir.constants[0].si_value == 2
    with pytest.raises(IngestionError, match="INGESTION_EMPTY"):
        ingest(tmp_path)
    with pytest.raises(IngestionError, match="UPLOAD_DISALLOWED_TYPE"):
        ingest(tmp_path, write(tmp_path, "bad.exe", "x"))


def test_disambiguation_isolates_malformed_siblings_and_json_fences():
    ir = _flagged_ir()
    flag = ir.disambiguation_flags[0]
    proposal = dict(flag_id=flag.flag_id, proposal_kind='constant_binding',symbol='k',raw_value='4',raw_unit='m',
        provenance={**flag.source_span.model_dump(),'quote':'Coefficient k was 4 m.'})
    flag.context_excerpt = 'Coefficient k was 4 m. y = c + k'
    class Client:
        def complete(self, prompt, **kwargs):
            return LLMResponse(content='```json\n'+json.dumps({'proposals':[proposal,{'flag_id':'other','proposal_kind':'unknown'}]})+'\n```',input_tokens=7,output_tokens=11)
    result = disambiguate_ir(ir,Client())
    assert next(c for c in result.constants if c.symbol=='k').si_value == 4
    assert all(f.resolved for f in result.disambiguation_flags)
    assert result.llm_disambiguation_tokens_used == 18


def test_disambiguation_schema_diagnostics_do_not_echo_rejected_values():
    ir = _flagged_ir()
    flag = ir.disambiguation_flags[0]
    result = disambiguate_ir(ir,ProposalClient(dict(flag_id=flag.flag_id,proposal_kind='constant_binding',
        symbol='k',raw_value=4,raw_unit='m',provenance=flag.source_span.model_dump())))
    assert 'raw_value:string_type' in result.disambiguation_flags[0].resolution_note
    assert not result.disambiguation_flags[0].resolved


@pytest.mark.parametrize('content',['', '{"proposals":[]}', '{"proposals":[{}]}', 'invalid raw sk-synthetic123456 <KEY>'])
def test_disambiguation_failure_capture_is_private_and_redacted(tmp_path,content):
    class Client:
        api_key='provider-bound-secret'
        def complete(self,*args,**kwargs):
            return LLMResponse(content=content.replace('<KEY>',self.api_key),input_tokens=2,output_tokens=3)
    directory = tmp_path/'diagnostics'
    result = disambiguate_ir(_flagged_ir(),Client(),failure_capture_dir=directory)
    path = next(directory.glob('*.json'))
    payload = json.loads(path.read_text())
    assert 'provider-bound-secret' not in path.read_text() and 'sk-synthetic123456' not in path.read_text()
    assert payload['input_tokens']==2 and payload['output_tokens']==3
    assert payload['raw_response']==content.replace('<KEY>','sk-****').replace('sk-synthetic123456','sk-****')
    assert path.stat().st_mode & 0o777 in ({0o600, 0o666} if os.name == 'nt' else {0o600})
    assert str(path) in result.disambiguation_flags[0].resolution_note


def test_successful_disambiguation_does_not_capture_failure(tmp_path):
    ir = _flagged_ir()
    flag = ir.disambiguation_flags[0]
    flag.context_excerpt='Coefficient k was 4 m.'
    directory = tmp_path/'diagnostics'
    proposal=dict(flag_id=flag.flag_id,proposal_kind='constant_binding',symbol='k',raw_value='4',raw_unit='m',
        provenance={**flag.source_span.model_dump(),'quote':flag.context_excerpt})
    result=disambiguate_ir(ir,ProposalClient(proposal),failure_capture_dir=directory)
    assert all(f.resolved for f in result.disambiguation_flags)
    assert not directory.exists()


def test_ingestion_pipeline_captures_failed_provider_response_in_workspace(tmp_path):
    path=write(tmp_path,'unresolved.tex',r'\begin{equation}\dot{z}=-beta*z\end{equation}'+'\nThe coefficient beta was 0.2 1/s.\n')
    class Client:
        api_key='bound-provider-secret'
        def complete(self,*args,**kwargs):
            return LLMResponse(content='malformed response bound-provider-secret',input_tokens=5,output_tokens=6)
    result=ingest(tmp_path,path,client=Client())
    saved=next((tmp_path/'.invar/builder/diagnostics/disambiguation').glob('*.json'))
    assert json.loads(saved.read_text())['raw_response']=='malformed response sk-****'
    assert any(str(saved) in (flag.resolution_note or '') for flag in result.disambiguation_flags)


def test_si_case_sensitive_and_unknown_units():
    for unit, value, powers in [("M", 1000, (-3, 0, 0, 0, 0, 1, 0)), ("m", 1, (1, 0, 0, 0, 0, 0, 0)),
                                ("cM", .01, (0,) * 7), ("cm", .01, (1, 0, 0, 0, 0, 0, 0))]:
        constant = normalize_constant(RawConstant(symbol="k", value="1", unit=unit, provenance=span()), "k")
        assert constant.si_value == value
        assert constant.dimension.exponents == powers
    with pytest.raises(FormalGateError, match="HIN_UNKNOWN_UNIT"):
        normalize_constant(RawConstant(symbol="k", value="1", unit="imaginary", provenance=span()), "k")


def test_hin_rejects_dimensional_sum_and_z3_rejects_source(tmp_path):
    with pytest.raises(FormalGateError, match="HIN_SI_DIM_MISMATCH"):
        lower_and_verify_equations([RawEquation(text="y = a + b", provenance=span())],
            [RawConstant(symbol="a", value="1", unit="m", provenance=span()), RawConstant(symbol="b", value="1", unit="s", provenance=span())])
    with pytest.raises(FormalGateError, match="Z3_SPEC_UNSAT"):
        ingest(tmp_path, write(tmp_path, "impossible.md", "$k=2$\n$k < 0$"))


def test_real_persisted_vault_zero_tokens(tmp_path):
    contract = VerificationContract(contract_id="positive", target_backend="cpp",
        invariants=[InvariantConstraint(variable_name="x", assertion="x >= 0")], initial_state=[1.0], dt=.02, steps=50)
    bp = ResearchBlueprint(blueprint_id="example", title="Real vault bundle", falsifiable_hypothesis="positive",
        search_queries=[], contracts=[contract], code_template="def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n",
        unified_module_code="def simulation_step(buf, dt):\n    buf.data[0] = buf.data[0]\n", extracted_constants={"length_scale": 2.0}, variable_map={"x": 0},
        structured_evidence=StructuredResearchEvidence(parameters={"length_scale": ExtractedParameter(
            name="length_scale", symbol="k", raw_value=200, raw_unit="cm", si_value=2, si_unit="m")},
            equations=[VaultEquation(lhs="dx/dt", rhs_latex="0", variables=["x"]), VaultEquation(lhs="y", rhs_latex="x + k", variables=["x", "k"])]),
        uast_spec={"ontology": {"quantities": {"x": {"symbol": "x", "si_dimension": [0, 1, 0, 0, 0, 0, 0], "unit": "m"}}},
                   "governance": {"equations": {}}})
    ResearchVaultManager(tmp_path).register_run(run_id="real_run", title=bp.title, blueprint=bp, report_md="Verified.", status="VERIFIED")
    ir = CanonicalResearchIngestionPipeline(tmp_path, llm_client=NoLLM()).ingest(IngestionRequest(vault_run_ids=["real_run"]))
    assert ir.constants[0].si_value == 2
    assert ir.constants[0].symbol == "k" and ir.constants[0].dimension.canonical_unit == "m"
    assert ir.equations[0].hin_homogeneous
    assert any(e.role.value == "INVARIANT_CONTRACT" and e.canonical_latex == "x >= 0" for e in ir.equations)
    assert ir.state_variables[0].dimension.exponents == (1, 0, 0, 0, 0, 0, 0)
    assert not ir.disambiguation_flags
    for item in [*ir.constants, *ir.equations, *ir.state_variables]:
        assert item.provenance.quote in Path(item.provenance.file).read_text()
    assert ir.state_variables[0].initial_value == 1
    assert ir.default_dt_si_s == .02 and ir.default_steps == 50
    assert ir.time_horizon_si_s == 1
    assert ir.pinned_kernel_code == bp.unified_module_code
    assert ir.llm_disambiguation_tokens_used == 0
    assert ir.vault_run_ids == ["real_run"]
    upload = write(tmp_path, "linked.md", "$z=k+k$")
    hybrid = CanonicalResearchIngestionPipeline(tmp_path, llm_client=NoLLM()).ingest(
        IngestionRequest(vault_run_ids=["real_run"], upload_paths=[str(upload)]))
    assert hybrid.source_mode == "hybrid" and all(e.hin_homogeneous for e in hybrid.equations)
    assert all(f.resolved for f in hybrid.disambiguation_flags)
    root = ResearchVaultManager(tmp_path).run_dir("real_run")
    invalid_blueprint = json.loads((root / "blueprint.json").read_text())
    invalid_blueprint["contracts"][0]["initial_state"] = [-1]
    (root / "blueprint.json").write_text(json.dumps(invalid_blueprint, indent=2))
    with pytest.raises(FormalGateError, match="Z3_SPEC_UNSAT"):
        CanonicalResearchIngestionPipeline(tmp_path, llm_client=NoLLM()).ingest(IngestionRequest(vault_run_ids=["real_run"]))


def _graph():
    return json.dumps(dict(schema_version="builder_uast.v1", project_id="test", canonical_ir_digest="a" * 64, active_profiles=[],
        nodes=[dict(kind="deterministic_module", node_id=name, module_role="python_sdk", emitter_id="sdk", target_relpaths=[f"{name}.py"],
            required_port_ids=[], provided_port_ids=[], emitter_config={}, his_digest="b" * 64) for name in ("a", "b", "c")],
        edges=[dict(from_node="a", to_node="b", edge_kind="verification_dep", port_id=None)]))


def test_eight_fallback_exports_and_native_parity():
    from invar_agent import _native

    graph = _graph()
    assert json.loads(_fallback.validate_builder_uast(graph)) == json.loads(_native.validate_builder_uast(graph))
    schedule = json.loads(_fallback.schedule_builder_wavefronts(graph))
    assert schedule == json.loads(_native.schedule_builder_wavefronts(graph))
    args = (json.dumps(schedule["resolvent_flat"]), json.dumps(schedule["node_order"]),
            '[{"from_node":"b","to_node":"c"}]', '["a"]')
    assert json.loads(_fallback.compute_builder_smw_delta(*args)) == json.loads(_native.compute_builder_smw_delta(*args))
    assert _fallback.compute_his_digest_v2("invar.ir.canonical.v1", "{}") == _native.compute_his_digest_v2("invar.ir.canonical.v1", "{}")
    fields = '["α","b","c","d","e","f"]'
    assert _fallback.compute_fault_fingerprint(fields) == _native.compute_fault_fingerprint(fields)
    dims = '{"x":{"exponents":[1,0,0,0,0,0,0]}}'
    expression = '{"kind":"symbol","name":"x"}'
    assert json.loads(_fallback.verify_equation_dimensions_hin(expression, dims))["output_dimensions"] == json.loads(_native.verify_equation_dimensions_hin(expression, dims))["output_dimensions"]
    assert json.loads(_fallback.extract_hin_causal_slice(graph, "a")) == json.loads(_native.extract_hin_causal_slice(graph, "a"))
    healed = json.loads(_fallback.heal_builder_tier0("class A:\n    def run(self)\n        return math.sqrt(1)\n"))
    assert "import math" in healed["code"] and "class A:" in healed["code"] and healed["llm_tokens_used"] == 0
    for implementation in (_fallback, _native):
        with pytest.raises(ValueError): implementation.compute_fault_fingerprint("[]")
        with pytest.raises(ValueError): implementation.validate_builder_uast("{}")
        with pytest.raises(ValueError, match="GOI_IRREDUCIBLE_CYCLE"):
            implementation.compute_builder_smw_delta(args[0], args[1], '[{"from_node":"b","to_node":"a"}]', "[]")
        with pytest.raises(ValueError): implementation.verify_equation_dimensions_hin("{}", "{}")


def test_ingestion_without_native_extension(tmp_path, monkeypatch):
    import sys
    import invar_agent

    monkeypatch.delattr(invar_agent, "_native", raising=False)
    monkeypatch.setitem(sys.modules, "invar_agent._native", None)
    ir = ingest(tmp_path, write(tmp_path, "paper.md", "$k=2$\n$y=k+k$"))
    assert ir.equations[0].hin_homogeneous and ir.llm_disambiguation_tokens_used == 0
    assert len(ir.ir_blake3_digest) == 64


def test_successful_disambiguation_and_scoped_prompt(tmp_path):
    path = write(tmp_path, "flagged.md", "c = 2 m\nCoefficient k was 4 m. $y = c + k$\n")
    ir = ingest(tmp_path, path)
    flag = next(f for f in ir.disambiguation_flags if f.target_symbol_or_eq == "k")
    proposal = dict(flag_id=flag.flag_id, proposal_kind="constant_binding", symbol="k", raw_value="4", raw_unit="m",
        provenance={**flag.source_span.model_dump(), "quote": "Coefficient k was 4 m."})
    client = ProposalClient(proposal)
    resolved = disambiguate_ir(ir, client)
    assert all(f.resolved for f in resolved.disambiguation_flags)
    assert resolved.equations[0].hin_homogeneous
    assert resolved.llm_disambiguation_tokens_used == 18
    assert next(c for c in resolved.constants if c.symbol == "k").disambiguated_by_llm
    assert len(json.loads(client.prompts[0])["flags"][0]["excerpt"]) <= 1500
    assert ir.constants != resolved.constants
    assert disambiguate_ir(resolved, NoLLM()).model_dump() == resolved.model_dump()


def test_disambiguation_rejects_unsat_proposal(tmp_path):
    path = write(tmp_path, "source.md", "Reported k was -1 m. $k >= 0$\n")
    ir = ingest(tmp_path, path)
    flag = next(f for f in ir.disambiguation_flags if f.target_symbol_or_eq == "k")
    client = ProposalClient(dict(flag_id=flag.flag_id, proposal_kind="constant_binding", symbol="k", raw_value="-1", raw_unit="m",
                                provenance=flag.source_span.model_dump()))
    rejected = disambiguate_ir(ir, client)
    assert "Z3_SPEC_UNSAT" in rejected.disambiguation_flags[0].resolution_note
    assert not rejected.disambiguation_flags[0].resolved and not rejected.constants


def test_disambiguation_cannot_bypass_flags_or_change_sign(tmp_path):
    ir = _flagged_ir()
    flag = ir.disambiguation_flags[0]
    client = ProposalClient(dict(flag_id=flag.flag_id, proposal_kind="state_ordering", ordered_state_symbols=[],
                                provenance=flag.source_span.model_dump()))
    rejected = disambiguate_ir(ir, client)
    assert not rejected.disambiguation_flags[0].resolved
    assert "DISAMBIGUATION_SCOPE_REJECTED" in rejected.disambiguation_flags[0].resolution_note
    ir.disambiguation_flags[0].resolved = True
    with pytest.raises(FormalGateError, match="IR_UNVERIFIED_FLAGGED"):
        apply_ir_override(ir, IRReviewOverride())
    path = write(tmp_path, "signed.md", "Coefficient k was -4 m. $y=k+k$\n")
    ir = ingest(tmp_path, path)
    flag = ir.disambiguation_flags[0]
    client = ProposalClient(dict(flag_id=flag.flag_id, proposal_kind="constant_binding", symbol="k", raw_value="4", raw_unit="m",
                                provenance=flag.source_span.model_dump()))
    rejected = disambiguate_ir(ir, client)
    assert not rejected.disambiguation_flags[0].resolved and not rejected.constants


def test_prose_binding_and_boundary_proposals(tmp_path):
    path = write(tmp_path, "prose.md", "c = 2 m\nCoefficient k was 4 m.\n")
    ir = ingest(tmp_path, path)
    flag = ir.disambiguation_flags[0]
    client = ProposalClient(dict(flag_id=flag.flag_id, proposal_kind="constant_binding", symbol="k", raw_value="4", raw_unit="m",
                                provenance=flag.source_span.model_dump()))
    bound = disambiguate_ir(ir, client)
    assert bound.disambiguation_flags[0].resolved
    assert next(c for c in bound.constants if c.symbol == "k").si_value == 4
    assert apply_ir_override(bound, IRReviewOverride()).user_approved
    text = r"Domain 0 <= x <= 1; \partial_t x = 0"
    path = write(tmp_path, "domain.tex", text)
    provenance = SourceSpan(file=str(path), location="line:1", quote=text)
    equations, constants, flags = lower_and_verify_equations(
        [RawEquation(text=r"\partial_t x = 0", provenance=provenance)], [], {"state_symbols": ["x"]})
    ir = CanonicalResearchIR(ir_id="boundary", source_mode="upload", title="Boundary", equations=equations, constants=constants,
        state_variables=[dict(symbol="x", slot_index=0, dimension={}, provenance=provenance)],
        disambiguation_flags=flags)
    assert flags[0].trigger == DisambiguationTriggerKind.IMPLICIT_BOUNDARY_CONDITION
    for limits, accepted in ((dict(lower=0, upper=1), True), (dict(lower=1, upper=0), False)):
        client = ProposalClient(dict(flag_id=flags[0].flag_id, proposal_kind="boundary_condition", bounds=limits,
                                    provenance=provenance.model_dump()))
        result = disambiguate_ir(ir, client)
        assert result.disambiguation_flags[0].resolved is accepted
        if accepted:
            assert result.state_variables[0].bounds.upper == 1
        else:
            assert "Z3_SPEC_UNSAT" in result.disambiguation_flags[0].resolution_note


def test_pdf_fallback_and_sparse_diagnostics(tmp_path, monkeypatch):
    import pdfplumber

    path = tmp_path / "fallback.pdf"
    _scientific_pdf(path)
    monkeypatch.setattr(pdfplumber, "open", lambda path: (_ for _ in ()).throw(ValueError("broken XRef")))
    doc = parse_document(path)
    assert "r = k" in doc.text
    assert doc.equations
    assert any(flag.trigger == DisambiguationTriggerKind.PROSE_ONLY_CONSTANT for flag in doc.flags)


def test_streamed_archive_total_limit_and_empty(tmp_path, monkeypatch):
    from invar_agent.builder.ingestion import archive_guard

    path = tmp_path / "bounded.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("one.md", "a" * 600)
        output.writestr("two.md", "b" * 600)
    monkeypatch.setattr(archive_guard, "MAX_BYTES", 1000)
    with pytest.raises(IngestionError, match="SANDBOX_ZIP_BOMB_REJECTED"):
        validate_and_unpack_upload(path, tmp_path / "bounded")
    assert not (tmp_path / "bounded").exists()
    with pytest.raises(IngestionError, match="INGESTION_EMPTY"):
        ingest(tmp_path, write(tmp_path, "empty.md", ""))


def test_evolution_initial_and_ambiguous_ordering(tmp_path):
    path = write(tmp_path, "dynamics.tex", r"\[k = 1\]" + "\n" + r"\[\dot{x} = 0\]" + "\n" + r"\[x(0) = 3\]")
    ir = ingest(tmp_path, path)
    assert {e.role.value for e in ir.equations} == {"EVOLUTION_LAW", "INITIAL_CONDITION"}
    assert ir.state_variables[0].symbol == "x"
    path = write(tmp_path, "ambiguous.tex", r"\[\dot{x} = 0\]" + "\n" + r"\[\dot{y} = 0\]")
    ambiguous = ingest(tmp_path, path)
    assert any(f.trigger == DisambiguationTriggerKind.AMBIGUOUS_STATE_ORDERING for f in ambiguous.disambiguation_flags)
    reviewed = apply_ir_override(ambiguous, IRReviewOverride(state_symbol_order=["y", "x"]))
    assert reviewed.user_approved and reviewed.state_variables[0].symbol == "y"
    assert all(f.resolved for f in reviewed.disambiguation_flags)


def test_override_dimensions_bounds_and_unknown_target(tmp_path):
    from invar_agent.builder.ingestion.schemas import IntervalBoundModel

    ir = ingest(tmp_path, write(tmp_path, "dimension.md", "a = 2 m\nb = 3 m\n$y = a + b$"))
    for patch in (ConstantOverridePatch(constant_id=ir.constants[0].constant_id, raw_unit="s"),
                  ConstantOverridePatch(constant_id=ir.constants[0].constant_id, bounds=IntervalBoundModel(lower=4, upper=1)),
                  ConstantOverridePatch(constant_id="unknown", si_value=3)):
        with pytest.raises(FormalGateError):
            apply_ir_override(ir, IRReviewOverride(constant_patches=[patch]))


def test_reference_registry_word_boundary_gating_and_untrusted_math(tmp_path):
    from invar_agent.research.extractor import lookup_reference_registry

    assert not lookup_reference_registry("nonlorenzian context with no known anchor")
    marker = tmp_path / "must_not_exist"
    path = write(tmp_path, "untrusted.md", f"$y = __import__('pathlib').Path('{marker}').touch()$")
    ir = ingest(tmp_path, path)
    assert not marker.exists()
    assert ir.disambiguation_flags and not ir.equations[0].hin_homogeneous


def test_static_function_signatures_and_macro_math(tmp_path):
    path = write(tmp_path, "kernel.py", "k=2\ndef compute(x: float, *, dt: float = 0.1) -> float:\n    raise RuntimeError('never execute')\n")
    document = parse_document(path)
    assert document.function_signatures == ["compute(x: float, *, dt: float=0.1) -> float"]
    assert "compute(" in ingest(tmp_path, path).summary
    path = write(tmp_path, "macro.tex", r"\ensuremath{k=2}")
    assert parse_document(path).equations[0].text == "k=2"


def test_review_preserves_raw_si_provenance(tmp_path):
    path = write(tmp_path, "rate.csv", "symbol,value,unit\nk,1.2e-6,M/min\n")
    ir = ingest(tmp_path, path)
    reviewed = apply_ir_override(ir, IRReviewOverride())
    assert reviewed.constants[0].raw_value == "1.2e-6"
    assert reviewed.constants[0].raw_unit == "M/min"
    assert reviewed.constants[0].si_value == pytest.approx(2e-5)


def test_fallback_healer_preserves_modules_and_strings():
    source = '# header\nclass A:\n    label = """if not code\nelse not code"""\n    def f(\n        self, x\n    )\n        if x: return x\n        return 0\n'
    healed = json.loads(_fallback.heal_builder_tier0(source))["code"]
    assert "else not code" in healed and "if x: return x" in healed
    assert healed.startswith("# header\nclass A:")
    assert json.loads(_fallback.heal_builder_tier0(healed))["changed"] is False
    source = "# header\nclass A:\n    def f(self, p):\n        return logit(logistic_sigmoid(p)) + 0\n"
    kernel = json.loads(_fallback.heal_builder_tier0(source, "kernel"))["code"]
    assert "return p" in kernel and kernel.startswith("# header\nclass A:")
    assert json.loads(_fallback.heal_builder_tier0(""))["changed"] is False
    with pytest.raises(ValueError, match="PY_SYNTAX_ERROR"):
        _fallback.heal_builder_tier0("def broken(:\n")
