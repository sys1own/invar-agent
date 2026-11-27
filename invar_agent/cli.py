"""invar-agent command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_nodes(path: str):
    from invar_agent.server import _node_from_dict, _patch_from_dict

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [_node_from_dict(n) for n in data.get("nodes", [])]
    patches = {k: [_patch_from_dict(p) for p in v] for k, v in data.get("patches_by_node", {}).items()}
    return nodes, patches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invar-agent", description="Mathematically Verified Autonomous Agent Kernel")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Code-repair kernel: ingest a repository graph")
    p_ingest.add_argument("path", nargs="?", default=".")

    p_plan = sub.add_parser("plan", help="Code-repair kernel: plan wavefronts for nodes JSON")
    p_plan.add_argument("workspace")
    p_plan.add_argument("nodes_json")

    p_exec = sub.add_parser("execute", help="Code-repair kernel: execute the 9-stage kernel over nodes JSON")
    p_exec.add_argument("workspace")
    p_exec.add_argument("nodes_json")
    p_exec.add_argument("--objective", default="")
    p_exec.add_argument("--sandbox-cmd", nargs="*", default=None)

    p_export = sub.add_parser("export", help="Code-repair kernel: export a curated artifact")
    p_export.add_argument("workspace")
    p_export.add_argument("type", choices=["workspace_zip", "patch_adjoint_bundle", "verification_certificate"])
    p_export.add_argument("--out", default=None)

    p_builder = sub.add_parser("builder", help="Builder Studio: ingest, plan, execute, and export scientific instruments")
    b_sub = p_builder.add_subparsers(dest="builder_command", required=True)

    pb_ingest = b_sub.add_parser("ingest", help="Ingest uploaded specs and/or verified vault run IDs into a CanonicalResearchIR")
    pb_ingest.add_argument("workspace", help="Workspace root directory")
    pb_ingest.add_argument("--upload", dest="uploads", action="append", default=[], help="Spec/data file to ingest (repeatable)")
    pb_ingest.add_argument("--vault-id", dest="vault_ids", action="append", default=[], help="Verified Research Vault run_id (repeatable)")
    pb_ingest.add_argument("--title", default="Scientific instrument")
    pb_ingest.add_argument("--out", default=None, help="Optional path to write CanonicalResearchIR JSON")

    pb_approve = b_sub.add_parser("approve", help="Approve an ingested CanonicalResearchIR JSON for planning and execution")
    pb_approve.add_argument("workspace")
    pb_approve.add_argument("ir_json")
    pb_approve.add_argument("--out", default=None, help="Output path (defaults to overwriting ir_json)")

    pb_override = b_sub.add_parser("override", help="Apply an IRReviewOverride JSON patch to an ingested IR")
    pb_override.add_argument("workspace")
    pb_override.add_argument("ir_json")
    pb_override.add_argument("override_json")
    pb_override.add_argument("--out", default=None, help="Output path (defaults to overwriting ir_json)")

    pb_plan = b_sub.add_parser("plan", help="Synthesize a verified Builder UAST & wavefront plan from an approved IR JSON")
    pb_plan.add_argument("workspace")
    pb_plan.add_argument("ir_json")
    pb_plan.add_argument("--goal", default="")
    pb_plan.add_argument("--profile", dest="profiles", action="append", default=None)
    pb_plan.add_argument("--out", default=None)

    pb_exec = b_sub.add_parser("execute", help="Execute a planned Builder UAST build in the isolated workspace")
    pb_exec.add_argument("workspace")
    pb_exec.add_argument("ir_json")
    pb_exec.add_argument("plan_json")
    pb_exec.add_argument("--build-id", default=None)

    pb_export = b_sub.add_parser("export", help="Export a receipt-verified Builder .zip archive for a COMPLETED build")
    pb_export.add_argument("workspace")
    pb_export.add_argument("build_id")
    pb_export.add_argument("--out", default=None)

    p_serve = sub.add_parser("serve", help="Launch the web console")
    p_serve.add_argument("workspace", nargs="?", default=None,
                         help="Project workspace directory (default: $INVAR_WORKSPACE_DIR or ./workspace)")
    p_serve.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                         help="Project workspace directory (overrides positional workspace and "
                              "$INVAR_WORKSPACE_DIR)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--provider", default=None,
                         help="LLM provider (gemini, openai, deepseek, openrouter); "
                              "overrides env detection and the saved config")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        from invar_agent.context.bundle_repo import WorkspaceIngestor

        report = WorkspaceIngestor(args.path).ingest_repository_graph()
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "plan":
        from invar_agent.core.agent import VerifiedAutonomousAgent

        agent = VerifiedAutonomousAgent(args.workspace)
        nodes, _ = _load_nodes(args.nodes_json)
        wavefronts = agent.plan_wavefronts(nodes)
        print(json.dumps({"wavefronts": wavefronts}, indent=2))
        return 0

    if args.command == "execute":
        from invar_agent.core.agent import VerifiedAutonomousAgent
        from invar_agent.server import _report_to_dict

        agent = VerifiedAutonomousAgent(args.workspace)
        nodes, patches = _load_nodes(args.nodes_json)
        report = agent.run_until_verified(args.objective, nodes, patches_by_node=patches,
                                          sandbox_cmd=args.sandbox_cmd)
        print(json.dumps(_report_to_dict(report), indent=2))
        return 0 if report.success else 1

    if args.command == "export":
        from invar_agent.export.exporter import ExportEngine

        engine = ExportEngine(args.workspace)
        artifact = engine.export(args.type, report={})
        out = Path(args.out) if args.out else Path(artifact.filename)
        out.write_bytes(artifact.data)
        print(f"wrote {out}")
        return 0

    if args.command == "builder":
        import asyncio
        import zipfile
        from uuid import uuid4

        from invar_agent.builder.his_cache import HISCache, blake3_file
        from invar_agent.builder.ingestion.pipeline import CanonicalResearchIngestionPipeline
        from invar_agent.builder.ingestion.schemas import CanonicalResearchIR, IngestionRequest
        from invar_agent.builder.planner import BuilderPlanner
        from invar_agent.builder.state_machine import BuilderStateMachine, BuildRequest
        from invar_agent.builder.uast_models import BuilderProfile, BuilderUAST

        ws = Path(args.workspace).resolve()
        if args.builder_command == "ingest":
            pipeline = CanonicalResearchIngestionPipeline(ws)
            ir = pipeline.ingest(
                IngestionRequest(
                    upload_paths=[str(Path(p).resolve()) for p in args.uploads],
                    vault_run_ids=list(args.vault_ids),
                    title=args.title,
                )
            )
            payload = ir.model_dump_json(indent=2)
            if args.out:
                Path(args.out).write_text(payload, encoding="utf-8")
            print(payload)
            return 0
        if args.builder_command == "approve":
            from invar_agent.builder.ingestion.pipeline import apply_ir_override
            from invar_agent.builder.ingestion.schemas import IRReviewOverride

            ir_path = Path(args.ir_json)
            ir = CanonicalResearchIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
            approved_ir = apply_ir_override(ir, IRReviewOverride(approve=True))
            payload = approved_ir.model_dump_json(indent=2)
            out_path = Path(args.out) if args.out else ir_path
            out_path.write_text(payload, encoding="utf-8")
            print(payload)
            return 0
        if args.builder_command == "override":
            from invar_agent.builder.ingestion.pipeline import apply_ir_override
            from invar_agent.builder.ingestion.schemas import IRReviewOverride

            ir_path = Path(args.ir_json)
            ir = CanonicalResearchIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
            ov = IRReviewOverride.model_validate_json(Path(args.override_json).read_text(encoding="utf-8"))
            patched_ir = apply_ir_override(ir, ov)
            payload = patched_ir.model_dump_json(indent=2)
            out_path = Path(args.out) if args.out else ir_path
            out_path.write_text(payload, encoding="utf-8")
            print(payload)
            return 0
        if args.builder_command == "plan":
            ir = CanonicalResearchIR.model_validate_json(Path(args.ir_json).read_text(encoding="utf-8"))
            profiles = [BuilderProfile(p) for p in args.profiles] if args.profiles else None
            planner = BuilderPlanner()
            uast, schedule, arch_spec = planner.plan(ir, build_goal=args.goal, profiles=profiles)
            plan_obj = {
                "ir_id": ir.ir_id,
                "ir_blake3_digest": ir.ir_blake3_digest,
                "uast": uast.model_dump(mode="json"),
                "wavefront_schedule": schedule.model_dump(mode="json"),
                "arch_spec": arch_spec.model_dump(mode="json"),
            }
            payload = json.dumps(plan_obj, indent=2)
            if args.out:
                Path(args.out).write_text(payload, encoding="utf-8")
            print(payload)
            return 0
        if args.builder_command == "execute":
            ir = CanonicalResearchIR.model_validate_json(Path(args.ir_json).read_text(encoding="utf-8"))
            plan_obj = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
            uast = BuilderUAST.model_validate(plan_obj["uast"])
            bid = args.build_id or uuid4().hex[:12]
            machine = BuilderStateMachine(ws)
            result = asyncio.run(machine.run(BuildRequest(build_id=bid, ir=ir, uast=uast)))
            print(result.model_dump_json(indent=2))
            return 0 if result.status in {"COMPLETED", "COMPLETED_WITH_FALLBACKS"} else 1
        if args.builder_command == "export":
            from invar_agent.builder.export_utils import ExportBlockedError, build_verified_export_zip

            build_dir = ws / ".invar" / "builds" / args.build_id
            receipt_path = build_dir / "build_receipt.json"
            build_status = None
            if receipt_path.is_file():
                try:
                    build_status = json.loads(receipt_path.read_text(encoding="utf-8")).get("status")
                except Exception:
                    build_status = None
            workspace = (build_dir / "workspace").resolve()
            cache_manifest = HISCache(ws, args.build_id, workspace).load() if workspace.is_dir() else None
            out = Path(args.out) if args.out else Path(f"{args.build_id}.zip")
            try:
                build_verified_export_zip(
                    workspace,
                    out,
                    build_status=build_status,
                    cache_manifest=cache_manifest,
                )
            except ExportBlockedError as exc:
                print(json.dumps({"error": exc.message, "code": exc.code, "build_id": args.build_id}))
                return 1
            print(f"wrote {out}")
            return 0
        return 2

    if args.command == "serve":
        import uvicorn

        from invar_agent.server import create_app

        app = create_app(args.workspace_dir or args.workspace, provider=args.provider)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
