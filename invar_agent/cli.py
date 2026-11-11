"""invar-agent command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_nodes(path: str):
    from invar_agent.server import _node_from_dict, _patch_from_dict

    data = json.loads(Path(path).read_text())
    nodes = [_node_from_dict(n) for n in data.get("nodes", [])]
    patches = {k: [_patch_from_dict(p) for p in v] for k, v in data.get("patches_by_node", {}).items()}
    return nodes, patches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invar-agent", description="Mathematically Verified Autonomous Agent Kernel")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a workspace")
    p_ingest.add_argument("path", nargs="?", default=".")

    p_plan = sub.add_parser("plan", help="Plan wavefronts for nodes JSON")
    p_plan.add_argument("workspace")
    p_plan.add_argument("nodes_json")

    p_exec = sub.add_parser("execute", help="Execute the 9-stage kernel over nodes JSON")
    p_exec.add_argument("workspace")
    p_exec.add_argument("nodes_json")
    p_exec.add_argument("--objective", default="")
    p_exec.add_argument("--sandbox-cmd", nargs="*", default=None)

    p_export = sub.add_parser("export", help="Export a curated artifact")
    p_export.add_argument("workspace")
    p_export.add_argument("type", choices=["workspace_zip", "patch_adjoint_bundle", "verification_certificate"])
    p_export.add_argument("--out", default=None)

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

    if args.command == "serve":
        import uvicorn

        from invar_agent.server import create_app

        app = create_app(args.workspace_dir or args.workspace, provider=args.provider)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
