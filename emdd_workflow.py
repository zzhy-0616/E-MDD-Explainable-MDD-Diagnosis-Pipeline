#!/usr/bin/env python3
"""E-MDD pipeline workflow CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.config import load_config
from workflow.runner import WorkflowRunner, steps_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate the E-MDD pipeline: EEGPT encoder → XAI → LLM CoT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List workflow steps and current completion status.")

    check = sub.add_parser("check", help="Validate inputs for selected steps.")
    check.add_argument("--steps", help="Comma-separated steps, e.g. step3,step4,step5")
    check.add_argument("--all", action="store_true", help="Check all steps.")

    run = sub.add_parser("run", help="Execute selected workflow steps.")
    run.add_argument("--steps", help="Comma-separated steps to run.")
    run.add_argument("--from", dest="from_step", help="Run from this step through the end.")
    run.add_argument("--all", action="store_true", help="Run the full pipeline.")
    run.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    run.add_argument("--skip-existing", action="store_true", help="Skip steps whose outputs already exist.")
    run.add_argument("--config", default=None, help="Path to workflow YAML config.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()
    runner = WorkflowRunner(cfg)

    if args.command == "list":
        runner.list_steps()
        return 0

    if args.command == "check":
        selected = steps_from_args(args.all, None, args.steps)
        return runner.check(selected)

    if args.command == "run":
        selected = steps_from_args(args.all, args.from_step, args.steps)
        if selected is None:
            parser.error("Specify --all, --steps, or --from for run.")
        return runner.run(
            selected,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
