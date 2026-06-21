from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from workflow.adapters.step0_matlab import run_step0_matlab
from workflow.adapters.step1_extract import run_step1
from workflow.adapters.step2_mne import run_step2
from workflow.artifacts import StepSpec, build_step_specs, step_inputs_ready, step_is_complete
from workflow.config import ROOT, load_config

STEP_ORDER = ("step0", "step1", "step2", "step3", "step4", "step5", "eval")


class WorkflowRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.specs = build_step_specs(cfg)
        self.run_dir: Path | None = None

    def list_steps(self) -> None:
        print("\nE-MDD Pipeline — Workflow Steps\n" + "=" * 60)
        for name in self._ordered_step_names():
            spec = self.specs[name]
            status = "DONE" if step_is_complete(spec) else "PENDING"
            optional = " (optional)" if spec.optional else ""
            print(f"\n[{status}] {name}{optional}: {spec.title}")
            print(f"  {spec.description}")
            ready, missing = step_inputs_ready(spec)
            if not ready and not step_is_complete(spec):
                print("  Missing inputs:")
                for item in missing[:5]:
                    print(f"    - {item}")

    def check(self, steps: Iterable[str] | None = None) -> int:
        selected = self._normalize_steps(steps)
        errors = 0
        print("\nWorkflow preflight check\n" + "=" * 60)
        for name in selected:
            spec = self.specs[name]
            print(f"\n{name}: {spec.title}")
            if step_is_complete(spec):
                print("  outputs: OK (already complete)")
                continue
            ready, missing = step_inputs_ready(spec)
            if ready:
                print("  inputs: OK")
            else:
                errors += 1
                print("  inputs: MISSING")
                for item in missing:
                    print(f"    - {item}")
        return errors

    def run(
        self,
        steps: Iterable[str] | None = None,
        *,
        dry_run: bool = False,
        skip_existing: bool = False,
    ) -> int:
        selected = self._normalize_steps(steps)
        self.run_dir = self._create_run_dir()
        self._write_run_metadata(selected, dry_run=dry_run)

        print(f"\nRun directory: {self.run_dir}")
        print("Steps:", ", ".join(selected))
        if dry_run:
            print("\n[DRY RUN] No commands will be executed.\n")

        failures = 0
        for name in selected:
            spec = self.specs[name]
            if skip_existing and step_is_complete(spec):
                print(f"\n[SKIP] {name} (outputs already exist)")
                continue

            if dry_run and step_is_complete(spec):
                print(f"\n[DRY RUN] {name}: outputs already exist (would re-run)")
                self._print_planned_command(name)
                continue

            ready, missing = step_inputs_ready(spec)
            if not ready:
                print(f"\n[FAIL] Cannot run {name}; missing inputs:")
                for item in missing:
                    print(f"   - {item}")
                failures += 1
                if not spec.optional:
                    break
                continue

            print(f"\n{'=' * 60}\n>> {name}: {spec.title}\n{'=' * 60}")
            if dry_run:
                self._print_planned_command(name)
                continue

            started = time.time()
            log_path = self.run_dir / "logs" / f"{name}.log"
            try:
                self._execute_step(name, log_path)
                elapsed = time.time() - started
                print(f"[OK] {name} finished in {elapsed:.1f}s")
                self._append_run_meta(name, "success", elapsed, log_path)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.time() - started
                print(f"[FAIL] {name} failed: {exc}")
                self._append_run_meta(name, "failed", elapsed, log_path, error=str(exc))
                failures += 1
                if not spec.optional:
                    break

        if failures:
            print(f"\nWorkflow finished with {failures} failure(s).")
            return failures
        print("\nWorkflow finished successfully.")
        return 0

    def _ordered_step_names(self) -> list[str]:
        return [name for name in STEP_ORDER if name in self.specs]

    def _normalize_steps(self, steps: Iterable[str] | None) -> list[str]:
        if steps is None:
            return self._ordered_step_names()
        normalized: list[str] = []
        for item in steps:
            for part in item.split(","):
                part = part.strip()
                if part:
                    normalized.append(part)
        unknown = [s for s in normalized if s not in self.specs]
        if unknown:
            available = list(self.specs.keys())
            raise ValueError(f"Unknown steps: {unknown}. Valid: {available}")
        return [s for s in STEP_ORDER if s in normalized]

    def _create_run_dir(self) -> Path:
        runs_dir = self.cfg["paths"]["runs_dir"]
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / stamp
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_run_metadata(self, steps: list[str], *, dry_run: bool) -> None:
        assert self.run_dir is not None
        resolved_cfg = self._serialize_cfg(self.cfg)
        (self.run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (self.run_dir / "command.txt").write_text(
            " ".join(sys.argv),
            encoding="utf-8",
        )
        env_lines = [
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"cwd={Path.cwd()}",
            f"dry_run={dry_run}",
            f"steps={','.join(steps)}",
        ]
        (self.run_dir / "environment.txt").write_text("\n".join(env_lines), encoding="utf-8")
        meta = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps_requested": steps,
            "dry_run": dry_run,
            "step_results": [],
        }
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_run_meta(
        self,
        step: str,
        status: str,
        elapsed: float,
        log_path: Path,
        *,
        error: str | None = None,
    ) -> None:
        assert self.run_dir is not None
        meta_path = self.run_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry = {
            "step": step,
            "status": status,
            "elapsed_sec": round(elapsed, 2),
            "log": str(log_path.relative_to(self.run_dir)),
        }
        if error:
            entry["error"] = error
        meta["step_results"].append(entry)
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def _serialize_cfg(self, cfg: dict) -> dict:
        out: dict = {}
        for key, value in cfg.items():
            if key == "_root":
                out[key] = str(value)
            elif isinstance(value, dict):
                out[key] = self._serialize_cfg(value)
            elif isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out

    def _print_planned_command(self, name: str) -> None:
        scripts = self.cfg.get("scripts", {})
        if name in {"step1", "step2"}:
            print(f"  -> workflow adapter: {name}")
            return
        if name == "step0":
            print("  -> MATLAB: preprocessing/matlab/emdd_run_preprocess.m")
            return
        script = scripts.get(name)
        if script:
            print(f"  -> python {script}")

    def _execute_step(self, name: str, log_path: Path) -> None:
        if name == "step0":
            assert self.run_dir is not None
            self._run_with_log(lambda c: run_step0_matlab(c, run_dir=self.run_dir), (self.cfg,), log_path)
            return
        if name == "step1":
            self._run_with_log(run_step1, (self.cfg,), log_path)
            return
        if name == "step2":
            self._run_with_log(run_step2, (self.cfg,), log_path)
            return
        if name == "step3":
            self._run_script(self.cfg["scripts"]["step3"], cwd=self.cfg["paths"]["step3_root"], log_path=log_path)
            return
        if name == "step4":
            self._ensure_deepseek_links()
            self._run_script(self.cfg["scripts"]["step4"], cwd=self.cfg["paths"]["step4_root"], log_path=log_path)
            return
        if name == "step5":
            self._ensure_deepseek_links()
            self._run_script(self.cfg["scripts"]["step5"], cwd=self.cfg["paths"]["step5_root"], log_path=log_path)
            return
        if name == "eval":
            self._run_eval(log_path)
            return
        raise ValueError(f"Unsupported step: {name}")

    def _run_script(self, script_rel: str | Path, *, cwd: Path, log_path: Path) -> None:
        script = (self.cfg["_root"] / script_rel).resolve()
        if not script.is_file():
            raise FileNotFoundError(f"Script not found: {script}")
        cmd = [sys.executable, str(script)]
        self._run_subprocess(cmd, cwd=cwd, log_path=log_path)

    def _classifier_artifact(self, key: str) -> Path:
        arts = self.cfg.get("artifacts", {}).get("classification", {})
        return self.cfg["paths"]["step3_dir"] / arts[key]

    def _run_eval(self, log_path: Path) -> None:
        paths = self.cfg["paths"]
        eval_cfg = self.cfg.get("eval", {})
        cmd = [
            sys.executable,
            str((self.cfg["_root"] / self.cfg["scripts"]["eval"]).resolve()),
            "--model-path",
            str(self._classifier_artifact("checkpoint")),
            "--test-csv",
            str(eval_cfg.get("test_csv", paths.get("external_test_csv", ""))),
            "--features-dir",
            str(eval_cfg.get("features_dir", paths.get("external_test_features", ""))),
            "--output-dir",
            str(paths["eval_output_dir"]),
            "--threshold-source-csv",
            str(self._classifier_artifact("best_fold_test_subjects")),
        ]
        features_zip = eval_cfg.get("features_zip")
        if features_zip:
            cmd.extend(["--features-zip", str(features_zip)])
        self._run_subprocess(cmd, cwd=paths["step3_root"], log_path=log_path)

    def _ensure_deepseek_links(self) -> None:
        target = self.cfg["paths"]["deepseek_model"]
        if not target.is_dir():
            raise FileNotFoundError(
                f"DeepSeek model directory not found: {target}\n"
                "Place local LLM weights there or override paths.deepseek_model in configs/emdd_local.yaml"
            )
        for link_parent in (self.cfg["paths"]["step4_root"], self.cfg["paths"]["step5_root"]):
            link = link_parent / "deepseek_model"
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(target, target_is_directory=True)
                print(f"[LINK] Linked {link} -> {target}")
            except OSError:
                if os.name == "nt":
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                        check=True,
                    )
                    print(f"[LINK] Junction {link} -> {target}")
                else:
                    raise

    def _run_with_log(self, func, args: tuple, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            class Tee:
                def write(self, data):
                    old_stdout.write(data)
                    log_file.write(data)
                def flush(self):
                    old_stdout.flush()
                    log_file.flush()
            sys.stdout = Tee()
            sys.stderr = Tee()
            try:
                func(*args)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def _run_subprocess(self, cmd: list[str], *, cwd: Path, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print("Command:", " ".join(cmd))
        print("CWD:", cwd)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}\n"
                f"See log: {log_path}\n\n--- log tail ---\n{tail}"
            )


def steps_from_args(all_flag: bool, from_step: str | None, steps: str | None) -> list[str] | None:
    if steps:
        return [s.strip() for s in steps.split(",") if s.strip()]
    if from_step:
        if from_step not in STEP_ORDER:
            raise ValueError(f"Unknown --from step: {from_step}")
        start = STEP_ORDER.index(from_step)
        return list(STEP_ORDER[start:])
    if all_flag:
        return list(STEP_ORDER)
    return None
