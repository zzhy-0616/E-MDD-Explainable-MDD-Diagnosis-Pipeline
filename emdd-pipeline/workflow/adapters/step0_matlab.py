from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _serialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def run_step0_matlab(cfg: dict, *, run_dir: Path | None = None) -> None:
    mp = cfg.get("matlab_preprocess", {})
    if not mp.get("enabled", False):
        print("[step0] matlab_preprocess.enabled=false，跳过")
        return

    matlab_exe = mp.get("executable") or shutil.which("matlab")
    if not matlab_exe:
        raise RuntimeError(
            "未找到 MATLAB 可执行文件。请在 configs/emdd_local.yaml 中设置 matlab_preprocess.executable"
        )

    repo_root = cfg["_root"]
    matlab_dir = repo_root / "preprocessing" / "matlab"
    json_path = (run_dir or repo_root / "runs" / "_step0_tmp") / "matlab_preprocess.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_matlab_json(cfg, repo_root)
    json_path.write_text(json.dumps(_serialize(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    batch_cmd = (
        f"addpath('{matlab_dir.as_posix()}'); "
        f"emdd_run_preprocess('{json_path.as_posix()}');"
    )
    cmd = [matlab_exe, "-batch", batch_cmd]
    print("[step0] MATLAB preprocess")
    print("  config:", json_path)
    print("  cmd:", " ".join(cmd[:2]), "...")

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MATLAB preprocess failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if proc.stderr:
        print(proc.stderr)


def _build_matlab_json(cfg: dict, repo_root: Path) -> dict:
    mp = cfg["matlab_preprocess"]
    paths = cfg["paths"]

    ica_script = mp.get("ica_script")
    if ica_script and not Path(ica_script).is_absolute():
        ica_script = str((repo_root / ica_script).resolve())

    return {
        "repo_root": str(repo_root),
        "eeglab_init": mp.get("eeglab_init", "eeglab nogui;"),
        "ica": {
            "enabled": bool(mp.get("run_ica", False)),
            "script": ica_script,
        },
        "epoch_6s": {
            "enabled": bool(mp.get("run_epoch_6s", True)),
            "input_dir": str(paths.get("ica_set_dir", paths.get("raw_eeg"))),
            "output_dir": str(paths["epoch_6_dir"]),
            "window_sec": float(mp.get("epoch_window_sec", 6)),
            "overlap_ratio": float(mp.get("epoch_overlap", 0.5)),
        },
        "split": {
            "enabled": bool(mp.get("run_split", True)),
            "input_dir": str(paths["epoch_6_dir"]),
            "train_dir": str(paths["train_set_dir"]),
            "test_dir": str(paths["test_set_dir"]),
            "train_ratio": float(mp.get("train_ratio", 0.7)),
            "random_seed": int(mp.get("random_seed", 42)),
        },
        "znorm": {
            "enabled": bool(mp.get("run_znorm", False)),
            "input_train": str(paths["train_set_dir"]),
            "input_test": str(paths["test_set_dir"]),
            "output_train": str(paths.get("train_norm_dir", paths["train_set_dir"])),
            "output_test": str(paths.get("test_norm_dir", paths["test_set_dir"])),
        },
    }
