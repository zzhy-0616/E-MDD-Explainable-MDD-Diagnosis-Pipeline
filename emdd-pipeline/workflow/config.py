from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "emdd_default.yaml"
LOCAL_CONFIG = ROOT / "configs" / "emdd_local.yaml"

# 除 paths 外，这些键下的字符串也会相对仓库根目录解析
_PATH_LIKE_KEYS = (
    ("eegpt_encoder", "root"),
    ("eegpt_encoder", "checkpoint"),
    ("matlab_preprocess", "ica_script"),
    ("matlab_preprocess", "executable"),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str | None, root: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_nested_paths(cfg: dict[str, Any], root: Path) -> None:
    for section, key in _PATH_LIKE_KEYS:
        block = cfg.get(section)
        if not isinstance(block, dict):
            continue
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            block[key] = _resolve_path(value, root)


def resolve_paths(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    resolved = copy.deepcopy(cfg)
    paths = resolved.setdefault("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            paths[key] = _resolve_path(value, root)
    _resolve_nested_paths(resolved, root)
    resolved["_root"] = root.resolve()
    return resolved


def load_config(
    config_path: Path | str | None = None,
    local_path: Path | str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    local_path = Path(local_path) if local_path else LOCAL_CONFIG

    if not config_path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if local_path.is_file():
        with local_path.open("r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local_cfg)

    return resolve_paths(cfg, ROOT)
