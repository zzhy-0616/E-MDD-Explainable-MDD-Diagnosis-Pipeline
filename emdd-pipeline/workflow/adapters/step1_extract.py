from __future__ import annotations

import os
import sys
from pathlib import Path


def run_step1(cfg: dict) -> None:
    root = cfg["_root"]
    paths = cfg["paths"]
    raw_eeg = paths["raw_eeg"]
    features_dir = paths["features"]
    features_dir.mkdir(parents=True, exist_ok=True)

    if not raw_eeg.is_dir():
        raise FileNotFoundError(f"Raw EEG directory not found: {raw_eeg}")

    sys.path.insert(0, str(root))
    import step1  # noqa: WPS433

    step1.DEFAULT_FEATURES_DIR = str(features_dir)
    eegpt_root = cfg.get("eegpt_encoder", {}).get("root") or cfg.get("eegpt", {}).get("root")
    if eegpt_root:
        step1.EEGPT_ROOT = str(Path(eegpt_root).resolve())
        step1.EEGPT_DOWNSTREAM_TUEG = str(Path(eegpt_root).resolve() / "downstream_tueg")
    ckpt = cfg.get("eegpt_encoder", {}).get("checkpoint") or cfg.get("eegpt", {}).get("checkpoint")
    if ckpt:
        step1.DEFAULT_CKPT_PATH = str(Path(ckpt).resolve())

    print(f"[step1] raw_eeg={raw_eeg}")
    print(f"[step1] features_dir={features_dir}")
    step1.process_folder(str(raw_eeg), save_folder=str(features_dir))
