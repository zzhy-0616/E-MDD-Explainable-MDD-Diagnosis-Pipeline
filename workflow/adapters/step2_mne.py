from __future__ import annotations

import os
import sys
from pathlib import Path


def run_step2(cfg: dict) -> None:
    root = cfg["_root"]
    paths = cfg["paths"]
    raw_eeg = paths["raw_eeg"]
    features_dir = paths["features"]
    mne_csv = paths["mne_csv"]

    if not raw_eeg.is_dir():
        raise FileNotFoundError(f"Raw EEG directory not found: {raw_eeg}")
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {raw_eeg}")

    sys.path.insert(0, str(root))
    import step2  # noqa: WPS433

    step2.data_path = str(raw_eeg)
    step2.features_dir = str(features_dir)
    mne_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[step2] raw_eeg={raw_eeg}")
    print(f"[step2] features_dir={features_dir}")
    print(f"[step2] output_csv={mne_csv}")

    files = [f for f in os.listdir(step2.data_path) if f.endswith(".set")]
    rows = [step2.process_one_subject(f) for f in files]
    import pandas as pd

    pd.DataFrame(rows).to_csv(mne_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ MNE CSV saved: {mne_csv}")
