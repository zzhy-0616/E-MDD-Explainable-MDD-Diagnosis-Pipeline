from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepSpec:
    name: str
    title: str
    description: str
    required_inputs: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]
    optional: bool = False


def _artifact(cfg: dict, group: str, key: str) -> str:
    return cfg.get("artifacts", {}).get(group, {}).get(key, "")


def _classifier_file(cfg: dict, key: str) -> Path:
    return cfg["paths"]["step3_dir"] / _artifact(cfg, "classification", key)


def _projector_file(cfg: dict, key: str) -> Path:
    return cfg["paths"]["step4_projector"] / _artifact(cfg, "projector", key)


def build_step_specs(cfg: dict) -> dict[str, StepSpec]:
    paths = cfg["paths"]
    specs: dict[str, StepSpec] = {}

    if cfg.get("matlab_preprocess", {}).get("enabled", False):
        specs["step0"] = StepSpec(
            name="step0",
            title="MATLAB EEG Preprocessing",
            description=(
                "EEGLAB pipeline: optional ICA script, 6s epoching, subject split, "
                "optional channel Z-norm (see preprocessing/matlab/)."
            ),
            required_inputs=(paths.get("ica_set_dir", paths["train_set_dir"]),),
            expected_outputs=(paths["train_set_dir"],),
            optional=False,
        )

    specs.update({
        "step1": StepSpec(
            name="step1",
            title="EEGPT Feature Extraction",
            description="Extract 512-d EEGPT embeddings from ICA-cleaned EEGLAB .set files.",
            required_inputs=(paths["raw_eeg"],),
            expected_outputs=(paths["features"],),
            optional=False,
        ),
        "step2": StepSpec(
            name="step2",
            title="MNE PSD + Asymmetry",
            description="Compute band power and frontal asymmetry aligned with step1 crop indices.",
            required_inputs=(paths["raw_eeg"], paths["features"]),
            expected_outputs=(paths["mne_csv"],),
            optional=False,
        ),
        "step3": StepSpec(
            name="step3",
            title="5-Fold Classification + SHAP Reports",
            description="Train MDD head with StratifiedGroupKFold; export epoch/subject XAI reports.",
            required_inputs=(paths["features"], paths["mne_csv"]),
            expected_outputs=(
                _classifier_file(cfg, "best_fold_test_epochs"),
                _classifier_file(cfg, "checkpoint"),
                paths["step3_test_features"],
            ),
            optional=False,
        ),
        "step4": StepSpec(
            name="step4",
            title="EEG→LLM Projector Training",
            description="Train a 2-layer MLP projector while keeping the LLM frozen.",
            required_inputs=(
                _classifier_file(cfg, "best_fold_train_epochs"),
                _classifier_file(cfg, "best_fold_test_epochs"),
                paths["step3_train_features"],
                paths["step3_test_features"],
                paths["deepseek_model"],
            ),
            expected_outputs=(_projector_file(cfg, "checkpoint"),),
            optional=False,
        ),
        "step5": StepSpec(
            name="step5",
            title="LLM CoT Generation",
            description="Generate explainable chain-of-thought reports for sampled test epochs.",
            required_inputs=(
                _classifier_file(cfg, "best_fold_test_epochs"),
                paths["step3_test_features"],
                _projector_file(cfg, "checkpoint"),
                paths["deepseek_model"],
            ),
            expected_outputs=(paths["step5_output"],),
            optional=True,
        ),
        "eval": StepSpec(
            name="eval",
            title="External Test Evaluation",
            description="Evaluate the saved checkpoint on an external held-out CSV/features set.",
            required_inputs=(_classifier_file(cfg, "checkpoint"),),
            expected_outputs=(paths["eval_output_dir"],),
            optional=True,
        ),
    })
    return specs


def step_is_complete(spec: StepSpec) -> bool:
    if not spec.expected_outputs:
        return False
    ok = True
    for path in spec.expected_outputs:
        if path.is_dir():
            ok = ok and path.is_dir() and any(path.glob("*"))
        else:
            ok = ok and path.exists()
    return ok


def step_inputs_ready(spec: StepSpec) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for path in spec.required_inputs:
        if path.is_dir():
            if not path.exists():
                missing.append(f"directory missing: {path}")
        elif not path.exists():
            missing.append(f"file missing: {path}")
    return len(missing) == 0, missing
