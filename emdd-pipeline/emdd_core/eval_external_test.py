import argparse
import json
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


class MDDClassificationHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


def ensure_features_dir(features_dir: str, features_zip: str) -> None:
    if os.path.isdir(features_dir):
        return
    if not os.path.isfile(features_zip):
        raise FileNotFoundError(
            f"Neither features directory nor zip found: {features_dir}, {features_zip}"
        )
    os.makedirs(features_dir, exist_ok=True)
    with zipfile.ZipFile(features_zip, "r") as zf:
        zf.extractall(features_dir)
    print(f"Extracted features zip to: {features_dir}")


def load_test_matrix(test_csv: str, features_dir: str):
    df = pd.read_csv(test_csv)
    x_list = []
    kept_indices = []
    missing = []

    for idx, row in df.iterrows():
        npy_name = f"{row['subject']}.npy"
        npy_path = os.path.join(features_dir, npy_name)
        if not os.path.isfile(npy_path):
            missing.append(npy_name)
            continue
        feat = np.load(npy_path).reshape(-1, 512).mean(0)
        x_list.append(feat)
        kept_indices.append(idx)

    if not x_list:
        raise RuntimeError("No matched test samples found between CSV and feature files.")

    filtered_df = df.loc[kept_indices].reset_index(drop=True)
    x = torch.tensor(np.array(x_list), dtype=torch.float32)
    y = filtered_df["label"].to_numpy().astype(int)
    return filtered_df, x, y, missing


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": float(auc),
        "ACC": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, y_pred


def get_real_subject_id(subject_name: str) -> str:
    parts = str(subject_name).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(subject_name)


def resolve_threshold(manual_threshold, threshold_source_csv: str) -> float:
    if manual_threshold is not None:
        return float(manual_threshold)
    if os.path.isfile(threshold_source_csv):
        src_df = pd.read_csv(threshold_source_csv)
        for col in ("best_threshold", "best_fold_train_threshold"):
            if col in src_df.columns:
                vals = src_df[col].dropna().to_numpy()
                if len(vals) > 0:
                    return float(vals[0])
    return 0.5


def standardize_features(x_np: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-6, 1.0, std)
    return ((x_np - mean) / safe_std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate external test set with trained best model."
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/classification/best_classifier.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--test-csv",
        default="subject_power_with_asym_test.csv",
        help="External test CSV path.",
    )
    parser.add_argument(
        "--features-dir",
        default="features_test",
        help="Directory containing <subject>.npy files for test set.",
    )
    parser.add_argument(
        "--features-zip",
        default="features_test.zip",
        help="Zip file for test features; auto-extracted if features-dir does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/external_eval",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold. 默认：ckpt 的 best_threshold（与 step3_classify_shap 最优折一致），否则 CSV 列 best_threshold。",
    )
    parser.add_argument(
        "--threshold-source-csv",
        default="artifacts/classification/best_fold_train_subjects.csv",
        help="Fallback CSV：列 best_threshold（或旧列 best_fold_train_threshold）。",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ensure_features_dir(args.features_dir, args.features_zip)
    test_df, x_test, y_test, missing = load_test_matrix(args.test_csv, args.features_dir)

    try:
        state = torch.load(args.model_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        # PyTorch>=2.6 defaults to weights_only=True, which may reject trusted
        # checkpoints containing numpy arrays (e.g., feature_mean/std).
        state = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        ckpt_state_dict = state["model_state_dict"]
    else:
        ckpt_state_dict = state
    hidden_dim = int(ckpt_state_dict["net.0.weight"].shape[0])
    model = MDDClassificationHead(hidden_dim=hidden_dim).to(device)
    ckpt_threshold = None
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(ckpt_state_dict)
        if "feature_mean" in state and "feature_std" in state:
            x_np = x_test.cpu().numpy()
            x_np = standardize_features(
                x_np,
                np.asarray(state["feature_mean"], dtype=np.float32),
                np.asarray(state["feature_std"], dtype=np.float32),
            )
            x_test = torch.tensor(x_np, dtype=torch.float32)
        if "best_threshold" in state:
            ckpt_threshold = float(state["best_threshold"])
        elif "decision_threshold" in state:
            ckpt_threshold = float(state["decision_threshold"])
    else:
        model.load_state_dict(ckpt_state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(x_test.to(device))
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    if args.threshold is not None:
        decision_threshold = float(args.threshold)
    elif ckpt_threshold is not None:
        decision_threshold = ckpt_threshold
    else:
        decision_threshold = resolve_threshold(args.threshold, args.threshold_source_csv)
    metrics, y_pred = compute_metrics(y_test, y_prob, decision_threshold)

    pred_df = test_df.copy()
    pred_df["pred_prob_mdd"] = y_prob
    pred_df["pred_label"] = y_pred
    pred_df["real_subject_id"] = pred_df["subject"].apply(get_real_subject_id)
    pred_path = os.path.join(args.output_dir, "external_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # Subject-level majority-vote ACC (vote on epoch predicted labels).
    subject_rows = []
    for sid, g in pred_df.groupby("real_subject_id"):
        true_label = int(g["label"].mode().iloc[0])
        votes_pos = int((g["pred_label"] == 1).sum())
        votes_neg = int((g["pred_label"] == 0).sum())
        pred_label_subject = 1 if votes_pos > votes_neg else 0
        # Tie-breaking: fall back to mean probability.
        if votes_pos == votes_neg:
            pred_label_subject = int(g["pred_prob_mdd"].mean() >= decision_threshold)
        subject_rows.append(
            {
                "real_subject_id": sid,
                "label": true_label,
                "pred_label_subject": pred_label_subject,
                "mean_pred_prob_mdd": float(g["pred_prob_mdd"].mean()),
                "n_epochs": int(len(g)),
            }
        )
    subject_pred_df = pd.DataFrame(subject_rows).sort_values("real_subject_id")
    subject_acc = accuracy_score(
        subject_pred_df["label"].to_numpy(), subject_pred_df["pred_label_subject"].to_numpy()
    )
    subject_pred_path = os.path.join(args.output_dir, "external_test_subject_predictions.csv")
    subject_pred_df.to_csv(subject_pred_path, index=False, encoding="utf-8-sig")

    metrics["missing_feature_files"] = int(len(missing))
    metrics["subject_level_ACC_majority_vote"] = float(subject_acc)
    metrics["n_subjects"] = int(len(subject_pred_df))
    metrics_path = os.path.join(args.output_dir, "external_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save confusion matrix figure
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig_cm.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig_cm)

    # Save ROC curve figure
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label="Chance")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    fig_roc.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    if missing:
        miss_path = os.path.join(args.output_dir, "missing_feature_files.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))

    print("External test evaluation done.")
    print(f"AUC = {metrics['AUC']:.4f}")
    print(f"ACC = {metrics['ACC']:.4f}")
    print(f"Precision = {metrics['Precision']:.4f}")
    print(f"Recall = {metrics['Recall']:.4f}")
    print(f"F1 = {metrics['F1']:.4f}")
    print(f"Sensitivity = {metrics['Sensitivity']:.4f}")
    print(f"Specificity = {metrics['Specificity']:.4f}")
    print(f"Threshold = {metrics['threshold']:.4f}")
    print(f"Subject-level ACC (majority vote) = {metrics['subject_level_ACC_majority_vote']:.4f}")
    print(f"n_samples = {metrics['n_samples']}")
    print(f"n_subjects = {metrics['n_subjects']}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved subject predictions to: {subject_pred_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")


if __name__ == "__main__":
    main()
