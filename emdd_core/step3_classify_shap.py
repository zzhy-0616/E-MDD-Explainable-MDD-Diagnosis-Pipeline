import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import shap
from scipy.stats import pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 基础配置
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(_SCRIPT_DIR, "features")
MNE_CSV_PATH = os.path.join(_SCRIPT_DIR, "subject_power_with_asym.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "artifacts", "classification")
DIR_STEP3_TRAIN = os.path.join(_SCRIPT_DIR, "features", "fold_train")
DIR_STEP3_TEST = os.path.join(_SCRIPT_DIR, "features", "fold_val")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS = 100
N_FOLDS = 5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
CLINICAL_COLS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'delta_asym', 'theta_asym', 'alpha_asym']
# 可视化：取 df 中行索引对应的一个 epoch，画出 512 维有符号 SHAP（解释 MDD 类输出）
SHAP_VIZ_EPOCH_IDX = 0


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_standardizer(x_np: np.ndarray):
    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x_np: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x_np - mean) / std).astype(np.float32)

# ==========================================
# 模型组件
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0

class MDDClassificationHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        if x.dim() == 3: x = x.mean(dim=1)
        return self.net(x)

def find_best_accuracy_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    # 用唯一概率值作为候选 cut-off，选择 train accuracy 最高者
    candidates = np.unique(y_prob)
    best_thr = 0.5
    best_acc = -1.0
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        acc = float((pred == y_true).mean())
        if (acc > best_acc) or (acc == best_acc and thr < best_thr):
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def compute_subject_level_acc(subject_ids, y_true, y_pred):
    sub_df = pd.DataFrame(
        {
            "sid": np.asarray(subject_ids),
            "y_true": np.asarray(y_true).astype(int),
            "y_pred": np.asarray(y_pred).astype(int),
        }
    )
    rows = []
    for sid, g in sub_df.groupby("sid"):
        true_label = int(g["y_true"].mode().iloc[0])
        votes_pos = int((g["y_pred"] == 1).sum())
        votes_neg = int((g["y_pred"] == 0).sum())
        pred_label = 1 if votes_pos > votes_neg else 0
        rows.append((sid, true_label, pred_label))
    if not rows:
        return 0.0
    arr = np.array(rows, dtype=object)
    return float((arr[:, 1].astype(int) == arr[:, 2].astype(int)).mean())


def find_threshold_with_inner_cv(y_train, train_prob, train_subject_ids, random_seed=42):
    y_train = np.asarray(y_train).astype(int)
    train_prob = np.asarray(train_prob).astype(float)
    train_subject_ids = np.asarray(train_subject_ids)

    unique_subjects = np.unique(train_subject_ids)
    if len(unique_subjects) < 3:
        thr, _ = find_best_accuracy_threshold(y_train, train_prob)
        return float(thr)

    n_splits_inner = min(3, len(unique_subjects))
    inner_cv = StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=random_seed
    )
    threshold_grid = np.linspace(0.05, 0.95, 91)

    best_thr = 0.5
    best_score = -1.0
    for thr in threshold_grid:
        fold_scores = []
        for _, iv_idx in inner_cv.split(train_prob, y_train, groups=train_subject_ids):
            y_iv = y_train[iv_idx]
            pred_iv = (train_prob[iv_idx] >= thr).astype(int)
            sid_iv = train_subject_ids[iv_idx]
            fold_scores.append(compute_subject_level_acc(sid_iv, y_iv, pred_iv))
        score = float(np.mean(fold_scores)) if fold_scores else -1.0
        # 平局时优先更接近 0.5 的阈值，降低过拟合风险
        if (score > best_score) or (
            score == best_score and abs(thr - 0.5) < abs(best_thr - 0.5)
        ):
            best_score = score
            best_thr = float(thr)
    return best_thr

# ==========================================
# SHAP 之后：各 EEGPT 维度与 PSD（临床频段）逐列 Pearson r、p；另返回 best_|r| 映射供报告
# ==========================================
def build_dim_psd_correlation_table(X_np: np.ndarray, df: pd.DataFrame, train_mask: np.ndarray, clinical_cols):
    mapping_dict = {}
    rows = []
    for d in range(X_np.shape[1]):
        x_d = X_np[train_mask, d].astype(np.float64)
        rs, ps = [], []
        row = {"dim": d}
        for col in clinical_cols:
            r, p = pearsonr(x_d, df[col].values[train_mask].astype(np.float64))
            rs.append(r)
            ps.append(p)
            row[f"{col}_r"] = r
            row[f"{col}_p"] = p
        idx = int(np.argmax(np.abs(rs)))
        row["best_psd_col"] = clinical_cols[idx]
        row["best_r"] = rs[idx]
        row["best_p"] = ps[idx]
        rows.append(row)
        mapping_dict[d] = {"col": clinical_cols[idx], "r": rs[idx], "p": ps[idx]}
    return pd.DataFrame(rows), mapping_dict

# ==========================================
# 可视化：单个 epoch 下 512 维 SHAP（有符号，对应 MDD 类输出）
# ==========================================
def save_shap_signed_full_dim_plot(
    shap_vec: np.ndarray,
    out_dir: str,
    filename: str,
    title_prefix: str,
):
    shap_vec = np.asarray(shap_vec, dtype=np.float64).ravel()
    n = shap_vec.shape[0]
    dims = np.arange(n)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = np.where(shap_vec >= 0.0, "#e74c3c", "#2980b9")
    ax.bar(dims, shap_vec, width=1.0, color=colors, edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("EEGPT dimension index")
    ax.set_ylabel("SHAP value (MDD class)")
    ax.set_title(title_prefix + "\n512-d SHAP per dim (signed; red≥0, blue<0)", fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 单 Epoch 报告：按真实标签选 SHAP 正负侧，再按 |SHAP| 取 top3（MDD→SHAP>0，HC→SHAP<0）
# ==========================================
def generate_epoch_report(row, shap_value, mapping_dict, threshold_display: float):
    prob = float(row["eegpt_mdd_prob"])
    label = int(row["label"])
    abs_sorted_dims = np.argsort(np.abs(shap_value))[::-1]
    # 类 1 = MDD：SHAP 针对 MDD _logits；MDD 样本关注推高 MDD 的维度（SHAP>0）；HC 样本关注压低 MDD 的维度（SHAP<0）
    if label == 1:
        candidates = [int(d) for d in abs_sorted_dims if shap_value[d] > 0]
    else:
        candidates = [int(d) for d in abs_sorted_dims if shap_value[d] < 0]

    top3_dims = candidates[:3]

    while len(top3_dims) < 3:
        for d in np.argsort(np.abs(shap_value))[::-1]:
            d = int(d)
            if d not in top3_dims:
                top3_dims.append(d)
                break

    evi = []
    for d in top3_dims:
        w = shap_value[d]
        info = mapping_dict[d]
        evi.append(f"{info['col']}(dim{d}, r={info['r']:.3f}, p={info['p']:.3f}, SHAP={w:.3f})")

    power = f"δ={row['delta']:.2f} θ={row['theta']:.2f} α={row['alpha']:.2f} β={row['beta']:.2f} γ={row['gamma']:.2f}"
    asym = f"δ_asym={row['delta_asym']:.2f} θ_asym={row['theta_asym']:.2f} α_asym={row['alpha_asym']:.2f}"
    lab_txt = "MDD" if label == 1 else "HC"

    report = (
        f"[Epoch报告] label={lab_txt} | MDDprob={prob:.4f} (阈值参考={threshold_display:.4f})\n"
        f"TOP3：按标签取 SHAP 侧（MDD:SHAP>0 / HC:SHAP<0）后按 |SHAP| 最大\n"
        f"频段：{power}\n"
        f"不对称：{asym}\n"
        f"TOP3特征（PSD 相关为最优折 train 被试上 dim~PSD）：{' | '.join(evi)}"
    )
    return report, top3_dims

# ==========================================
# 被试最终报告：对该被试全部 epoch 的 top3 做维度频数，取 top5
# ==========================================
def generate_subject_report(subject_df, mapping_dict):
    mean_prob = float(subject_df["eegpt_mdd_prob"].mean())
    vote = defaultdict(int)
    for dims in subject_df["top3_dims_epoch"]:
        for d in dims:
            vote[int(d)] += 1

    top5 = sorted(vote.keys(), key=lambda x: -vote[x])[:5]
    if len(top5) < 5:
        vote_all = defaultdict(int)
        for dims in subject_df["top3_dims_epoch"]:
            for d in dims:
                vote_all[int(d)] += 1
        for d in sorted(vote_all.keys(), key=lambda x: -vote_all[x]):
            if d not in top5:
                top5.append(d)
            if len(top5) >= 5:
                break

    evi = []
    for d in top5:
        info = mapping_dict[d]
        evi.append(f"{info['col']}(dim{d}, r={info['r']:.3f}, p={info['p']:.3f})")

    rule = "全部 epoch 的 TOP3（标签侧 |SHAP|）频数 TOP5"
    final = (
        f"[被试最终报告] 平均MDD概率={mean_prob:.4f}（{rule}）\n"
        f"TOP5特征：{' | '.join(evi)}"
    )
    return final, top5

# ==========================================
# 主流程：5折 + 最优折保存 + 两层报告
# ==========================================
if __name__ == "__main__":
    set_global_seed(RANDOM_SEED)
    mne_df = pd.read_csv(MNE_CSV_PATH)
    mne_df["real_subject_id"] = mne_df["subject"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    X_list, y_list, rows = [], [], []
    for _, row in mne_df.iterrows():
        fp = os.path.join(FEATURES_DIR, f"{row['subject']}.npy")
        if os.path.exists(fp):
            f = np.load(fp).reshape(-1,512).mean(0)
            X_list.append(f)
            y_list.append(row['label'])
            rows.append(row)
    X_np = np.array(X_list, dtype=np.float32)
    X = torch.tensor(X_np).float()
    y = torch.tensor(y_list).long()
    df = pd.DataFrame(rows)
    df['real_subject_id'] = df['subject'].apply(lambda x: '_'.join(x.split('_')[:2]))

    # ========== 5折交叉验证 ==========
    best_val_subject_acc_for_selection = -1
    best_model = None
    best_fold = 0
    best_train_subjects = None
    best_test_subjects = None
    best_threshold = 0.5
    best_train_mean_prob = 0.0
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_val_subject_acc = 0.0
    best_val_mean_prob = 0.0
    df['eegpt_mdd_prob'] = 0.0
    df['eegpt_mdd_pred'] = 0
    df['fold_train_threshold'] = np.nan
    groups = df['real_subject_id'].values
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_train_subs_list = []
    fold_test_subs_list = []

    for fold, (t_idx, v_idx) in enumerate(sgkf.split(X,y,groups=groups)):
        print(f"\n========== Fold {fold+1}/{N_FOLDS} ==========")
        train_subs = np.unique(groups[t_idx])
        test_subs = np.unique(groups[v_idx])
        fold_train_subs_list.append(train_subs)
        fold_test_subs_list.append(test_subs)

        fold_mean, fold_std = fit_standardizer(X_np[t_idx])
        X_train_std = torch.tensor(apply_standardizer(X_np[t_idx], fold_mean, fold_std)).float()
        X_val_std = torch.tensor(apply_standardizer(X_np[v_idx], fold_mean, fold_std)).float()
        X_all_std = torch.tensor(apply_standardizer(X_np, fold_mean, fold_std)).float()

        train_loader = DataLoader(TensorDataset(X_train_std, y[t_idx]), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(X_val_std, y[v_idx]), batch_size=BATCH_SIZE)
        model = MDDClassificationHead().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        y_train_np = y[t_idx].cpu().numpy().astype(int)
        class_counts = np.bincount(y_train_np, minlength=2).astype(float)
        class_weights = np.where(class_counts > 0, len(y_train_np) / (2.0 * class_counts), 1.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE),
            label_smoothing=LABEL_SMOOTHING,
        )
        es = EarlyStopping(PATIENCE)

        for _ in range(EPOCHS):
            model.train()
            for bx,by in train_loader:
                # BatchNorm1d 在训练态要求 batch size > 1；单样本 batch 跳过
                if bx.size(0) < 2:
                    continue
                optimizer.zero_grad()
                criterion(model(bx.to(DEVICE)), by.to(DEVICE)).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                v_loss = np.mean([criterion(model(bx.to(DEVICE)), by.to(DEVICE)).item() for bx,by in val_loader])
            es(v_loss, model)
            if es.early_stop: break

        model.load_state_dict(es.best_state)
        with torch.no_grad():
            prob = torch.softmax(model(X_all_std.to(DEVICE)), dim=1)[:,1].cpu().numpy()
        y_val_np = y[v_idx].cpu().numpy()
        train_prob = prob[t_idx]
        val_prob = prob[v_idx]
        train_subject_ids = groups[t_idx]
        val_subject_ids = groups[v_idx]
        fold_train_mean_prob = float(np.mean(train_prob))
        fold_threshold = find_threshold_with_inner_cv(
            y_train_np,
            train_prob,
            train_subject_ids,
            random_seed=RANDOM_SEED + fold,
        )
        train_pred = (train_prob >= fold_threshold).astype(int)
        fold_train_acc = float((train_pred == y_train_np).mean())
        val_pred = (prob[v_idx] >= fold_threshold).astype(int)
        df.loc[v_idx, 'eegpt_mdd_prob'] = prob[v_idx]
        df.loc[v_idx, 'eegpt_mdd_pred'] = val_pred
        df.loc[v_idx, 'fold_train_threshold'] = fold_threshold
        fold_val_acc = float((val_pred == y_val_np).mean())
        fold_val_subject_acc = compute_subject_level_acc(val_subject_ids, y_val_np, val_pred)
        auc = roc_auc_score(y_val_np, val_prob)
        print(
            f"Fold {fold+1} Train innerCV-threshold = {fold_threshold:.4f}, "
            f"Train mean prob = {fold_train_mean_prob:.4f}, "
            f"Train Acc = {fold_train_acc:.4f}, Val Acc = {fold_val_acc:.4f}, "
            f"Val Subject Acc = {fold_val_subject_acc:.4f}, Val AUC = {auc:.4f}"
        )

        if fold_val_subject_acc > best_val_subject_acc_for_selection:
            best_val_subject_acc_for_selection = fold_val_subject_acc
            best_model = model
            best_fold = fold
            best_train_subjects = train_subs
            best_test_subjects = test_subs
            best_threshold = fold_threshold
            best_train_mean_prob = fold_train_mean_prob
            best_train_acc = fold_train_acc
            best_val_acc = fold_val_acc
            best_val_subject_acc = fold_val_subject_acc
            best_val_mean_prob = float(np.mean(val_prob))

    print(f"\n✅ Best Fold = {best_fold+1}, Best Val Subject Acc = {best_val_subject_acc_for_selection:.4f}")
    print(f"✅ Best Fold Train InnerCV Threshold = {best_threshold:.4f}")
    print(f"✅ Best Fold Train Mean Prob = {best_train_mean_prob:.4f}")
    print(f"✅ Best Fold Train Acc = {best_train_acc:.4f}, Val Acc = {best_val_acc:.4f}, Val Subject Acc = {best_val_subject_acc:.4f}")
    print(f"✅ Best Fold Val Mean Prob = {best_val_mean_prob:.4f}")

    # 使用全量 OOF 概率进行更稳健的阈值估计，减少单折阈值方差
    global_threshold = find_threshold_with_inner_cv(
        y.cpu().numpy().astype(int),
        df["eegpt_mdd_prob"].values.astype(float),
        groups,
        random_seed=RANDOM_SEED + 999,
    )
    print(f"✅ Global OOF InnerCV Threshold = {global_threshold:.4f}")

    # 在全量数据上训练最终导出模型（用于外部测试），并使用被试级分组划分做早停
    full_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 2026)
    full_train_idx, full_val_idx = next(full_cv.split(X_np, y.cpu().numpy(), groups=groups))
    full_mean, full_std = fit_standardizer(X_np[full_train_idx])
    X_full_std = torch.tensor(apply_standardizer(X_np, full_mean, full_std)).float()
    X_full_train_std = X_full_std[full_train_idx]
    X_full_val_std = X_full_std[full_val_idx]
    y_np = y.cpu().numpy().astype(int)

    final_model = MDDClassificationHead().to(DEVICE)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    full_class_counts = np.bincount(y_np[full_train_idx], minlength=2).astype(float)
    full_class_weights = np.where(
        full_class_counts > 0, len(full_train_idx) / (2.0 * full_class_counts), 1.0
    )
    full_class_weights[1] = full_class_weights[1] * 1.8
    final_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(full_class_weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=LABEL_SMOOTHING,
    )
    final_es = EarlyStopping(PATIENCE)
    final_train_loader = DataLoader(
        TensorDataset(X_full_train_std, y[full_train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    final_val_loader = DataLoader(TensorDataset(X_full_val_std, y[full_val_idx]), batch_size=BATCH_SIZE)

    for _ in range(EPOCHS):
        final_model.train()
        for bx, by in final_train_loader:
            if bx.size(0) < 2:
                continue
            final_optimizer.zero_grad()
            final_criterion(final_model(bx.to(DEVICE)), by.to(DEVICE)).backward()
            final_optimizer.step()
        final_model.eval()
        with torch.no_grad():
            v_loss = np.mean(
                [final_criterion(final_model(bx.to(DEVICE)), by.to(DEVICE)).item() for bx, by in final_val_loader]
            )
        final_es(v_loss, final_model)
        if final_es.early_stop:
            break
    final_model.load_state_dict(final_es.best_state)

    # ========== SHAP ==========
    best_model.eval()
    bg_indices = []
    for s in best_train_subjects:
        idxs = df[df['real_subject_id']==s].index.tolist()
        bg_indices.extend(np.random.choice(idxs, size=min(4,len(idxs)), replace=False))
    X_bg = X[bg_indices].to(DEVICE)
    explainer = shap.DeepExplainer(best_model, X_bg)
    shap_values = explainer.shap_values(X.to(DEVICE))
    mdd_shaps = shap_values[1] if isinstance(shap_values,list) else shap_values[...,1]

    # 单个 epoch：512 维有符号 SHAP（不取绝对值）
    viz_i = int(np.clip(SHAP_VIZ_EPOCH_IDX, 0, len(df) - 1))
    _vr = df.iloc[viz_i]
    _title = (
        f"epoch row={viz_i}, subject={_vr.get('subject', '')}, "
        f"label={int(_vr['label'])}, P(MDD)={float(_vr['eegpt_mdd_prob']):.4f}"
    )
    save_shap_signed_full_dim_plot(
        mdd_shaps[viz_i],
        OUTPUT_DIR,
        "shap_single_epoch_512dim.png",
        _title,
    )

    # SHAP 之后再算：各 dim 与 PSD（CLINICAL_COLS）的 r、p；仅用最优折 train 被试，避免泄漏
    train_mask = df["real_subject_id"].isin(best_train_subjects).values
    dim_psd_df, mapping_dict = build_dim_psd_correlation_table(
        X_np, df, train_mask, CLINICAL_COLS
    )

    # ========== 两层报告 ==========
    epoch_reports, top3_list = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        rep, dims = generate_epoch_report(row, mdd_shaps[i], mapping_dict, global_threshold)
        epoch_reports.append(rep)
        top3_list.append(dims)
    df['epoch_report'] = epoch_reports
    df['top3_dims_epoch'] = top3_list

    # ====================== 生成全量被试级表格 ======================
    sub_rows = []
    for sub in df['real_subject_id'].unique():
        sub_df = df[df['real_subject_id'] == sub]
        label = sub_df['label'].iloc[0]
        sub_report, top5_dims = generate_subject_report(sub_df, mapping_dict)

        row = {
            "real_subject_id": sub,
            "label": label,
            "mean_mdd_prob": sub_df['eegpt_mdd_prob'].mean(),
            "best_fold_train_threshold": global_threshold,
            "top5_dims_subject": top5_dims,
            "subject_final_report": sub_report,
            "delta": sub_df['delta'].mean(),
            "theta": sub_df['theta'].mean(),
            "alpha": sub_df['alpha'].mean(),
            "beta": sub_df['beta'].mean(),
            "gamma": sub_df['gamma'].mean(),
            "delta_asym": sub_df['delta_asym'].mean(),
            "theta_asym": sub_df['theta_asym'].mean(),
            "alpha_asym": sub_df['alpha_asym'].mean(),
        }
        sub_rows.append(row)
    subject_df = pd.DataFrame(sub_rows)

    # ====================== 【关键】最优折：拆分 epoch & subject 表格 ======================
    best_train_epoch = df[df['real_subject_id'].isin(best_train_subjects)].copy()
    best_test_epoch = df[df['real_subject_id'].isin(best_test_subjects)].copy()

    best_train_subject = subject_df[subject_df['real_subject_id'].isin(best_train_subjects)].copy()
    best_test_subject = subject_df[subject_df['real_subject_id'].isin(best_test_subjects)].copy()

    # 最优折 train/test：将该被试在 CSV 中所有 epoch 对应的 features/*.npy 复制到 features/fold_train / features/fold_val
    for d in (DIR_STEP3_TRAIN, DIR_STEP3_TEST):
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                os.remove(os.path.join(d, fn))

    def _copy_epoch_npy_for_subjects(subject_ids, dest_dir):
        sid_set = set(np.atleast_1d(subject_ids).tolist())
        for _, row in mne_df[mne_df["real_subject_id"].isin(sid_set)].iterrows():
            name = f"{row['subject']}.npy"
            src = os.path.join(FEATURES_DIR, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, name))

    _copy_epoch_npy_for_subjects(best_train_subjects, DIR_STEP3_TRAIN)
    _copy_epoch_npy_for_subjects(best_test_subjects, DIR_STEP3_TEST)

    # ====================== 保存所有输出 ======================
    dim_psd_df.to_csv(
        os.path.join(OUTPUT_DIR, "dim_psd_correlation.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 全量
    df.to_csv(os.path.join(OUTPUT_DIR, "epoch_report.csv"), index=False, encoding='utf-8-sig')
    subject_df.to_csv(os.path.join(OUTPUT_DIR, "subject_report.csv"), index=False, encoding='utf-8-sig')

    # 最优折 - epoch
    best_train_epoch.to_csv(os.path.join(OUTPUT_DIR, "best_fold_train_epochs.csv"), index=False, encoding='utf-8-sig')
    best_test_epoch.to_csv(os.path.join(OUTPUT_DIR, "best_fold_test_epochs.csv"), index=False, encoding='utf-8-sig')

    # 最优折 - subject
    best_train_subject.to_csv(os.path.join(OUTPUT_DIR, "best_fold_train_subjects.csv"), index=False, encoding='utf-8-sig')
    best_test_subject.to_csv(os.path.join(OUTPUT_DIR, "best_fold_test_subjects.csv"), index=False, encoding='utf-8-sig')

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "feature_mean": full_mean,
            "feature_std": full_std,
            "decision_threshold": float(global_threshold),
        },
        os.path.join(OUTPUT_DIR, "best_classifier.pth"),
    )

    print("\n✅ 全部完成！最终输出：")
    print(
        f"   shap_single_epoch_512dim.png   单个epoch 512维有符号SHAP（行索引={viz_i}）"
    )
    print("0. dim_psd_correlation.csv    512维×PSD 相关与 p 值（最优折 train 被试估计）")
    print("1. epoch_report.csv          全量epoch报告")
    print("2. subject_report.csv       全量被试报告")
    print("3. best_fold_train_epochs.csv  最优折训练集epoch报告")
    print("4. best_fold_test_epochs.csv   最优折测试集epoch报告")
    print("5. best_fold_train_subjects.csv 最优折训练集被试报告")
    print("6. best_fold_test_subjects.csv  最优折测试集被试报告")
    print(f"7. {DIR_STEP3_TRAIN}/                     最优折训练被试的全部 epoch 特征 .npy")
    print(f"8. {DIR_STEP3_TEST}/                      最优折测试被试的全部 epoch 特征 .npy")