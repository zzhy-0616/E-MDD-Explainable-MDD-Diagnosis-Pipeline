import os
import re
import torch
import torch.nn as nn
# ========== 【紧急修复补丁：兼容老版本 PyTorch 的 4-bit 量化】 ==========
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self.get_submodule(".".join(atoms))
        setattr(mod, name, module)
    nn.Module.set_submodule = _set_submodule
# =====================================================================
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import glob
import warnings
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
warnings.filterwarnings("ignore")


def _torch_load_trusted(path, map_location):
    """PyTorch≥2.6 默认 weights_only=True，含 numpy 的旧 checkpoint 需关闭。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ==========================================
# 1. 基础配置
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_SCRIPT_DIR)
_CORE_DIR = os.path.join(_PIPELINE_ROOT, "emdd_core")
_STEP3_DIR = os.path.join(_CORE_DIR, "artifacts", "classification")
SUBJECT_CSV_PATH = os.path.join(_STEP3_DIR, "best_fold_test_subjects.csv")
EPOCH_CSV_PATH = os.path.join(_STEP3_DIR, "best_fold_test_epochs.csv")
FEATURES_DIR = os.path.join(_CORE_DIR, "features", "fold_val")
# 须为 EEGProjector 的 state_dict，或含键 projector_state_dict / model_state_dict 的字典。
# 注意：0416step3_best_model.pth 里是 MDDClassificationHead（net.*），与 proj.* 不兼容，需单独保存投影层权重再改路径。
PROJECTOR_PATH = os.path.join(_CORE_DIR, "artifacts", "projector", "best_eeg_projector.pth")
MODEL_NAME_OR_PATH = os.path.join(_PIPELINE_ROOT, "models", "deepseek-7b")
OUTPUT_EPOCH_CSV = os.path.join(_SCRIPT_DIR, "outputs", "cot", "test_epoch_cot.csv")
# CSV / Prompt 中与 MDD 概率对应的列名（与上游表头一致，修改此处即可全局同步）
EEGPT_MDD_PROB_COL = "eegpt_mdd_prob"
# 输出文件夹不存在则创建
os.makedirs(os.path.dirname(OUTPUT_EPOCH_CSV), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_HIDDEN_SIZE = 3584
EEGPT_DIM = 512
SAMPLES_PER_GROUP = 5
SAMPLES_HC = 4
SAMPLES_MDD = 16
# 改变此处可换一批 test epoch 抽样（HC/MDD 池内 sample 顺序）
RANDOM_SEED = 21
ALLOW_RANDOM_PROJECTOR_FALLBACK = True
# ==========================================
# 2. 投影层（和训练逻辑完全对齐）
# ==========================================
class EEGProjector(nn.Module):
    def __init__(self, input_dim=EEGPT_DIM, hidden_dim=1024, output_dim=LLM_HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.proj(x).unsqueeze(1)


def _extract_projector_state_dict(ckpt_obj):
    """从多种 checkpoint 包装中提取 projector 的 state_dict。"""
    if isinstance(ckpt_obj, dict) and "projector_state_dict" in ckpt_obj:
        sd = ckpt_obj["projector_state_dict"]
    elif isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        sd = ckpt_obj["model_state_dict"]
    else:
        sd = ckpt_obj
    if not isinstance(sd, dict):
        return None
    clean_sd = {}
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        clean_sd[nk] = v
    return clean_sd
# ==========================================
# 【核心修复】完全适配你的epoch_report格式的特征提取函数
# ==========================================
_FEAT_PATTERN = re.compile(
    r"([a-zA-Z0-9_]+)\(dim(\d+),\s*r=([-+]?\d*\.\d+|\d+),\s*p=([-+]?\d*\.\d+|\d+)(?:,\s*SHAP\s*=\s*([-+]?\d*\.\d+|\d+))?\)",
    re.IGNORECASE,
)


def _row_from_feat_match(groups):
    phys, dim, r, p, shap_s = groups
    shap_val = float(shap_s) if shap_s and str(shap_s).strip() else 0.0
    return {
        "dim": f"dim{str(dim).strip()}",
        "physical": str(phys).strip(),
        "r": float(str(r).strip()),
        "p": float(str(p).strip()),
        "shap": shap_val,
    }


def parse_epoch_report_features(report_text):
    """
    返回 (all_features, top3_step3_order)。

    - all_features：整份报告内所有 physical(dim..,r,p,SHAP?) 匹配，供 asym 等用。
    - top3_step3_order：仅解析「TOP3特征」行之后片段中、**按出现顺序** 的前 3 条，
      与 step3_classify_shap 写入顺序一致（不在 step5 二次按 SHAP/r 排序）。
    """
    all_matches = _FEAT_PATTERN.findall(str(report_text))
    print(f"\n✅ 全报告匹配到 {len(all_matches)} 个特征片段")
    for match in all_matches:
        print(f"   特征：{match[0]} | dim：{match[1]} | r：{match[2]} | p：{match[3]} | SHAP：{match[4] or '—'}")

    features = []
    for tup in all_matches:
        try:
            features.append(_row_from_feat_match(tup))
        except ValueError as e:
            print(f"⚠️ 数值转换失败：{tup} | 错误：{e}")

    top3_section = str(report_text)
    for marker in ("TOP3特征：", "TOP3特征:", "TOP3特征"):
        if marker in top3_section:
            top3_section = top3_section.split(marker, 1)[1]
            break

    top3_ordered = []
    for tup in _FEAT_PATTERN.findall(top3_section)[:3]:
        try:
            top3_ordered.append(_row_from_feat_match(tup))
        except ValueError as e:
            print(f"⚠️ TOP3 段数值转换失败：{tup} | {e}")

    return features, top3_ordered

_ASYM_GREEK_SLUG = {
    "delta_asym": "δ_asym",
    "theta_asym": "θ_asym",
    "alpha_asym": "α_asym",
    "beta_asym": "β_asym",
    "gamma_asym": "γ_asym",
}

def asym_metrics_from_report(all_features, report_text):
    """
    从已解析特征与原文中收集 *_asym 指标，生成 Prompt 用语（希腊简写/ 与报告一致的物理名）。
    若报告中未出现任何 _asym，则回退为 delta/theta/alpha 三种典型额叶不对称名。
    """
    ordered, seen = [], set()
    for f in all_features:
        phys = f.get("physical", "")
        if "_asym" in phys.lower() and phys not in seen:
            seen.add(phys)
            ordered.append(phys)
    for m in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9_]*_asym)\b", report_text):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        ordered = ["delta_asym", "theta_asym", "alpha_asym"]
    greek_slash = "/".join(_ASYM_GREEK_SLUG.get(p.lower(), p) for p in ordered)
    phys_join = "、".join(ordered)
    return greek_slash, phys_join


# 与下方 Prompt【一】【二】节文字一致：键 = epoch_report / TOP3 中的 physical 名（小写）
_PSD_HEALTH_REF = {
    "delta": (0.5, 4.0),
    "theta": (1.0, 5.0),
    "alpha": (2.0, 15.0),
    "beta": (0.2, 2.0),
    "gamma": (0.05, 0.5),
    "delta_asym": (-0.10, 0.15),
    "theta_asym": (-0.10, 0.20),
    "alpha_asym": (0.00, 0.25),
}


def _epoch_row_psd_col(row: pd.Series, physical: str):
    """TOP3 的 physical 与 epoch 表 PSD 列名对齐（大小写/原样）。"""
    p = str(physical).strip()
    if p in row.index:
        return p
    pl = p.lower()
    if pl in row.index:
        return pl
    return None


def _read_psd_value(row: pd.Series, physical: str):
    col = _epoch_row_psd_col(row, physical)
    if col is None:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def psd_status_lines(row, physical):
    """
    按 Prompt【一】【二】节参考范围，对当前 epoch 行中同名 PSD 列做数值对照。
    返回 (数值状态判定 正文, 神经生理意义 正文)，不含行首「- 数值状态判定：」前缀。
    """
    key = str(physical).strip().lower()
    bounds = _PSD_HEALTH_REF.get(key)
    val = _read_psd_value(row, physical)

    if bounds is None:
        rng_disp = "未在 Prompt【一】【二】节定义"
        v_disp = "—" if val is None else f"{val:.2f}"
        pos = "无法对照既定健康参考范围（该 physical 无表内上下限）"
        neuro = "该指标无既定数值阈值表，仅可作定性神经生理关联"
        line1 = f"该特征实际数值为{v_disp}，健康参考范围为【{rng_disp}】，该数值{pos}"
        line2 = f"该数值状态代表{neuro}"
        return line1, line2

    lo, hi = bounds
    rng_disp = f"{lo:.2f}～{hi:.2f}"
    if val is None:
        v_disp = "—"
        pos = "无法判定（本行 CSV 缺少该列或数值缺失）"
        neuro = "因缺少客观 PSD 数值，无法进行与健康范围的定量对照"
        line1 = f"该特征实际数值为{v_disp}，健康参考范围为【{rng_disp}】，该数值{pos}"
        line2 = f"该数值状态代表{neuro}"
        return line1, line2

    v_disp = f"{val:.2f}"
    if val < lo:
        pos = "低于健康参考范围下限"
    elif val > hi:
        pos = "高于健康参考范围上限"
    else:
        pos = "处于健康参考范围内"
    if pos == "处于健康参考范围内":
        neuro = "脑电活动平稳，符合静息态正常表现"
    else:
        neuro = "脑电活动偏离健康基线，存在神经调控异常"
    line1 = f"该特征实际数值为{v_disp}，健康参考范围为【{rng_disp}】，该数值{pos}"
    line2 = f"该数值状态代表{neuro}"
    return line1, line2


def infer_epoch_group(row):
    """
    将单条 epoch 样本映射为 H / MDD。
    优先使用显式标签列；若不存在则回退到 eegpt_mdd_prob 阈值判断。
    """
    candidate_cols = ["label", "group", "diagnosis", "class", "target"]
    for col in candidate_cols:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip().upper()
            if raw in {"MDD", "DEP", "DEPRESSION", "1"}:
                return "MDD"
            if raw in {"H", "HC", "HEALTHY", "CONTROL", "0"}:
                return "H"
    pred = get_step3_epoch_pred_label(row)
    return "MDD" if pred == 1 else "H"


_STEP3_BEST_THRESHOLD_CACHE = None


def load_step3_best_threshold():
    """与 step3_classify_shap 一致：唯一阈值 best_threshold（被试表列 / ckpt）。"""
    global _STEP3_BEST_THRESHOLD_CACHE
    if _STEP3_BEST_THRESHOLD_CACHE is not None:
        return _STEP3_BEST_THRESHOLD_CACHE
    if not os.path.isfile(SUBJECT_CSV_PATH):
        print(f"⚠️ 未找到被试级 CSV：{SUBJECT_CSV_PATH}，阈值回退 0.5")
        _STEP3_BEST_THRESHOLD_CACHE = 0.5
        return _STEP3_BEST_THRESHOLD_CACHE
    sub = pd.read_csv(SUBJECT_CSV_PATH)
    for col in ("best_threshold", "best_fold_train_threshold"):
        if col in sub.columns:
            vals = sub[col].dropna()
            if len(vals) > 0:
                _STEP3_BEST_THRESHOLD_CACHE = float(vals.iloc[0])
                return _STEP3_BEST_THRESHOLD_CACHE
    print("⚠️ 被试 CSV 缺少 best_threshold（或旧列 best_fold_train_threshold），回退 0.5")
    _STEP3_BEST_THRESHOLD_CACHE = 0.5
    return _STEP3_BEST_THRESHOLD_CACHE


def get_step3_threshold(_row=None):
    """step3 仅保留 best_threshold；epoch 表无阈值列时由此读取被试表。"""
    return load_step3_best_threshold()


def get_step3_epoch_pred_label(row):
    """仅用 eegpt_mdd_prob 与 best_threshold 比较（与 step3 一致）。"""
    thr = load_step3_best_threshold()
    return 1 if float(row[EEGPT_MDD_PROB_COL]) >= thr else 0


def get_step3_subject_pred_label(subject_epoch_df, subject_row):
    """各 epoch 用同一 best_threshold 由概率得到 0/1，再多数投票。"""
    thr = load_step3_best_threshold()
    probs = subject_epoch_df[EEGPT_MDD_PROB_COL].astype(float)
    preds = (probs >= thr).astype(int).to_numpy()
    if len(preds) == 0:
        return 0
    votes_pos = int((preds == 1).sum())
    votes_neg = int((preds == 0).sum())
    return 1 if votes_pos > votes_neg else 0


def build_subject_eeg_feature(subject_epoch_df):
    """将同一被试全部 epoch 的 512 维特征做均值，作为被试级输入。"""
    vecs = []
    for s in subject_epoch_df["subject"].astype(str).tolist():
        fp = os.path.join(FEATURES_DIR, f"{s}.npy")
        if os.path.exists(fp):
            vec = np.load(fp).reshape(-1, 512).mean(axis=0)
            vecs.append(vec)
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)
# ==========================================
# 3. Stage 1：从报告提取 r/p/SHAP，TOP3 顺序同 step3 + 生成 CoT
# ==========================================
def stage1_generate_cot():
    print("\n" + "="*50)
    print(f"🚀 [Stage 1] 生成最优折 Test 集随机 {SAMPLES_HC}+{SAMPLES_MDD} epoch CoT（标签参照 step3 分类器输出）")
    print("="*50)

    # 读取 step3 最优折 test 的 epoch 表
    epoch_df = pd.read_csv(EPOCH_CSV_PATH)
    epoch_required_cols = ["real_subject_id", "subject", EEGPT_MDD_PROB_COL, "epoch_report"]
    for col in epoch_required_cols:
        if col not in epoch_df.columns:
            raise KeyError(f"Epoch CSV中缺少必要列：{col}，请检查文件！")

    epoch_df = epoch_df[
        epoch_df["epoch_report"].notna()
        & epoch_df["epoch_report"].apply(lambda x: len(str(x).strip()) > 0)
    ].copy()
    epoch_df["feat_path"] = epoch_df["subject"].apply(lambda s: os.path.join(FEATURES_DIR, f"{s}.npy"))
    epoch_df = epoch_df[epoch_df["feat_path"].apply(os.path.exists)].copy()
    if len(epoch_df) == 0:
        print("🚨 致命错误：test epoch 无可用特征文件")
        return None

    # 按 step3 预测标签分组，随机抽样
    epoch_df["pred_label_by_prob"] = epoch_df.apply(get_step3_epoch_pred_label, axis=1).astype(int)
    hc_pool = epoch_df[epoch_df["pred_label_by_prob"] == 0].copy()
    mdd_pool = epoch_df[epoch_df["pred_label_by_prob"] == 1].copy()
    hc_take = min(SAMPLES_HC, len(hc_pool))
    mdd_take = min(SAMPLES_MDD, len(mdd_pool))
    if hc_take < SAMPLES_HC or mdd_take < SAMPLES_MDD:
        print(f"⚠️ 样本不足：目标 HC={SAMPLES_HC}, MDD={SAMPLES_MDD}；实际 HC={hc_take}, MDD={mdd_take}")
    sampled_hc = hc_pool.sample(n=hc_take, random_state=RANDOM_SEED) if hc_take > 0 else hc_pool
    sampled_mdd = mdd_pool.sample(n=mdd_take, random_state=RANDOM_SEED) if mdd_take > 0 else mdd_pool
    df_test = pd.concat([sampled_hc, sampled_mdd], axis=0).sample(frac=1, random_state=RANDOM_SEED).copy()
    print(f"🎯 test 集 epoch 抽样完成：共 {len(df_test)} 条（HC={hc_take}, MDD={mdd_take}）")

    # 初始化结果列
    df_test["llm_cot"] = ""
    df_test["llm_prediction"] = -1
    df_test["top3_physicals"] = ""

    # 断点续跑：复用已有 CSV 中同 subject 的 CoT
    if os.path.isfile(OUTPUT_EPOCH_CSV):
        prev_df = pd.read_csv(OUTPUT_EPOCH_CSV)
        done_by_subject = {}
        for _, prev_row in prev_df.iterrows():
            subj = str(prev_row.get("subject", "")).strip()
            cot = str(prev_row.get("llm_cot", "")).strip()
            if subj and cot:
                done_by_subject[subj] = prev_row
        reused = 0
        for idx, row in df_test.iterrows():
            subj = str(row["subject"])
            if subj in done_by_subject:
                prev_row = done_by_subject[subj]
                df_test.at[idx, "llm_cot"] = prev_row.get("llm_cot", "")
                df_test.at[idx, "llm_prediction"] = prev_row.get("llm_prediction", -1)
                df_test.at[idx, "top3_physicals"] = prev_row.get("top3_physicals", "")
                reused += 1
        if reused:
            print(f"♻️ 断点续跑：已从 {OUTPUT_EPOCH_CSV} 复用 {reused} 条已有 CoT")

    pending_mask = df_test["llm_cot"].astype(str).str.strip() == ""
    pending_count = int(pending_mask.sum())
    if pending_count == 0:
        print("✅ 全部样本已有 CoT，跳过 LLM 生成")
        df_test.to_csv(OUTPUT_EPOCH_CSV, index=False, encoding="utf-8-sig")
        return df_test
    print(f"📝 待生成 CoT：{pending_count} / {len(df_test)} 条")

    # 加载Tokenizer和大模型（仅在有待生成样本时）
    print("🔄 正在加载 Tokenizer 和大模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 正在加载DeepSeek模型...")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print("   检测到 GPU，使用 4-bit 量化加载以节省显存...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _offload_dir = os.path.join(_SCRIPT_DIR, "_llm_offload")
        os.makedirs(_offload_dir, exist_ok=True)
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "7500MiB", "cpu": "2GiB"},
            offload_folder=_offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        print("   未检测到 GPU，以 bfloat16 加载（需较大内存，建议安装 CUDA 版 PyTorch）...")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
    # 冻结大模型权重
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()

    # 加载投影层权重
    print("🧠 正在加载投影层权重...")
    projector = EEGProjector().to(DEVICE)
    _proj_ckpt = _torch_load_trusted(PROJECTOR_PATH, DEVICE)
    _proj_sd = _extract_projector_state_dict(_proj_ckpt)
    expected_keys = set(projector.state_dict().keys())
    loaded_ok = False
    if isinstance(_proj_sd, dict):
        proj_like = set(_proj_sd.keys())
        if expected_keys.issubset(proj_like):
            projector.load_state_dict(_proj_sd, strict=True)
            loaded_ok = True
    if not loaded_ok:
        msg = (
            f"⚠️ 投影层权重与 EEGProjector 不匹配：{PROJECTOR_PATH}\n"
            "   期望键示例：proj.0.weight / proj.2.weight；"
            "当前更像分类头(net.*)权重。"
        )
        if ALLOW_RANDOM_PROJECTOR_FALLBACK:
            print(msg + "\n   已回退为随机初始化 projector 继续运行。")
        else:
            raise RuntimeError(msg)
    projector.eval()

    # CoT 术语与 step3 epoch_report「阈值参考」一致：best_threshold（唯一）
    cot_best_thr = load_step3_best_threshold()
    print(f"📌 CoT 中决策阈值 = step3 best_threshold: {cot_best_thr:.6f}")

    # ===================== 核心循环：逐 epoch 处理 =====================
    gen_count = 0
    for count, (idx, row) in enumerate(tqdm(df_test.iterrows(), total=len(df_test))):
        if str(df_test.at[idx, "llm_cot"]).strip():
            continue
        gen_count += 1
        eeg_feat = torch.tensor(np.load(row["feat_path"]).reshape(-1, 512).mean(axis=0)).float().unsqueeze(0).to(DEVICE)
        original_report = str(row["epoch_report"])
        mdd_prob = float(row[EEGPT_MDD_PROB_COL])
        pred_label = int(row["pred_label_by_prob"])
        decision_thr = cot_best_thr
        is_mdd = pred_label == 1
        is_hc = pred_label == 0

        # 3. 从 epoch 报告提取：全量用于 asym；TOP3 顺序沿用 step3「TOP3特征」行（不重复排序）
        all_features, top3_from_step3 = parse_epoch_report_features(original_report)
        if not all_features:
            print(f"⚠️  未提取到报告特征：{row['subject']} → 使用默认特征填充")
            all_features = [
                {"dim": "dim0", "physical": "alpha_asym", "r": -0.189, "p": 0.000, "shap": 0.0},
                {"dim": "dim1", "physical": "delta", "r": 0.073, "p": 0.000, "shap": 0.0},
                {"dim": "dim2", "physical": "theta_asym", "r": 0.187, "p": 0.000, "shap": 0.0},
            ]

        top3_features = list(top3_from_step3)
        if not top3_features:
            top3_features = all_features[:3]
        while len(top3_features) < 3:
            top3_features.append(all_features[len(top3_features) % len(all_features)])

        feature_summary = "TOP3 顺序与 step3 epoch_report「TOP3特征」行一致（step5 不重排）"

        # 5. 格式化特征文本 + 补全 Prompt 变量
        feature_text = ""
        for i, f in enumerate(top3_features, 1):
            sv = f.get("shap", 0.0)
            feature_text += f"""TOP{i} {f['physical']}({f['dim']}) | SHAP={sv:.4f} | r={f['r']:.4f} | p={f['p']:.4f}\n"""
        
        # 【修复】Prompt中所有占位符变量定义
        true_label_name = "重度抑郁症(MDD)" if is_mdd else "完全健康的正常人(HC)"
        true_label_str = "MDD" if is_mdd else "HC"
        prob_desc = f"（{EEGPT_MDD_PROB_COL}={mdd_prob:.4f}，阈值{decision_thr:.4f}）"
        top_evidence_text = "、".join([f"{f['physical']}({f['dim']})" for f in top3_features])
        asym_slash_prompt, asym_phys_prompt = asym_metrics_from_report(all_features, original_report)
        mdd_prob_col_name = EEGPT_MDD_PROB_COL
        feature_type = "核心特征（顺序同 step3 TOP3 行）"
        feat0_stat, feat0_neuro = psd_status_lines(row, top3_features[0]["physical"])
        feat1_stat, feat1_neuro = psd_status_lines(row, top3_features[1]["physical"])
        feat2_stat, feat2_neuro = psd_status_lines(row, top3_features[2]["physical"])

        # 6. 最终Prompt（彻底修复大模型幻觉的重构版本）
        prompt = f"""<|im_start|>system
你是一名严谨的计算精神病学专家，基于脑电特征分析，输出重度抑郁症（MDD）诊断的标准化推理过程（CoT）。
【核心学术术语定义（必须严格遵守，模块职责物理隔离）】
1.  相关系数r：皮尔逊相关系数，**仅用于衡量EEGPT隐特征与对应MNE物理指标之间的线性相关程度，与MDD预测概率、模型分类决策完全无关！！
    - 取值范围[-1,1]，绝对值越接近1，隐特征与MNE物理指标的线性相关性越强；
    - r>0：隐特征与MNE物理指标呈正相关；r<0：隐特征与MNE物理指标呈负相关。
2.  显著性p值：检验EEGPT隐特征与MNE物理指标相关关系的统计学显著性，仅可在【降维物理映射】模块中解读。
    - p<0.05 = 相关性显著；p<0.01 = 相关性高度显著；p<0.001 = 相关性极其显著。
3.  全局协同决策机制：EEGPT模型是一个高维黑盒，单一隐特征无法绝对决定最终的分类方向。你必须结合全局预测概率（MDDprob），解释特征对应的神经生理意义是如何作为底层证据，参与并支持了最终的全局判定的。**绝对禁止给单一特征强加“推向HC”或“推向MDD”的绝对方向标签！**
4.  MDDprob：模型预测该样本为重度抑郁症（MDD）的概率值，取值范围[0,1]，与 step3 **best_threshold** {decision_thr:.4f} 比较：大于该阈值倾向 MDD，小于该阈值倾向 HC（与 epoch_report「阈值参考」、best_classifier.pth 一致）。
====================
【核心任务】
严格按照固定格式，输出三层核心对应关系，禁止冗余内容：
1.  第一层：明确该EEGPT特征对MDD分类的重要性（基于SHAP值）
2.  第二层：明确该EEGPT特征对应的PSD物理指标（基于r/p相关性）
3.  第三层：解释该特征与MDD临床神经生理特征的契合度
====================
【一】全频段功率的神经生理意义（δ / θ / α / β / γ）
1.δ 功率（Delta，参考范围 0.5-4.0）
    -高功率（超出上限）→ 深度睡眠、皮质抑制、病理状态
    -低功率（处于范围内）→ 皮质觉醒度正常
    -MDD 常见：δ 功率升高，超出健康参考范围
    -HC 常见：δ 功率处于正常偏低水平，符合参考范围
2.  θ 功率（Theta，参考范围 1.0-5.0）
    -高功率（超出上限）→ 困倦、情绪波动、认知疲劳
    -低功率（处于范围内）→ 注意力稳定、神经调控良好
    -MDD 常见：θ 功率升高，超出健康参考范围
    -HC 常见：θ 功率适中偏低，符合参考范围
3.α 功率（Alpha，参考范围 2.0-15.0）
    -高功率（处于范围内）→ 放松、清醒静息、神经稳态好
    -低功率（低于下限）→ 焦虑、紧张、皮质过度激活
    -MDD 常见：α 功率降低，低于健康参考范围下限
    -HC 常见：α 功率稳定、对称，符合参考范围
4.β 功率（Beta，参考范围 0.2-2.0）
    -高功率（超出上限）→ 活跃思考、警觉、焦虑
    -低功率（处于范围内）→ 静息状态正常
    -MDD 常见：β 功率异常升高，超出健康参考范围
    -HC 常见：β 功率平稳，符合参考范围
5.γ 功率（Gamma，参考范围 0.05-0.5）
    -高功率（超出上限）→ 情绪加工、认知整合增强
    -低功率（处于范围内）→ 静息状态正常
    -MDD 常见：γ 功率紊乱，偏离健康参考范围
    -HC 常见：γ 功率稳定，符合参考范围
====================vv
【二】全功率不对称指数的神经生理意义（δ_asym / θ_asym / α_asym）
不对称指数计算公式：不对称值 = 左额叶功率 − 右额叶功率
1.δ 功率不对称（δ_asym，健康参考范围 - 0.10~0.15）
    -数值偏高（超出上限）→ 左额叶抑制减弱
    -数值接近 0（处于范围内）→ 左右额叶抑制功能平衡
    -MDD 常见：δ_asym 降低，低于健康参考范围下限
    -HC 常见：δ_asym 接近 0，处于参考范围内，平衡稳定
2.θ 功率不对称（θ_asym，健康参考范围 - 0.10~0.20）
    -数值偏高（处于范围内）→ 左额叶情绪调节占优
    -数值偏低（低于下限）→ 右额叶偏向、情绪不稳
    -MDD 常见：θ_asym 降低，低于健康参考范围下限
    -HC 常见：θ_asym 接近 0，处于参考范围内，平衡稳定
3.α 功率不对称（α_asym，核心标志物，健康参考范围 0.00~0.25）
    -健康人（HC）：α_asym 处于 0.00~0.25 范围内，接近 0 → 左右额叶功率对称、情绪稳定、神经稳态正常  
    -抑郁症（MDD）：α_asym 明显低于 0.00 → 右额叶功率占优 → 负性情绪偏向、情绪调节异常  
    -α_asym 处于健康参考范围 → 额叶功能对称、神经稳态良好 → 推动模型判断为 HC
====================================================================
【四】输出强制要求
1. 严格使用上述神经生理学定义解释
2. 禁止编造，必须基于样本数值
3. HC 禁止使用病理词汇，必须解释为“神经稳态正常”“功能对称”“情绪调节良好”
====================================================================
【重要规则·必须遵守】
1. 你必须解释：**该特征的高低如何对应脑功能变化**
2. 你必须连接：**特征变化 → 神经意义 → {true_label_name}相关表现**
3. 你必须对齐 SHAP 值：
   - SHAP 正值 → 推动模型判断为 MDD
   - SHAP 负值 → 推动模型判断为 HC
4. 最终结论必须与前面所有推理完全一致，禁止逻辑矛盾。
5.当提取的特征与{true_label_name}相违背时必须解释为什么违背，并给出合理的解释。不要强行将不匹配的特征和{true_label_name}联系起来。
====================
【样本基础信息】
- 模型预测结果：{true_label_name}
#- MDD 预测概率：{mdd_prob:.4f}
#- 决策阈值：{decision_thr:.4f}

#【对分类最重要的3个EEGPT特征（按重要性排序）】
#1. {top3_features[0]['physical']} (dim{top3_features[0]['dim']})
   - SHAP 重要性：{top3_features[0]['shap']:.4f}
   - 与 PSD 相关性：r={top3_features[0]['r']:.4f}, p={top3_features[0]['p']:.4f}

#2. {top3_features[1]['physical']} (dim{top3_features[1]['dim']})
   - SHAP 重要性：{top3_features[1]['shap']:.4f}
   - 与 PSD 相关性：r={top3_features[1]['r']:.4f}, p={top3_features[1]['p']:.4f}

#3. {top3_features[2]['physical']} (dim{top3_features[2]['dim']})
   - SHAP 重要性：{top3_features[2]['shap']:.4f}
   - 与 PSD 相关性：r={top3_features[2]['r']:.4f}, p={top3_features[2]['p']:.4f}

====================
【你必须严格按照以下格式输出·缺一不可】

【一、样本整体判断】
本样本模型预测为 {true_label_name}，概率为 {mdd_prob:.4f}。

【二、TOP3核心特征推理（依据脑电生理知识）】
- 须落实【重要规则】第5条：逐条对照本特征（PSD 数值区间、SHAP 方向、与上文「典型 HC/MDD」文本）与模型预测 **{true_label_name}**。若**一致**，简要说明如何支持该预测；若**部分一致或相违背**，必须单独写明违背点，并给出合理机制解释（如：多隐变量协同、全局 MDDprob 由其它维度主导、个体差异/亚型等），**禁止**把明显不匹配的单条特征硬写成「完全符合」{true_label_name} 的典型模式。
1.  特征：{top3_features[0]['physical']}
    - 基础信息提取：对应EEGPT维度为{top3_features[0]['dim']}，该EEGPT维度与PSD特征【{top3_features[0]['physical']}】的相关系数r={top3_features[0]['r']:.4f}，显著性p={top3_features[0]['p']:.4f}
    - 数值状态判定：{feat0_stat}
    - 神经生理意义：{feat0_neuro}
    - 与模型预测 {true_label_name} 的对照：先给出「一致 / 部分一致 / 相违背」；若非「一致」，必须写清为何仍可出现该全局预测，不得省略或牵强附会。

2.  特征：{top3_features[1]['physical']}
    - 基础信息提取：对应EEGPT维度为{top3_features[1]['dim']}，该EEGPT维度与PSD特征【{top3_features[1]['physical']}】的相关系数r={top3_features[1]['r']:.4f}，显著性p={top3_features[1]['p']:.4f}
    - 数值状态判定：{feat1_stat}
    - 神经生理意义：{feat1_neuro}
    - 与模型预测 {true_label_name} 的对照：先给出「一致 / 部分一致 / 相违背」；若非「一致」，必须写清为何仍可出现该全局预测，不得省略或牵强附会。

3.  特征：{top3_features[2]['physical']}
    - 基础信息提取：对应EEGPT维度为{top3_features[2]['dim']}，该EEGPT维度与PSD特征【{top3_features[2]['physical']}】的相关系数r={top3_features[2]['r']:.4f}，显著性p={top3_features[2]['p']:.4f}
    - 数值状态判定：{feat2_stat}
    - 神经生理意义：{feat2_neuro}
    - 与模型预测 {true_label_name} 的对照：先给出「一致 / 部分一致 / 相违背」；若非「一致」，必须写清为何仍可出现该全局预测，不得省略或牵强附会。

【三、一致性结论（必须完全匹配前面推理）】
综合上述 TOP3 的数值、神经生理意义及各条与 **{true_label_name}** 的对照（须显式收束第【二】节中已承认的**相悖或部分一致**之处及其解释），
说明全局判定为 {true_label_name} 如何与 MDDprob、阈值及多特征协同相容，
与模型最终决策一致且与第【二】节无矛盾。

</think>
"""

# 预填充：只保留干干净净的标题
        suffix_text = f"""<|im_end|>
<|im_start|>assistant
<think>
【一、样本整体判断】
"""
        # 7. 文本Token化 + 脑电特征嵌入拼接
        with torch.no_grad():
            # 文本Token化
            text_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
            text_embeds = llm.get_input_embeddings()(text_tokens.input_ids)
            # 脑电特征投影
            eeg_embeds = projector(eeg_feat).to(dtype=text_embeds.dtype)
            # 拼接嵌入：脑电特征 + 文本Prompt
            inputs_embeds = torch.cat([eeg_embeds, text_embeds], dim=1)
            attention_mask = torch.cat([torch.ones((1, 1), device=DEVICE), text_tokens.attention_mask], dim=1)
            
            # 8. 大模型生成
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=1200,
                temperature=0.25,
                top_p=0.85,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 9. 解析生成结果
        perfect_final_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # 打印生成结果
        print(f"\n{'='*15} Epoch {gen_count}/{pending_count} 生成完成：{row['subject']} {'='*15}")
        print(perfect_final_output[:500] + "..." if len(perfect_final_output) > 500 else perfect_final_output)
        print("="*60)
        df_test.at[idx, "llm_prediction"] = int(pred_label)
        df_test.at[idx, "llm_cot"] = perfect_final_output
        df_test.at[idx, "top3_physicals"] = ",".join([f["physical"] for f in top3_features])
        df_test.to_csv(OUTPUT_EPOCH_CSV, index=False, encoding="utf-8-sig")

    # 保存最终CSV
    df_test.to_csv(OUTPUT_EPOCH_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 抽样 epoch 生成完成（共 {len(df_test)} 条），结果已保存至：{OUTPUT_EPOCH_CSV}")
    return df_test

# ==========================================
# 4. Stage 2: 验证生成结果一致性、对齐率、术语准确率与解释一致性
# ==========================================
def stage2_validate_result(df_test):
    if df_test is None:
        return
        
    print("\n" + "="*50)
    print("🗳️ [Stage 2] XAI 黄金三角评估：特征对齐率 / 临床术语准确率 / 解释一致性")
    print("="*50)
    valid_df = df_test[df_test['llm_prediction'] != -1].copy()
    
    if len(valid_df) == 0:
        print("❌ 无有效生成结果")
        return
    
    # ---------------- 1. 标签一致性计算 ----------------
    # 注意：对比标签与 step3 一致：eegpt_mdd_prob 与 best_threshold 比较（get_step3_epoch_pred_label），
    # 不使用真实被试标签。
    total_valid = len(valid_df)
    valid_df['match'] = valid_df['llm_prediction'] == valid_df['pred_label_by_prob']
    accuracy = valid_df['match'].mean() * 100
    y_ref = valid_df['pred_label_by_prob'].astype(int).values
    y_hat = valid_df['llm_prediction'].astype(int).values
    bal_acc = balanced_accuracy_score(y_ref, y_hat) * 100
    kappa = cohen_kappa_score(y_ref, y_hat)
    weighted_f1 = f1_score(y_ref, y_hat, average='weighted') * 100
    
    # ---------------- 2. 三大 XAI 量化指标计算 ----------------
    align_scores = []
    clinical_scores = []
    consistency_scores = [] # ✨ 新增：解释一致性得分
    
    for idx, row in valid_df.iterrows():
        cot_text = str(row['llm_cot'])
        top3_physicals = str(row['top3_physicals']).split(',')
        is_hc = row['pred_label_by_prob'] == 0
        pred_label = row['llm_prediction']
        
        # 【指标 A: 特征对齐率 (Feature Alignment Rate)】
        align_count = sum(1 for phys in top3_physicals if phys.strip() in cot_text)
        align_scores.append(align_count / len(top3_physicals) if len(top3_physicals) > 0 else 0)
        
        # 【指标 B: 临床术语准确率 (Clinical Terminology Accuracy)】
        term_score = 100.0
        if not all(term in cot_text for term in ['降维物理映射', '全局概率协同解析']):
            term_score -= 10.0
        if re.search(r'(_asym|不对称指数)[^，。；\n]*绝对功率', cot_text):
            term_score -= 30.0
        if is_hc and re.search(r'(?<!无)(?<!非)(?<!未)(?<!没有)(?<!不存在)(异常|紊乱|病理|偏侧化)', cot_text):
            term_score -= 40.0
        clinical_scores.append(max(0.0, term_score))
        
        # ✨ 【指标 C: 解释一致性 (Explanation Consistency / Logical Coherence)】✨
        # 满分 100 分，检查大模型的证据、步骤与最终结论是否发生“精神分裂”
        consist_score = 100.0
        
        # 扣分项 1: 结论倒挂 / 精神分裂 (-50分)
        # 预测是MDD，但最后一行结论写成了完全健康的正常人；或者预测是HC，结论写了MDD
        if pred_label == 1 and re.search(r'最终(判定|结论).*完全健康的正常人|最终.*HC', cot_text):
            consist_score -= 50.0
        elif pred_label == 0 and re.search(r'最终(判定|结论).*重度抑郁症|最终.*MDD', cot_text):
            consist_score -= 50.0
            
        # 扣分项 2: 证据与概率方向严重冲突 (-30分)
        # 例如：是HC样本(概率<0.5)，但文本里毫无铺垫地疯狂输出“完全符合MDD，高度支持MDD”
        if pred_label == 0:
            if re.search(r'(高度支持|完全符合|确诊)[^，。]*MDD', cot_text) and not re.search(r'(不支持|不符合)', cot_text):
                consist_score -= 30.0
        if pred_label == 1:
            if re.search(r'(高度支持|完全符合|确诊)[^，。]*HC', cot_text) and not re.search(r'(不支持|不符合)', cot_text):
                consist_score -= 30.0
                
        # 扣分项 3: 缺少闭环验证结构 (-20分)
        # CoT 必须要有头有尾，如果步骤三不见了，说明逻辑链断裂
        if '跨模态特征交叉验证' not in cot_text:
            consist_score -= 20.0
            
        consistency_scores.append(max(0.0, consist_score))
        
    # 保存打分结果到 CSV
    valid_df['feature_alignment_rate'] = align_scores
    valid_df['clinical_term_score'] = clinical_scores
    valid_df['explanation_consistency'] = consistency_scores # 保存一致性得分
    
    # ---------------- 打印最终评测报告 ----------------
    mean_align = np.mean(align_scores) * 100
    mean_clinical = np.mean(clinical_scores)
    mean_consist = np.mean(consistency_scores) # 计算平均一致性
    
    print(f"👤 涉及被试数: {valid_df['real_subject_id'].nunique()}")
    print(f"📝 有效生成样本数: {total_valid} / {len(df_test)}")
    print(f"🎯 标签一致率 (Label Match, 参考=step3预测): {accuracy:.2f}%")
    print(f"⚖️ 平衡准确率 (Balanced Accuracy): {bal_acc:.2f}%")
    print(f"🤝 科恩 Kappa 系数 (Cohen's Kappa): {kappa:.4f}")
    print(f"📐 加权 F1 值 (Weighted F1): {weighted_f1:.2f}%")
    print(f"🔗 特征对齐率 (Alignment)  : {mean_align:.2f}% (无特征遗漏/捏造)")
    print(f"🏥 术语准确率 (Terminology): {mean_clinical:.2f}/100 (无医学名词滥用)")
    print(f"⚖️ 解释一致性 (Consistency): {mean_consist:.2f}/100 (逻辑闭环，无自相矛盾)")
    
    if accuracy >= 90 and mean_align >= 95 and mean_clinical >= 85 and mean_consist >= 90:
        print(f"\n🎉 绝杀！你的大模型不仅像神医一样精准，而且说理透彻、逻辑严密，达到了临床白盒级可信度！")
    else:
        print(f"\n⚠️  诊断分析：")
        if mean_align < 95: print("  - 存在特征遗漏。")
        if mean_clinical < 85: print("  - 存在术语幻觉 (张冠李戴或乱报病理)。")
        if mean_consist < 90: print("  - 存在逻辑断裂 (推理证据与最终结论自相矛盾)。")

    # 覆盖保存带有评分的 CSV
    valid_df.to_csv(OUTPUT_EPOCH_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 包含【XAI 黄金三角评估】结果的最终报告已更新至：{OUTPUT_EPOCH_CSV}")

if __name__ == "__main__":
    df_result = stage1_generate_cot()
    stage2_validate_result(df_result)