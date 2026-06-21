import sys
import os
import numpy as np
import torch
import random
import mne
from functools import partial
import glob

# ===================== 默认路径（workflow 会通过 configs/*.yaml 覆盖） =====================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EEGPT_ROOT = os.path.join(_SCRIPT_DIR, "vendor", "EEGPT")
EEGPT_DOWNSTREAM_TUEG = os.path.join(EEGPT_ROOT, "downstream_tueg")

for p in [EEGPT_DOWNSTREAM_TUEG, EEGPT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_CKPT_PATH = os.path.join(EEGPT_ROOT, "checkpoint", "eegpt_mcae_58chs_4s_large4E.ckpt")
DEFAULT_FEATURES_DIR = os.path.join(_SCRIPT_DIR, "emdd_core", "features")

# ===================== 加载模型 =====================
def load_eegpt_encoder(checkpoint_path=None, device=None):
    from Modules.models.EEGPT_mcae_finetune_change import EEGTransformer
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGTransformer(
        img_size=[19, 1024], patch_size=64, embed_dim=512, embed_num=4,
        depth=8, num_heads=8, mlp_ratio=4.0, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )
    model = model.to(device)
    model.eval()

    if os.path.isfile(DEFAULT_CKPT_PATH):
        ckpt_data = torch.load(DEFAULT_CKPT_PATH, map_location=device, weights_only=False)
        state = ckpt_data.get("state_dict", ckpt_data)
        encoder_state = {k[15:]: v for k, v in state.items() if k.startswith("target_encoder.")}
        model.load_state_dict(encoder_state, strict=False)
        print("✅ 模型加载成功\n")
    return model

# ===================== 核心：双分支预处理提取 =====================
def process_one_file(set_path, encoder, device):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    eeg = raw.get_data()  # (19, 1536)

    # 1. 裁剪到 4s
    C, T = eeg.shape
    crop_start = random.randint(0, T - 1024)
    eeg_cropped = eeg[:, crop_start : crop_start + 1024]

    # 2. 预处理分支：基线校正 -> Z-Score (仅给 EEGPT)
    channel_means = np.mean(eeg_cropped, axis=1, keepdims=True)
    eeg_baseline_corrected = eeg_cropped - channel_means # 消除直流漂移
    
    global_std = np.std(eeg_baseline_corrected)
    eeg_zscore = eeg_baseline_corrected / (global_std + 1e-6) # 归一化以稳定模型

    # 转张量
    tensor = torch.from_numpy(eeg_zscore).float().unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        feat = encoder(tensor)

    del tensor
    torch.cuda.empty_cache()

    return feat.cpu().numpy(), crop_start

# ===================== 批量处理文件夹 =====================
def process_folder(folder_path, save_folder=None):
    if save_folder is None:
        save_folder = DEFAULT_FEATURES_DIR
    os.makedirs(save_folder, exist_ok=True)

    set_files = glob.glob(os.path.join(folder_path, "*.set"))
    print(f"📂 找到 {len(set_files)} 个 .set 文件")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_eegpt_encoder()

    for i, f in enumerate(set_files, 1):
        name = os.path.basename(f).replace(".set", "") # 保持干净的 subject 名字
        print(f"\n[{i}/{len(set_files)}] 正在处理：{name}")

        feat, crop_start = process_one_file(f, encoder, device)

        np.save(os.path.join(save_folder, f"{name}.npy"), feat)
        np.save(os.path.join(save_folder, f"{name}.crop_start.npy"), np.int64(crop_start))

        print(f"✅ 已保存特征：{name}.npy")

if __name__ == "__main__":
    # ⚠️ 请将这里的路径换成你刚刚做完 ICA、还没做 Norm 的数据文件夹
    YOUR_ICA_FOLDER = r"D:\1\mumtaz_1\train_6" 
    process_folder(YOUR_ICA_FOLDER)