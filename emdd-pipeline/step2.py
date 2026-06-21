import os
import warnings
import numpy as np
import mne
import pandas as pd

warnings.filterwarnings("ignore")

# ⚠️ 请确保这里的路径与第一步的 YOUR_ICA_FOLDER 完全一致
data_path = ""  # set by workflow adapter 
features_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emdd_core", "features")
sfreq = 256
CROP_LEN = 1024  

freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
pairs = [('Fp1', 'Fp2'), ('F7', 'F8'), ('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2'), ('T3', 'T4'), ('T5', 'T6')]

def load_eeglab_cropped(path):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    raw.pick(picks="eeg", verbose=False)

    stem = os.path.splitext(os.path.basename(path))[0]
    crop_file = os.path.join(features_dir, f"{stem}.crop_start.npy")
    fs = float(raw.info["sfreq"])

    if os.path.isfile(crop_file):
        crop_start = int(np.load(crop_file))
    else:
        raise FileNotFoundError(f"❌ 找不到 {stem} 的裁剪对齐文件，请先运行 eeg_feature_extract.py")

    # ==========================================
    # 🌟 修复点：放弃 MNE 的时间裁剪，直接做 Numpy 切片
    # 与第一步提取深度特征时的矩阵操作保持 100% 像素级对齐
    # ==========================================
    data = raw.get_data() # 获取全长 6s 数据，形状 (19, 1536)
    data_cropped = data[:, crop_start : crop_start + CROP_LEN] # 切片出对应的 4s，形状 (19, 1024)

    # 预处理分支：只做基线校正，消除直流漂移，保留真实的微伏物理量级
    channel_means = np.mean(data_cropped, axis=1, keepdims=True)
    data_corrected = data_cropped - channel_means
    
    # 增加 batch 维度并转换为微伏
    data_microvolts = data_corrected[np.newaxis, :, :] * 1e6 
    
    return data_microvolts, raw.ch_names, fs
def get_ch_idx(ch_names, ch_name):
    try: return ch_names.index(ch_name)
    except ValueError: return None

def compute_asym(power, left_idx, right_idx):
    if left_idx is None or right_idx is None: return 0.0
    L, R = power[left_idx].mean(), power[right_idx].mean()
    if L + R < 1e-10: return 0.0
    return (L - R) / (L + R)

def process_one_subject(file):
    path = os.path.join(data_path, file)
    data, ch_names, fs = load_eeglab_cropped(path)

    band_val = {}
    for b, (fmin, fmax) in freq_bands.items():
        psd, _ = mne.time_frequency.psd_array_welch(data, fs, fmin, fmax, n_fft=256, verbose=False)
        band_val[b] = float(np.mean(psd))

    asym = {'delta_asym': [], 'theta_asym': [], 'alpha_asym': []}
    for ch_l, ch_r in pairs:
        il, ir = get_ch_idx(ch_names, ch_l), get_ch_idx(ch_names, ch_r)
        if il is None or ir is None: continue
        for b in ['delta', 'theta', 'alpha']:
            fmin, fmax = freq_bands[b]
            psd, _ = mne.time_frequency.psd_array_welch(data, fs, fmin, fmax, n_fft=256, verbose=False)
            ai = compute_asym(psd.mean(axis=0), il, ir)
            asym[f'{b}_asym'].append(ai)

    def mean_or_zero(lst): return np.mean(lst) if len(lst) > 0 else 0.0

    stem = file.replace('.set', '')
    return {
        'subject': stem,
        'group': 'MDD' if 'MDD' in file else 'Health',
        'label': 1 if 'MDD' in file else 0,
        **band_val,
        'delta_asym': mean_or_zero(asym['delta_asym']),
        'theta_asym': mean_or_zero(asym['theta_asym']),
        'alpha_asym': mean_or_zero(asym['alpha_asym']),
    }

if __name__ == "__main__":
    files = [f for f in os.listdir(data_path) if f.endswith('.set')]
    rows = [process_one_subject(f) for f in files]
    df = pd.DataFrame(rows)
    df.to_csv("subject_power_with_asym.csv", index=False, encoding='utf-8-sig')
    print("\n✅ MNE 物理特征计算完成并对齐：subject_power_with_asym.csv")