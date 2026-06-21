# E-MDD MATLAB 预处理

本目录包含 **E-MDD workflow step0** 的 MATLAB 脚本，对应常见 EEG 预处理流程：

| 仓库内脚本 | 典型对应步骤 | 作用 |
|-----------|-------------|------|
| `emdd_epoch_6s.m` | epoch 切分 | 6s 滑窗切 epoch（50% 重叠） |
| `emdd_split_train_test.m` | train/test 划分 | 按被试 7:3 划分 train/test |
| `emdd_znorm.m` | Z-norm | 通道级 Z 归一化（统计量仅来自训练集） |
| `emdd_run_preprocess.m` | — | Python workflow 调用的总入口 |

**ICA + ICLabel 去伪迹** 可使用你本地的自定义 `.m` 脚本（在 `configs/emdd_local.yaml` 的 `matlab_preprocess.ica_script` 中配置）。

## 数据流

```
原始 BDF/EDF
    → [可选 ICA 脚本] → data/eeg/ica/*.set
    → emdd_epoch_6s → data/eeg/epoch_6/
    → emdd_split_train_test → data/eeg/train_6/ + data/eeg/test_6/
    → [emdd_znorm 可选] → data/eeg/train_norm_6/ + data/eeg/test_norm_6/
    → Python step1 (EEGPT 编码)
```

目录名与 `configs/emdd_default.yaml` 一致，可在 `emdd_local.yaml` 中覆盖。

## 启用 step0

1. 复制本地配置：

```bash
copy configs\emdd_local.yaml.example configs\emdd_local.yaml
```

2. 编辑 `configs/emdd_local.yaml`：

```yaml
matlab_preprocess:
  enabled: true
  executable: ""          # 留空自动查找；或填 MATLAB 安装路径
  run_ica: false          # 若已有 data/eeg/ica，保持 false
  ica_script: ""          # 可选：你的 ICA .m 脚本（相对或绝对路径）
  run_epoch_6s: true
  run_split: true
  run_znorm: false        # true 时请将 paths.raw_eeg 改为 train_norm_dir
```

3. 运行：

```bash
python emdd_workflow.py run --steps step0
python emdd_workflow.py run --from step1
```

## 依赖

- MATLAB R2020a+
- EEGLAB（含 `pop_loadset` / `pop_saveset`）
- 若 `run_ica: true`：还需 ICLabel、biosig 等 EEGLAB 插件

## 已有预处理结果？

若你**已有** `train_6/` 等 `.set` 文件，将其放入 `data/eeg/train_6/`（或在 `emdd_local.yaml` 指向你的目录），保持 `matlab_preprocess.enabled: false`，直接从 step1 开始即可。
