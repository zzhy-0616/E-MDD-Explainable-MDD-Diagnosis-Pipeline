# 数据目录

预处理后的 EEG 与中间文件建议放在此目录下，与 `configs/emdd_default.yaml` 默认路径一致：

```
data/eeg/
├── ica/              # ICA 清洗后的 .set（step0 输入，可选）
├── epoch_6/          # 6s 滑窗切分结果
├── train_6/          # 训练集 .set → step1 默认输入
├── test_6/           # 测试集 .set
├── train_norm_6/     # Z-norm 训练集（可选）
└── test_norm_6/      # Z-norm 测试集（可选）
```

## 获取数据

1. 使用公开数据集 [Mumtaz et al.](https://figshare.com/)（遵循其许可条款）。
2. 运行 step0 MATLAB 预处理（见 [`preprocessing/matlab/README.md`](../preprocessing/matlab/README.md)），或放置你已有的 `train_6/` 结果。
3. 在 `configs/emdd_local.yaml` 中确认 `paths.raw_eeg` 指向正确的 `.set` 目录。

> 原始 BDF/EDF、完整特征 `.npy`、模型权重均不在公开仓库中。
