# Pipeline Architecture (E-MDD)

## 架构命名

- **E-MDD**：本项目的整体架构（Explainable MDD diagnosis）
- **EEGPT**：仅作为 step1 的预训练脑电**编码器**，输出 512 维表征
- **MDDClassificationHead**：E-MDD 的分类模块（step3）
- **EEGProjector**：E-MDD 的多模态桥接模块（step4，512→LLM hidden）
- **LLM CoT**：E-MDD 的可解释输出模块（step5）

## 设计目标

1. **可复现**：固定随机种子、被试级划分、运行日志归档
2. **可解释**：SHAP + PSD 物理映射 + LLM CoT 三层解释
3. **可编排**：workflow 统一配置、分步执行、依赖检查

## 数据流

```
Raw BDF/EDF
    → step0 MATLAB: [可选 ICA] → 6s epoch → train/test 划分 → [Z-norm]
    → data/eeg/train_6/*.set
    → step1 EEGPT encoder → 512-d .npy
    → step2 MNE PSD + asymmetry
    → step3 (5-fold CV) → step4 / step5 / eval
```

## 核心模块

### MDDClassificationHead (step3)

- 输入：512 维 EEGPT 特征
- 结构：`Linear(512→256) → BN → ReLU → Dropout → Linear(256→2)`
- 训练：加权交叉熵 + label smoothing
- 输出：`eegpt_mdd_prob`、SHAP 报告

### EEGProjector (step4/5)

- 输入：512 维 EEGPT 特征
- 结构：`Linear(512→1024) → GELU → Linear(1024→3584)`
- 训练：冻结 LLM，最小化 next-token loss（标签文本 MDD/HC）
- 输出：1 个 LLM 兼容 token embedding

### CoT Generator (step5)

- 输入：epoch_report + TOP3 SHAP 特征 + 投影 EEG token
- 输出：结构化 CoT（样本判断 / TOP3 推理 / 一致性结论）
- 评估：特征对齐率、术语准确率、解释一致性

## Workflow 层

| 模块 | 职责 |
|------|------|
| `configs/emdd_default.yaml` | E-MDD 默认路径与脚本映射（相对仓库根） |
| `configs/emdd_local.yaml` | 本机路径覆盖（不提交 Git，见 `.example`） |
| `workflow/config.py` | 加载与解析配置 |
| `workflow/artifacts.py` | 每步 input/output 契约 |
| `workflow/runner.py` | 执行、日志、symlink、subprocess |
| `workflow/adapters/` | step1/2 配置驱动适配 |

## 防泄露策略

- `StratifiedGroupKFold` 按 `real_subject_id` 分组
- 阈值在 train 折内搜索，应用到 val/test
- step3 完成后才复制对应 fold 的 `.npy` 到 train/val 目录

## 扩展

- 对照模型 baseline 脚本（开发树可选）：EEGNet512、EEGConformer512 pipeline
- 发布包仅包含 `emdd_core/` 与 `llm_explanation/` 主链路
