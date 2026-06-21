# Sample Outputs (Redacted)

> 以下为脱敏后的代表性结果，用于 portfolio / 招聘展示。完整 CSV 与 CoT 全文不在公开仓库中。

## External Test — EEGPT (canonical)

| 指标 | 值 |
|------|-----|
| AUC | 0.8465 |
| Epoch ACC | 0.7656 |
| Subject ACC | 0.8235 |
| Weighted F1 (epoch) | 0.7609 |
| Sensitivity | 0.7316 |
| Specificity | 0.8010 |
| n_epochs | 1702 |
| n_subjects | 17 |

混淆矩阵（epoch-level）：TN=668, FP=166, FN=233, TP=635

## Baseline Comparison (same external test)

| Model | AUC | Subject ACC |
|-------|-----|-------------|
| EEGPT + clinical features | 0.8465 | 0.8235 |
| EEGNet512 | 0.8940 | 0.7647 |
| EEGConformer512 | 0.8573 | 0.8235 |

## CoT Generation (step5)

- 抽样策略：test 集按 step3 预测标签分层抽样（默认 4 HC + 16 MDD）
- 输出字段：`llm_cot`, `top3_physicals`, `feature_alignment_rate`, `clinical_term_score`, `explanation_consistency`
- 断点续跑：已有 CoT 的 subject 自动复用

### CoT 片段示例（匿名化）

```
【一、样本整体判断】
本样本模型预测为 重度抑郁症(MDD)，概率为 0.9464。

【二、TOP3核心特征推理（依据脑电生理知识）】
1. 特征：alpha_asym
   - 数值状态判定：该特征实际数值为 -0.25，低于健康参考范围下限
   - 与模型预测 重度抑郁症(MDD) 的对照：一致。右额叶 α 优势符合 MDD 情绪调节异常模式。

【三、一致性结论】
综合 TOP3 特征，全局判定为 MDD 与 MDDprob、阈值及多特征协同相容。
```

## Workflow Run Manifest 示例

```json
{
  "started_at": "2026-04-20T08:30:00Z",
  "steps_requested": ["step3", "step4", "step5"],
  "step_results": [
    {"step": "step3", "status": "success", "elapsed_sec": 1823.4},
    {"step": "step4", "status": "success", "elapsed_sec": 945.2},
    {"step": "step5", "status": "success", "elapsed_sec": 2401.7}
  ]
}
```

## 数据来源

- 公开数据集：Mumtaz et al.（与原始论文相同的纳入/排除标准）
- 标签规则：文件名含 `MDD` → label=1，否则 label=0（非本项目内人工标注）
